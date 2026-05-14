"""Online continual learning adapters for ACE.

Methods:
  - UniformCLAdapter   : Experience Replay (uniform buffer + MSE).  Baseline.
  - ACEAdapter         : ACE (smooth-EMA λ + L2 anchor + priority replay).  Ours.

Both follow the same self-supervised scheme:
  1. NN predicts action from (state, desired_next_state)
  2. Execute action, observe the next state
  3. Use the realised transition as a self-supervised inverse-dynamics label
  4. Push to buffer, sample a batch, one gradient step

The legacy class name ``ACELiteAdapter`` is kept as an alias for
backwards compatibility with older scripts/tests.
"""
import numpy as np
import torch
import torch.nn.functional as torchF
import copy

from models.evidential_net import evidential_loss, epistemic_score, aleatoric_score
from models.adaptive_evidential_net import AdaptiveEvidentialNet
from training.continual_methods import OnlineEWC
from controllers.nn_controller import (
    compute_desired_next_state as _unicycle_compute_desired,
    _clamp_input_to_training_range,
)
from controllers.state_feedback import StateFeedback, StateFeedback3D
from utils.buffers import RandomReplayBuffer, PriorityReplayBuffer


def _maybe_make_feedback(env, dt: float, feedback_gain: float):
    """Helper: build a StateFeedback if gain > 0 and the env supports it.

    Pass a *nominal* env (one with default mass / friction multipliers,
    captured at training time) to keep the feedback K matrix matched to
    the offline linearization — the comparison with LQR is fair when
    both have the same prior knowledge.

    Dispatches to :class:`StateFeedback` for the 2-D unicycle or
    :class:`StateFeedback3D` for the 3-D AUV / Drone, based on the env's
    advertised kind.  Returns ``None`` if the gain is zero or the env
    doesn't expose a recognisable physics interface.
    """
    if feedback_gain <= 0.0:
        return None
    # 3-D AUV / Drone: env exposes MASS + I_zz + the damping coefficients
    # consumed by LQR3D.  AUV is distinguished from Drone by the presence
    # of a buoyancy attribute ``F_BOUY``.
    if hasattr(env, "MASS") and hasattr(env, "I_zz"):
        kind = "auv" if hasattr(env, "F_BOUY") else "drone"
        return StateFeedback3D(env, kind=kind, dt=dt, gain=feedback_gain)
    # 2-D unicycle: env exposes effective_mass / effective_Iz / effective_friction
    needed = ("effective_mass", "effective_Iz", "effective_friction")
    if not all(hasattr(env, n) for n in needed):
        return None
    return StateFeedback(env, dt, gain=feedback_gain)


def _select_feedback_env(env, feedback_env):
    """Return ``feedback_env`` if provided, else fall back to ``env``."""
    return feedback_env if feedback_env is not None else env


# --------------------------------------------------------------------------
# Platform-agnostic helpers
# --------------------------------------------------------------------------

def _resolve_compute_desired(env, state, positions, step_idx, dt):
    """Dispatch to the env's own waypoint follower if present (AUV/Drone),
    otherwise fall back to the legacy unicycle helper.
    """
    if hasattr(env, "compute_desired_next_state"):
        return env.compute_desired_next_state(state, positions, step_idx, dt)
    return _unicycle_compute_desired(state, positions, step_idx, dt)


def _clip_action_for_env(env, action):
    """Clip action using env.ACTION_LIMITS if defined, else unicycle defaults."""
    lim = getattr(env, "ACTION_LIMITS", None)
    if lim is None:
        action[0] = float(np.clip(action[0], -20.0, 20.0))
        action[1] = float(np.clip(action[1], -10.0, 10.0))
    else:
        for i, (lo, hi) in enumerate(lim):
            action[i] = float(np.clip(action[i], lo, hi))
    return action


# Back-compat shim: legacy callers use the positional unicycle signature
def compute_desired_next_state(*args, **kwargs):
    return _unicycle_compute_desired(*args, **kwargs)


class UniformCLAdapter:
    """Continual Learning with uniform replay (baseline).

    Uses a circular FIFO buffer and uniform sampling.
    Learning rate warms up over the first warmup_steps to avoid
    catastrophic forgetting before the buffer has diversity.
    """

    def __init__(self, model, env, dt, device='cpu',
                 buffer_capacity=500, batch_size=16,
                 lr=5e-4, lambda_reg=0.01,
                 update_every=1, min_buffer_size=8,
                 feature_mode="full",
                 update_threshold=0.0,
                 feedback_gain: float = 0.0,
                 feedback_env=None):
        self.model = copy.deepcopy(model)
        self.model.to(device)
        self.model.eval()  # Keep in eval mode; switch to train only during updates
        self.device = device
        self.dt = dt
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.update_every = update_every
        self.min_buffer_size = min_buffer_size
        self.target_lr = lr
        self.feature_mode = getattr(model, 'feature_mode', feature_mode)
        self.feedback_gain = float(feedback_gain)
        self._feedback = _maybe_make_feedback(
            _select_feedback_env(env, feedback_env), dt, self.feedback_gain)

        self.buffer = RandomReplayBuffer(capacity=buffer_capacity)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Cache normalization stats as numpy and torch
        self.inp_mean = model.input_mean.to(device)
        self.inp_std = model.input_std.to(device)
        self.tgt_mean = model.target_mean.to(device)
        self.tgt_std = model.target_std.to(device)
        self.inp_mean_np = model.input_mean.cpu().numpy()
        self.inp_std_np = model.input_std.cpu().numpy()
        self.tgt_mean_np = model.target_mean.cpu().numpy()
        self.tgt_std_np = model.target_std.cpu().numpy()

        self.step_count = 0
        self.update_count = 0
        self.last_epistemic = 0.0
        self.last_aleatoric = 0.0
        self.update_threshold = update_threshold  # skip update when ep < threshold
        self._prev_state = None
        self._prev_action = None

    @property
    def info_str(self):
        return f"CL: buf={len(self.buffer)}/{self.buffer.capacity} updates={self.update_count}"

    def step(self, env, state, positions, step_idx):
        """One step: predict action, execute, collect from PREVIOUS step, optionally update."""

        # --- Collect from previous step (we now know state_after) ---
        if self._prev_state is not None and self._prev_action is not None:
            prev_state = self._prev_state
            cur_state = state  # this is the result of the previous action
            label_action = self._prev_action

            actual_input = env.build_inverse_dynamics_input(
                prev_state, cur_state, feature_mode=self.feature_mode)
            # Clamp to training range before normalizing
            actual_input = _clamp_input_to_training_range(
                actual_input, self.inp_mean_np, self.inp_std_np, k=3.0
            )
            actual_inp_norm = (actual_input - self.inp_mean_np) / self.inp_std_np
            actual_tgt_norm = (label_action - self.tgt_mean_np) / self.tgt_std_np
            self.buffer.push(actual_inp_norm, actual_tgt_norm)

        # --- Compute desired next state and NN action ---
        desired_next = _resolve_compute_desired(env, state, positions, step_idx, self.dt)

        nn_input = env.build_inverse_dynamics_input(
            state, desired_next, feature_mode=self.feature_mode)
        # Clamp to training range (±5σ to allow mild OOD for adaptation)
        nn_input = _clamp_input_to_training_range(
            nn_input, self.inp_mean_np, self.inp_std_np, k=5.0
        )
        inp_t = torch.FloatTensor(nn_input).unsqueeze(0).to(self.device)
        inp_norm = (inp_t - self.inp_mean) / self.inp_std

        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(inp_norm)
        action_norm = gamma.squeeze(0).cpu().numpy()
        action = action_norm * self.tgt_std_np + self.tgt_mean_np
        if self._feedback is not None:
            action = action + self._feedback.compute(state, positions, step_idx)
        _clip_action_for_env(env, action)
        ep_val = epistemic_score(nu, alpha, beta).item()
        al_val = aleatoric_score(nu, alpha, beta).item()

        # --- Execute ---
        self._prev_state = state.copy()
        env.step(action)
        # Always record the COMMANDED action (not actual torque).
        # Using last_actual_torque (= command + wind) as the label causes the
        # NN to learn to output (command + wind), which is then applied as the
        # new command → actual = (command + wind) + wind = command + 2×wind →
        # exponential blowup under any non-zero wind.  The correct inverse-
        # dynamics formulation trains the NN to predict the COMMAND needed to
        # produce an observed transition (the wind effect is implicit in the
        # transition features themselves and the NN learns to compensate).
        self._prev_action = action.copy()

        # --- Optionally update ---
        self.step_count += 1
        if (self.step_count % self.update_every == 0 and
                len(self.buffer) >= self.min_buffer_size and
                ep_val >= self.update_threshold):
            self._do_update()

        self.last_epistemic = ep_val
        self.last_aleatoric = al_val
        return action, {
            "epistemic": ep_val,
            "aleatoric": al_val,
            "desired_next": desired_next,
            "buffer_size": len(self.buffer),
        }

    def _do_update(self):
        """One gradient step on a uniformly sampled batch.

        Uses MSE loss on gamma (predicted mean) for stable online adaptation.
        Learning rate ramps up linearly over first 50 updates.
        """
        # Warmup: ramp lr from 10% to 100% over first 50 updates
        warmup_steps = 50
        if self.update_count < warmup_steps:
            warmup_frac = 0.1 + 0.9 * (self.update_count / warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.target_lr * warmup_frac

        inputs, targets = self.buffer.sample(self.batch_size)
        inp_t = torch.FloatTensor(inputs).to(self.device)
        tgt_t = torch.FloatTensor(targets).to(self.device)

        self.model.train()
        gamma, nu, alpha, beta = self.model(inp_t)
        # MSE on gamma only — stable, doesn't collapse beta
        loss = torchF.mse_loss(gamma, tgt_t)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()
        self.update_count += 1



# =============================================================================
# ACEAdapter — NIG + priority replay + smooth-EMA λ + L2 anchor.  Ours.
# =============================================================================

class ACEAdapter:
    """ACE: a minimal, theoretically-clean active continual learner.

    The method has **three** mechanisms:

      1. NIG evidential head (γ, ν, α, β) — gives epistemic uncertainty
         ε = β / (ν · (α-1)) in one forward pass.
      2. Priority replay buffer keyed by ε — informative samples (high ε)
         are replayed more often than already-known transitions.
      3. Smooth-EMA λ schedule on ε:
              λ_{t+1} = λ_t + η · (g(ε̄_t) − λ_t),
              g(ε)    = κ / (ε + κ),
         where ε̄_t is the running EMA of the *observation-time* epistemic
         uncertainty on actual transitions.  λ is then used as the weight
         on the evidence-regulariser inside the NIG loss
              L = NLL + λ_0 · λ_t · |y − γ| · (2ν + α) + α_mse · MSE.
         An optional L2 anchor κ_a‖θ − θ_0‖² (constant, **not** coupled
         to λ) keeps the model from drifting on Nominal.

    Six hyperparameters: ``lr, lambda_eta, lambda_kappa, anchor_strength,
    batch_size, buffer_capacity`` (the rest are infrastructure).

    K-feedback is supported with a constant gain (``feedback_gain``) so
    the comparison with LQR is fair across all NN methods, but the
    feedback is *not* part of the contribution — it is a control
    architecture choice shared by every method in the benchmark.
    """

    def __init__(self, model, env, dt, device="cpu",
                 # Method hyperparameters
                 lr: float = 1e-4,
                 lambda_eta: float = 0.10,
                 lambda_kappa: float = 0.05,
                 anchor_strength: float = 5e-3,
                 batch_size: int = 32,
                 buffer_capacity: int = 1000,
                 # Lower-bound floor on the schedule.  Aleatoric-aware
                 # adaptation (in step()) raises the *effective* floor
                 # automatically when running noise is high, so this
                 # static knob sets only the floor under low-noise
                 # conditions.  3-D platforms historically used 0.99 to
                 # collapse ACE to DER; with the aleatoric-aware floor
                 # this is no longer required (the floor self-tunes),
                 # but we keep the override for the strict ACE ≥ DER
                 # safety guarantee.
                 lambda_floor: float = 0.10,
                 # iter-30 opt-in flags. Defaults reproduce iter-29
                 # byte-for-byte; new behaviour engages only when a flag
                 # is set away from its default.
                 priority_clip: tuple[float, float] | None = None,
                 priority_alpha: float = 1.0,
                 schedule_mode: str = "ev_reg",
                 relative_eps: bool = False,
                 # iter-31 tempered-loss reformulation (off by default).
                 # ``"iter29"``  : L = NLL + lambda_0*lambda_k*R_ev + anchor
                 #                 (lambda_k modulates a tiny regulariser; weak lever).
                 # ``"tempered"``: L = (1-lambda_k)*(NLL+mse) + lambda_0*R_ev + anchor
                 #                 (lambda_k directly weights the data terms; Bayesian
                 #                  tempered-posterior interpretation).  When in this
                 #                  mode, ``schedule_mode`` is ignored because the
                 #                  schedule's lever lives in the data weight, not the
                 #                  anchor or the evidence regulariser.
                 loss_form: str = "iter29",
                 # Initial value of lambda_schedule.  Only useful in
                 # combination with ``lambda_eta=0`` (the ACE* fixed
                 # ablation) to control where the constant schedule sits.
                 # Default 1.0 matches iter-29 behaviour.
                 lambda_schedule_init: float = 1.0,
                 # iter-32 schedule-signal reformulation (off by default).
                 # ``"absolute"``       : iter-29/31 default.
                 #     lam_target = kappa / (eps_bar + kappa)
                 #     floor      = max(lambda_floor, al_bar / (al_bar + kappa))
                 #     Two competing terms (saturation map vs aleatoric floor)
                 #     with one shared scale knob ``lambda_kappa``.
                 # ``"aleatoric_share"``: iter-32 single-equation reformulation.
                 #     lam_target = al_bar / (eps_bar + al_bar + delta)
                 #     floor      = lambda_floor   (hard safety only)
                 #     The schedule weight is the aleatoric fraction of total
                 #     predictive uncertainty; cite Kendall & Gal (2017).  No
                 #     ``lambda_kappa`` knob (one fewer hyperparameter).
                 schedule_signal: str = "absolute",
                 # iter-33 noise-gate reformulation (off by default).
                 # ``"binary"``    : iter-29/31/32 default.  Skip the gradient
                 #     step entirely when ``al_bar_fast > kappa/20`` or jitter.
                 #     Discrete veto on every cell where aleatoric is high.
                 # ``"continuous"``: iter-33 multiplicative attenuation.
                 #     Always run _do_update (subject to warmup).  Inside the
                 #     update, scale the data-fit term by
                 #         noise_attn = 1 / (1 + max(r_fast, r_slow, r_jit)^4)
                 #     where r_fast = al_bar_fast / threshold,
                 #           r_slow = al_bar      / threshold,
                 #           r_jit  = obs_jitter  / jitter_threshold.
                 #     Quartic decay around the same thresholds the binary
                 #     gate uses, but smooth: lambda's value matters on every
                 #     cell, not just on cells where the binary gate happens
                 #     to be open.
                 noise_gate: str = "binary",
                 # iter-33 noise_gate_scale.  Multiplies the noise/jitter
                 # thresholds before computing the attenuation: smaller value
                 # => stricter gate (continuous attn collapses to zero faster).
                 # Default 1.0 keeps thresholds at iter-31 values.  Drone-like
                 # platforms with naturally high baseline aleatoric on the
                 # offline NN can pass e.g. 0.05 to make the gate effectively
                 # binary on noise cells while keeping it open elsewhere.
                 noise_gate_scale: float = 1.0,
                 # iter-34 sustained-bias detector ("shift mode").  Off by default.
                 # Tracks the EMA of the residual between the model's predicted
                 # action and the actually-applied action; when the SNR of this
                 # bias exceeds ``shift_threshold`` AND the running epistemic
                 # estimate is also elevated (eps_bar > lambda_kappa), ACE
                 # bypasses the binary noise gate so the schedule + data term
                 # can drive adaptation toward the new operating point.  This
                 # is the iter-34 mechanism that addresses cases where
                 # iter-31's noise gate would otherwise veto every update on a
                 # cell with a real but noise-shaped parametric shift (the
                 # AUV / MassShift pathology).  Adds one EMA and one threshold;
                 # all four core ACE mechanisms remain unchanged.
                 shift_detection: bool = False,
                 shift_threshold: float = 2.0,
                 # iter-35: when shift_mode is active, multiply the effective
                 # learning rate by ``shift_lr_boost`` for that single update
                 # step.  Only takes effect when shift_detection=True and the
                 # shift_mode gate also fires (low jitter, high eps_bar,
                 # statistically-significant residual bias).  Default 1.0 keeps
                 # iter-34 behaviour byte-identical.
                 shift_lr_boost: float = 1.0,
                 # Option E: when a sustained shift_mode period ends, copy the
                 # anchor weights theta_0 back into the model.  This forces
                 # rapid recovery to the offline solution after the shift is
                 # removed.  Default False keeps prior behaviour identical.
                 shift_anchor_reset: bool = False,
                 # Infrastructure (not method knobs)
                 lambda_0: float = 0.01,
                 mse_weight: float = 1.0,
                 update_every: int = 5,
                 min_buffer_size: int = 64,
                 feature_mode: str = "full",
                 feedback_gain: float = 0.0,
                 feedback_env=None):
        # Wrap into AdaptiveEvidentialNet so we can call ace_loss()
        if not isinstance(model, AdaptiveEvidentialNet):
            self.model = AdaptiveEvidentialNet.from_pretrained(model, lambda_tau=1.0)
        else:
            self.model = copy.deepcopy(model)
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.dt = dt
        self.lr = float(lr)
        self.lambda_0 = float(lambda_0)
        self.lambda_eta = float(lambda_eta)
        self.lambda_kappa = float(lambda_kappa)
        self.lambda_floor = float(lambda_floor)
        self.anchor_strength = float(anchor_strength)
        self.mse_weight = float(mse_weight)
        self.batch_size = int(batch_size)
        self.update_every = int(update_every)
        self.min_buffer_size = int(min_buffer_size)
        self.feature_mode = getattr(model, "feature_mode", feature_mode)

        # Feedback (optional, fixed gain; not part of the method)
        self.feedback_gain = float(feedback_gain)
        self._feedback = _maybe_make_feedback(
            _select_feedback_env(env, feedback_env), dt, self.feedback_gain)

        # iter-30 flag validation + storage. Done before allocating the
        # buffer so PriorityReplayBuffer's sampling_alpha lines up with
        # the user's choice on the first sample() call.
        if schedule_mode not in ("ev_reg", "anchor"):
            raise ValueError(
                f"schedule_mode must be 'ev_reg' or 'anchor', got {schedule_mode!r}")
        if loss_form not in ("iter29", "tempered"):
            raise ValueError(
                f"loss_form must be 'iter29' or 'tempered', got {loss_form!r}")
        if schedule_signal not in ("absolute", "aleatoric_share"):
            raise ValueError(
                f"schedule_signal must be 'absolute' or 'aleatoric_share', "
                f"got {schedule_signal!r}")
        if noise_gate not in ("binary", "continuous"):
            raise ValueError(
                f"noise_gate must be 'binary' or 'continuous', got {noise_gate!r}")
        if noise_gate_scale <= 0:
            raise ValueError(
                f"noise_gate_scale must be > 0, got {noise_gate_scale}")
        if shift_threshold <= 0:
            raise ValueError(
                f"shift_threshold must be > 0, got {shift_threshold}")
        if shift_lr_boost <= 0:
            raise ValueError(
                f"shift_lr_boost must be > 0, got {shift_lr_boost}")
        if not (0.0 <= lambda_schedule_init <= 1.0):
            raise ValueError(
                f"lambda_schedule_init must be in [0, 1], got {lambda_schedule_init}")
        if priority_clip is not None:
            lo, hi = priority_clip
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(
                    f"priority_clip must be (lo, hi) with lo<hi and both finite, got {priority_clip!r}")
        if priority_alpha <= 0:
            raise ValueError(
                f"priority_alpha must be > 0, got {priority_alpha}")
        self.priority_clip = priority_clip
        self.priority_alpha = float(priority_alpha)
        self.schedule_mode = schedule_mode
        self.relative_eps = bool(relative_eps)
        self.loss_form = loss_form
        self._lambda_schedule_init = float(lambda_schedule_init)
        self.schedule_signal = schedule_signal
        self.noise_gate = noise_gate
        self.noise_gate_scale = float(noise_gate_scale)
        self.shift_detection = bool(shift_detection)
        self.shift_threshold = float(shift_threshold)
        self.shift_lr_boost = float(shift_lr_boost)
        self.shift_anchor_reset = bool(shift_anchor_reset)
        self._prev_shift_mode = False
        self._shift_active_steps = 0
        self._shift_off_steps = 0
        self._shift_active_max = 0
        self._reset_done_this_episode = True  # no reset before any shift
        self._max_bias_in_episode = 0.0

        # Buffer + optimizer
        self.buffer = PriorityReplayBuffer(
            capacity=int(buffer_capacity),
            sampling_alpha=self.priority_alpha,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Anchor snapshot for the optional L2 term
        self._anchor = {n: p.detach().clone()
                        for n, p in self.model.named_parameters()
                        if p.requires_grad}

        # Normalisation stats cached for speed
        self.inp_mean    = self.model.input_mean.to(device)
        self.inp_std     = self.model.input_std.to(device)
        self.tgt_mean    = self.model.target_mean.to(device)
        self.tgt_std     = self.model.target_std.to(device)
        self.inp_mean_np = self.model.input_mean.cpu().numpy()
        self.inp_std_np  = self.model.input_std.cpu().numpy()
        self.tgt_mean_np = self.model.target_mean.cpu().numpy()
        self.tgt_std_np  = self.model.target_std.cpu().numpy()

        # State
        self.lambda_schedule = self._lambda_schedule_init  # iter-29: 1.0 ("preserve")
        self.eps_bar = 0.0               # EMA of observation-time epistemic
        self.al_bar = 0.0                # EMA of observation-time aleatoric
        self.al_bar_fast = 0.0           # fast EMA for the noise update gate
        self.eps_bar_eta = 0.05          # fixed; not a method knob
        # iter-30 baseline tracking for relative_eps. Captured at the
        # end of the warmup window (`step_count == warmup_steps`),
        # which is when the noise gate releases anyway. Stays None on
        # default `relative_eps=False` so iter-29 traces are byte-
        # identical.
        self.eps_baseline: float | None = None
        # Model-free observation-jitter detector for sensor noise.
        # The aleatoric β output of the offline-trained NIG head is
        # not large on the first noisy inputs (it was trained on
        # clean data), so it lags the actual noise level by ~150
        # control steps — long enough that ~30 gradient updates
        # corrupt the predictive mean before the model recognises
        # the regime.  We supplement the model-based al_bar gate
        # with a model-free statistical check on the observed
        # second-difference of position; if the high-frequency
        # content of the state stream rises above a threshold the
        # observations are noisy and we skip the gradient step.
        self._obs_history: list = []
        self.obs_jitter = 0.0
        # iter-33 continuous noise gate: per-step attenuation factor
        # written by step() and consumed by _do_update(); 1.0 in the
        # binary-gate default so iter-29/31 behaviour is unchanged.
        self._noise_attn = 1.0
        # iter-34 sustained-bias detector.  EMA of the residual between
        # the model's predicted action (re-evaluated at the realized
        # transition) and the actually-applied action, per dim.  Lazily
        # sized on first update so it works for any action dimension.
        self._residual_bias: np.ndarray | None = None
        self._residual_bias_eta = 0.05  # slow EMA: matches eps_bar_eta
        self.shift_signal = 0.0          # exposed for diagnostics
        self._shift_mode = False         # exposed for diagnostics
        self.step_count = 0
        self.update_count = 0
        self.last_epistemic = 0.0
        self.last_aleatoric = 0.0
        self._prev_state = None
        self._prev_action = None

    @property
    def info_str(self):
        return (f"ACE: buf={len(self.buffer)}/{self.buffer.capacity} "
                f"upd={self.update_count} λ={self.lambda_schedule:.3f} "
                f"ε̄={self.eps_bar:.4f}")

    def step(self, env, state, positions, step_idx):
        """One control step: predict → execute → buffer → update."""

        # ── Track observation jitter (model-free noise detector) ──
        # Keep a short ring buffer of recent observations and compute
        # the std of the second-difference of (x, y) position.  Under
        # smooth motion this is ~0; under sensor noise σ_pos the
        # second-difference std grows to ≈ σ_pos · sqrt(6).
        self._obs_history.append(np.asarray(state, dtype=np.float64).copy())
        if len(self._obs_history) > 10:
            self._obs_history.pop(0)
        if len(self._obs_history) >= 5:
            arr = np.array(self._obs_history)
            second_diff = np.diff(arr[:, :2], n=2, axis=0)
            self.obs_jitter = float(np.std(second_diff))
        else:
            self.obs_jitter = 0.0

        # ── Buffer the *previous* transition and update λ schedule ──
        if self._prev_state is not None and self._prev_action is not None:
            actual_input = env.build_inverse_dynamics_input(
                self._prev_state, state, feature_mode=self.feature_mode)
            actual_input = _clamp_input_to_training_range(
                actual_input, self.inp_mean_np, self.inp_std_np, k=3.0)
            inp_norm = (actual_input - self.inp_mean_np) / self.inp_std_np
            tgt_norm = (self._prev_action - self.tgt_mean_np) / self.tgt_std_np

            with torch.no_grad():
                inp_t = torch.FloatTensor(inp_norm).unsqueeze(0).to(self.device)
                gamma_s, nu_s, alpha_s, beta_s = self.model(inp_t)
            ep_obs = self.model.epistemic_score(nu_s, alpha_s, beta_s).item()
            al_obs = self.model.aleatoric_score(nu_s, alpha_s, beta_s).item()
            # iter-34: compute the residual between the inverse-dynamics
            # prediction at the realized transition and the action that
            # was actually applied.  Under transient noise this residual
            # averages to zero; under a sustained parametric shift the
            # residual carries a directional bias that the EMA below picks
            # up over a few seconds.  Done unconditionally so iter-31
            # diagnostics are available even when shift_detection=False.
            predicted_norm = gamma_s.squeeze(0).cpu().numpy()
            residual_norm = predicted_norm - tgt_norm  # both are normalised
            if self._residual_bias is None:
                self._residual_bias = np.zeros_like(residual_norm)
            self._residual_bias = (
                (1.0 - self._residual_bias_eta) * self._residual_bias
                + self._residual_bias_eta * residual_norm
            )

            # Noise-discounted priority: ε / (1 + aleatoric).  The raw
            # epistemic ε = β/(ν(α-1)) rises with both ignorance *and*
            # data noise, so under disturbance / actuator noise a buffer
            # keyed on ε alone fills with high-priority noisy
            # transitions that average to zero in expectation.
            # Dividing by (1 + β/(α-1)) cancels the β contribution and
            # keeps the buffer focused on transitions that genuinely
            # signal model ignorance.  γ = 1 fixed (no new
            # hyperparameter).
            priority = ep_obs / (1.0 + al_obs)
            # iter-30 priority clip: clamp the priority before pushing
            # so a single rare high-ε transition cannot dominate the
            # buffer for the rest of the rollout.  None ⇒ no-op
            # (iter-29 byte-identical).
            if self.priority_clip is not None:
                lo, hi = self.priority_clip
                priority = float(np.clip(priority, lo, hi))
            self.buffer.push(inp_norm, tgt_norm, priority=priority)

            # λ schedule — drives toward g(ε̄) but is clipped from below
            # by an aleatoric-aware floor.  When data noise (aleatoric)
            # is high, the effective floor rises so the regulariser
            # stays strong and the optimiser does not over-adapt to
            # noisy targets.  When aleatoric is low (genuine
            # parametric shift), the floor is the user-set
            # ``lambda_floor`` and the schedule can fade for
            # adaptation.  This keeps the method's response self-
            # tuning to the *kind* of distribution shift without
            # adding a hyperparameter.
            self.eps_bar = ((1.0 - self.eps_bar_eta) * self.eps_bar
                            + self.eps_bar_eta * ep_obs)
            self.al_bar = ((1.0 - self.eps_bar_eta) * self.al_bar
                            + self.eps_bar_eta * al_obs)
            # Fast aleatoric EMA for the noise gate.  The slow EMA
            # ``al_bar`` has eta = 0.05 and lags spikes by ~50 steps;
            # the gate needs to fire faster so it catches the first
            # noisy update.  Use a stiffer η = 0.30 (~3-step
            # response) for the gate variable only — does not
            # affect the schedule's aleatoric-aware floor.
            al_eta_fast = 0.30
            self.al_bar_fast = ((1.0 - al_eta_fast) * self.al_bar_fast
                                + al_eta_fast * al_obs)
            if self.schedule_signal == "aleatoric_share":
                # iter-32 reformulation.  Two-part target:
                #   lam_share = al_bar / (eps_bar + al_bar + 1e-6)
                #   lam_sat   = kappa / (eps_bar + kappa)
                #   lam_target = max(lam_share, lam_sat)
                # Interpretation: the schedule weight is the larger of
                # (a) the aleatoric fraction of total predictive
                # uncertainty \citep{kendall2017uncertainties} and
                # (b) a soft "no-adapt-when-epistemic-is-small" prior.
                # The max is a safety choice: adapt only when both
                # signals say it is appropriate (high epistemic in
                # absolute scale AND high epistemic share of the total
                # uncertainty).  On noise cells (al_bar ≈ eps_bar both
                # rise), share is ~0.5 but sat is ~1, so max keeps
                # preserve mode.  On Disturbance (eps_bar large, al_bar
                # small), share is small AND sat is small, so max =
                # small → adapt.
                denom = self.eps_bar + self.al_bar + 1e-6
                lam_share = self.al_bar / denom if denom > 0 else 1.0
                lam_sat = self.lambda_kappa / (
                    self.eps_bar + self.lambda_kappa)
                lam_target = max(float(lam_share), float(lam_sat))
                self.lambda_schedule += self.lambda_eta * (
                    lam_target - self.lambda_schedule)
                self.lambda_schedule = float(np.clip(
                    self.lambda_schedule, self.lambda_floor, 1.0))
            else:
                # iter-29/31 default: saturation map + aleatoric-aware floor.
                # iter-30 relative-ε normalisation (off by default) optionally
                # divides eps_bar by a per-rollout baseline before saturation.
                eps_for_target = self.eps_bar
                if self.relative_eps:
                    if (self.eps_baseline is None
                            and self.step_count >= 64        # warmup_steps
                            and self.eps_bar > 1e-9):
                        self.eps_baseline = float(self.eps_bar)
                    if self.eps_baseline is not None:
                        eps_for_target = self.eps_bar / max(
                            self.eps_baseline, 1e-9)
                lam_target = self.lambda_kappa / (
                    eps_for_target + self.lambda_kappa)
                self.lambda_schedule += self.lambda_eta * (
                    lam_target - self.lambda_schedule)
                a_kappa = self.lambda_kappa
                al_floor = self.al_bar / (self.al_bar + a_kappa)
                eff_floor = max(self.lambda_floor, al_floor)
                self.lambda_schedule = float(np.clip(
                    self.lambda_schedule, eff_floor, 1.0))

        # ── Predict action ──
        desired_next = _resolve_compute_desired(env, state, positions,
                                                step_idx, self.dt)
        nn_input = env.build_inverse_dynamics_input(
            state, desired_next, feature_mode=self.feature_mode)
        nn_input = _clamp_input_to_training_range(
            nn_input, self.inp_mean_np, self.inp_std_np, k=5.0)
        inp_t = torch.FloatTensor(nn_input).unsqueeze(0).to(self.device)
        inp_norm_t = (inp_t - self.inp_mean) / self.inp_std

        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(inp_norm_t)
        action_norm = gamma.squeeze(0).cpu().numpy()
        action = action_norm * self.tgt_std_np + self.tgt_mean_np

        ep_val = self.model.epistemic_score(nu, alpha, beta).item()
        al_val = self.model.aleatoric_score(nu, alpha, beta).item()

        # Optional fixed-gain K-feedback (not part of the method).
        if self._feedback is not None:
            action = action + self._feedback.compute(state, positions, step_idx)
        _clip_action_for_env(env, action)

        # Execute.  We store the *commanded* (FF + FB) action as the
        # inverse-dynamics target: the dynamics saw this combined
        # action, so it is what the inverse-dynamics map should
        # reproduce on a (state_{t-1}, state_t) pair.  This matches
        # the convention used by every other adapter in the codebase.
        self._prev_state = state.copy()
        env.step(action)
        self._prev_action = action.copy()

        # ── Update ──
        # Iter-23 noise-aware update gate.  Two complementary checks:
        # (1) Warmup: no updates for the first ~4 s so the aleatoric
        #     EMA has time to populate.  Without this, sensor-noise
        #     rollouts would corrupt the model with the first 30-40
        #     gradient steps before ``al_bar_fast`` rises above the
        #     gate threshold (the model's β output lags the actual
        #     noise level by ~150 control steps because the offline
        #     network was trained on clean data).
        # (2) Aleatoric gate: once warm, skip the step if either the
        #     fast-EMA aleatoric or the slow-EMA exceeds the
        #     schedule's saturation knee ``κ_λ``.  Reuses
        #     ``lambda_kappa`` — no new hyperparameter.
        # The model-free observation-jitter detector responds in <10
        # control steps; the model-based aleatoric EMAs catch up
        # later.  We use both checks: jitter alone catches sensor
        # noise immediately; the EMAs catch slower noise build-up
        # (e.g. actuator-noise drift) that doesn't show as
        # second-difference spikes.  A short warmup keeps the buffer
        # filling before any gradient step.
        warmup_steps = 64    # match min_buffer_size — first update lands as before
        noise_threshold = self.lambda_kappa / 20.0
        # Jitter threshold: above 0.01 m on second-diff std means
        # σ_pos > ~0.004 m, i.e. a sensor-noise regime we want to
        # gate.  Nominal smooth motion sits at < 0.001 m.
        jitter_threshold = 0.01
        update_blocked_by_warmup = self.step_count < warmup_steps
        # iter-34 sustained-bias detector.  Compute the SNR of the
        # residual bias (||r̄|| / sqrt(ā)).  Shift mode requires both
        # (a) the bias is statistically significant above the running
        # aleatoric scale, and (b) the running epistemic estimate is
        # also elevated.  Only active when shift_detection is on; the
        # diagnostics ``self.shift_signal`` and ``self._shift_mode``
        # are populated either way for traceability.
        if self._residual_bias is not None and self.al_bar > 0:
            self.shift_signal = float(
                np.linalg.norm(self._residual_bias) / (np.sqrt(self.al_bar) + 1e-6))
        else:
            self.shift_signal = 0.0
        # iter-35 false-positive gate (jitter-only).  shift_mode requires
        # statistically-significant residual bias AND elevated epistemic
        # AND clean observations (obs_jitter below the iter-29 jitter
        # threshold).  The slow-aleatoric check from a first attempt was
        # removed because 3-D platforms have naturally high baseline
        # aleatoric on the offline NN, which would have re-blocked the
        # AUV/MassShift adaptation that iter-34 unlocked.  Jitter alone
        # cleanly separates sensor-noise cells (high jitter) from
        # parametric-shift cells (low jitter, even on 3-D platforms).
        _jitter_thr_inline = 0.01
        self._shift_mode = (
            self.shift_detection
            and self.shift_signal > self.shift_threshold
            and self.eps_bar > self.lambda_kappa
            and self.obs_jitter < _jitter_thr_inline
        )
        # Option E: anchor reset on TRUE shift removal.  Distinguishing
        # "shift removed" from "ACE adapted to it" requires looking at
        # absolute residual-bias magnitude rather than the threshold-
        # crossing of shift_mode.  Under sustained adaptation, ACE may
        # drive shift_signal below shift_threshold but the residual bias
        # itself is still non-trivial.  When the dynamics actually revert
        # to nominal (the learn-unlearn recovery phase), the residual
        # bias collapses toward zero.  We trigger reset only when the
        # bias EMA has been small in absolute terms (< 0.25 of the prior
        # max) for >= 100 consecutive steps, after a sustained shift
        # episode of >= 100 steps.  No-op if shift_anchor_reset=False.
        SHIFT_HOLD = 100
        OFF_HOLD   = 100
        bias_norm = (np.linalg.norm(self._residual_bias)
                     if self._residual_bias is not None else 0.0)
        if self._shift_mode:
            self._shift_active_steps += 1
            self._shift_active_max = max(self._shift_active_max,
                                          self._shift_active_steps)
            self._shift_off_steps = 0
            self._max_bias_in_episode = max(
                getattr(self, '_max_bias_in_episode', 0.0), bias_norm)
            if self._shift_active_steps >= SHIFT_HOLD:
                self._reset_done_this_episode = False
        else:
            self._shift_active_steps = 0
            # Off-counter advances only when residual bias is truly small
            max_bias = getattr(self, '_max_bias_in_episode', 0.0)
            if max_bias > 0 and bias_norm < 0.25 * max_bias:
                self._shift_off_steps += 1
            else:
                self._shift_off_steps = 0

        if (self.shift_anchor_reset
                and not self._reset_done_this_episode
                and self._shift_active_max >= SHIFT_HOLD
                and self._shift_off_steps >= OFF_HOLD):
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if p.requires_grad and name in self._anchor:
                        p.copy_(self._anchor[name])
            for pg in self.optimizer.param_groups:
                for p in pg["params"]:
                    if p in self.optimizer.state:
                        del self.optimizer.state[p]
            self._reset_done_this_episode = True
            self._shift_active_max = 0
        self._prev_shift_mode = self._shift_mode

        if self.noise_gate == "continuous":
            # iter-33: hybrid noise gate.
            #   ratio_max > BINARY_CUTOFF (5x threshold) => hard skip (same as binary).
            #   ratio_max < BINARY_CUTOFF              => continuous quartic attenuation.
            # Rationale: continuous attenuation alone leaks under sustained
            # high noise because even small attn values accumulate over
            # hundreds of updates; the binary cutoff at 5x threshold (where
            # the original binary gate would have skipped anyway) restores
            # the no-drift guarantee for heavy-noise cells while keeping
            # smooth adaptation in the transition region around 1x threshold.
            BINARY_CUTOFF = 5.0
            scale = max(self.noise_gate_scale, 1e-9)
            thr_noise = noise_threshold * scale
            thr_jitter = jitter_threshold * scale
            ratio_al_fast = self.al_bar_fast / max(thr_noise, 1e-9)
            ratio_al_slow = self.al_bar      / max(thr_noise, 1e-9)
            ratio_jit     = self.obs_jitter  / max(thr_jitter, 1e-9)
            ratio_max = max(ratio_al_fast, ratio_al_slow, ratio_jit)
            if ratio_max > BINARY_CUTOFF:
                # Heavy noise => binary skip.  attn=0 still triggers the
                # early-exit in _do_update.
                self._noise_attn = 0.0
                update_blocked_by_noise = True
            else:
                self._noise_attn = float(1.0 / (1.0 + ratio_max ** 4))
                update_blocked_by_noise = False
        else:
            self._noise_attn = 1.0
            update_blocked_by_noise = (
                self.al_bar_fast > noise_threshold
                or self.al_bar      > noise_threshold
                or self.obs_jitter  > jitter_threshold
            )
        # iter-34: shift_mode overrides any noise-based veto.  When the
        # residual bias is statistically significant AND epistemic is
        # elevated, the cell is showing a sustained parametric shift
        # rather than transient noise; let the schedule + data term act.
        # The anchor stays active so the bypass is bounded.
        if self._shift_mode:
            update_blocked_by_noise = False
            # Also force noise_attn to 1 so the data term has full weight
            # if the continuous gate would have attenuated it.
            self._noise_attn = 1.0
        self.step_count += 1
        if (self.step_count % self.update_every == 0
                and len(self.buffer) >= self.min_buffer_size
                and not update_blocked_by_warmup
                and not update_blocked_by_noise):
            self._do_update()

        self.last_epistemic = ep_val
        self.last_aleatoric = al_val
        return action, {
            "epistemic":       ep_val,
            "aleatoric":       al_val,
            "eps_bar":         self.eps_bar,
            "lambda_schedule": self.lambda_schedule,
            "desired_next":    desired_next,
            "buffer_size":     len(self.buffer),
        }

    def _do_update(self):
        """One gradient step on the ACE loss, plus optional L2 anchor.

        Iter-23 noise-aware MSE gate: scale ``mse_weight`` by
        ``1 − a / (a + κ_λ)`` where ``a = self.al_bar`` is the
        running aleatoric.  Under genuine parametric shift the
        observation-time aleatoric stays small and the gate ≈ 1, so
        adaptation proceeds at full strength.  Under measurement /
        actuator noise the aleatoric rises, the gate fades toward 0,
        and the MSE-on-γ term that would otherwise pull the
        predictive mean toward noisy targets is suppressed.  The
        NIG NLL term remains active and absorbs the noise via the
        aleatoric β channel rather than the action mean γ.  Reuses
        ``lambda_kappa`` as the saturation knee — no new hyperparameter.
        """
        buf_inputs, buf_targets = self.buffer.sample(self.batch_size)
        inp_t = torch.FloatTensor(buf_inputs).to(self.device)
        tgt_t = torch.FloatTensor(buf_targets).to(self.device)

        a_kappa = self.lambda_kappa
        al_floor = self.al_bar / (self.al_bar + a_kappa)
        mse_gate = max(0.0, 1.0 - float(al_floor))

        # iter-30 / iter-31 loss-form dispatch.
        #
        # iter-29 (loss_form="iter29"): the schedule modulates the evidence
        #   regulariser, optionally modulates the anchor (schedule_mode="anchor").
        #   Both have small leverage relative to the NIG NLL.
        #
        # iter-31 (loss_form="tempered"): tempered Bayesian formulation
        #     L = (1-lambda_k) * (NLL + mse_weight*mse) + lambda_0*R_ev + anchor.
        #   The schedule directly weights the data-fitting terms.  At
        #   lambda=1 the model only feels the anchor (preserve mode); at
        #   lambda=0 it sees the full data loss with the constant anchor as
        #   soft regulariser.  schedule_mode is ignored in this branch.
        # iter-33: continuous noise attenuation produced by step().  In
        # the binary-gate default this is always 1.0; in the continuous
        # gate it is the quartic-decay attenuation from step().  Scales
        # the data-fit term in every loss branch.
        noise_attn = float(self._noise_attn)
        # Continuous-gate early exit.  When attenuation is small enough that
        # cumulative drift across the rollout would be larger than the
        # binary-gate baseline allows, treat the update as a no-op.  We
        # return BEFORE building the loss because the evidence regulariser
        # R_ev = |y - mu| * (2*nu + alpha) has a data-dependent gradient
        # even with noise_attn = 0 multiplying the NLL/MSE, so the leak
        # would otherwise let the model drift toward the noisy target.
        # Threshold 1e-2 corresponds to attn at ~3x the noise threshold
        # under the quartic decay 1/(1+r^4): at r=3 attn=1/82~=1.2e-2.
        # Above 3x threshold the binary gate would have skipped anyway,
        # so this preserves binary's "no update" semantics in the heavy-noise
        # regime while keeping continuous behaviour in the transition zone
        # around 1x threshold.
        if noise_attn < 1e-2:
            self.update_count += 1
            return

        self.model.train()
        gamma, nu, alpha, beta = self.model(inp_t)
        if self.loss_form == "tempered":
            # Tempered branch.  We bypass ace_loss to keep control over
            # which terms the (1-lambda) factor multiplies.
            nll = self.model._nig_nll(tgt_t, gamma, nu, alpha, beta)
            reg = torch.abs(tgt_t - gamma) * (2.0 * nu + alpha)
            mse = torchF.smooth_l1_loss(gamma, tgt_t, beta=1.0)
            data_term = nll.mean() + (self.mse_weight * mse_gate) * mse
            evreg_term = self.lambda_0 * reg.mean()
            lam = float(self.lambda_schedule)
            # The data weight = (1 - lambda) * noise_attn.  Both factors
            # are in [0, 1] and act multiplicatively: the schedule
            # decides preserve-vs-adapt based on epistemic uncertainty,
            # the noise attenuation decides trust-the-batch based on
            # aleatoric uncertainty.  Anchor and evidence regulariser
            # are NOT attenuated -- they keep the model calibrated
            # regardless of the batch's noise level.
            loss = (1.0 - lam) * noise_attn * data_term + evreg_term
            anchor_scale = 1.0
        elif self.schedule_mode == "anchor":
            lambda_for_evreg = 1.0
            anchor_scale = float(self.lambda_schedule)
            loss, _ = self.model.ace_loss(
                gamma, nu, alpha, beta, tgt_t,
                lambda_0=self.lambda_0,
                lambda_schedule=lambda_for_evreg,
                mse_weight=noise_attn * self.mse_weight * mse_gate,
                robust=True)
        else:  # iter-29 default
            lambda_for_evreg = self.lambda_schedule
            anchor_scale = 1.0
            loss, _ = self.model.ace_loss(
                gamma, nu, alpha, beta, tgt_t,
                lambda_0=self.lambda_0,
                lambda_schedule=lambda_for_evreg,
                mse_weight=noise_attn * self.mse_weight * mse_gate,
                robust=True)

        if self.anchor_strength > 0.0:
            anchor_pen = 0.0
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                ref = self._anchor.get(name)
                if ref is None:
                    continue
                anchor_pen = anchor_pen + ((p - ref) ** 2).sum()
            loss = loss + 0.5 * self.anchor_strength * anchor_scale * anchor_pen

        if not torch.isfinite(loss):
            self.model.eval()
            self.update_count += 1
            return

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # iter-35 Phase 2: temporary lr boost during shift_mode.  When the
        # sustained-bias detector fires (shift_mode True), the cell is showing
        # a real parametric shift and the binary noise gate has already been
        # bypassed; multiplying lr by shift_lr_boost lets Adam take a single
        # larger step toward the new operating point.  The constant L2 anchor
        # still bounds drift.  Default boost=1.0 leaves iter-34 unchanged.
        boosted = self._shift_mode and self.shift_lr_boost != 1.0
        if boosted:
            # iter-35 Phase 2: aleatoric-aware boost taper.
            # full boost when al_bar is small (genuine parametric shift, the
            # AUV/MassShift case); automatically tapers toward 1 when al_bar
            # rises (noisy cells where shift_mode false-positives could be
            # amplified).  Saturation knee reuses lambda_kappa, no new
            # hyperparameter.  At al_bar=0 -> boost=shift_lr_boost; at
            # al_bar=lambda_kappa -> boost=1 + (shift_lr_boost-1)/2; at
            # al_bar -> infinity -> boost -> 1.
            taper = self.lambda_kappa / (self.al_bar + self.lambda_kappa)
            eff_boost = 1.0 + (self.shift_lr_boost - 1.0) * float(taper)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr * eff_boost
        self.optimizer.step()
        if boosted:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr
        self.model.eval()
        self.update_count += 1


# Backwards-compat alias for older scripts/tests.
ACELiteAdapter = ACEAdapter
