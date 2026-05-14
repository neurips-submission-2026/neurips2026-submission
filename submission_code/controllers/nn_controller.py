"""Neural network inverse dynamics controller.

Architecture:
    The NN predicts u[k] = f_theta(I[k]) where I[k] encodes the transition
    from current state x[k] to desired next state x*[k+1].

    The desired next state is computed from the reference trajectory using
    the SAME finite-difference scheme as the LQR's reference computation
    (see utils/unicycle_controller.py). This ensures consistency: the NN
    sees inputs at inference that are similar to what it saw during training.

    All clipping/clamping is derived from training data statistics (input_mean,
    input_std) stored in the model, so this generalizes across platforms.
"""

import numpy as np
import torch
from models.evidential_net import epistemic_score, aleatoric_score


def compute_desired_next_state(state, positions, step, dt):
    """Compute desired next state as a one-step correction toward the reference.

    Combines:
    - Nearest-point progress tracking (prevents step-counter drift from causing loops)
    - Error-proportional lookahead (no overshoot near the reference)
    - Smooth position-error correction (no dead-zone discontinuity)
    - Damped velocity/omega blending (gentler response when tracking well)
    - Cross-track + along-track feedback that scales the desired
      one-step displacement so the inverse-dynamics NN sees a transition
      that explicitly *reduces* tracking error on the next step.  This is
      the lightweight feedback channel that lets purely-feedforward NN
      controllers compensate for sustained disturbances (wind, mass).

    The desired next state is always a small delta from the current state,
    keeping the NN input in-distribution with training data.

    Parameters
    ----------
    state : array, shape (5,)
        Current state [x, y, theta, v, omega].
    positions : array, shape (N, 2)
        Reference trajectory waypoints.
    step : int
        Current time step (used to bound the search window only).
    dt : float
        Timestep.

    Returns
    -------
    desired_next : array, shape (5,)
        Desired next state [x, y, theta, v, omega].
    """
    n = len(positions)

    # --- Nearest-point progress tracking ---
    # The step counter advances in real time even when the robot loops/drifts.
    # If we use step directly as the trajectory index, the reference point drifts
    # ahead of the robot, creating corrections that point in the wrong direction
    # and cause oscillating loops.  Instead, find the spatially nearest trajectory
    # point within ±80 steps of the step counter (covers ~1.6s of slowdown at
    # dt=0.02).
    search_lo = max(0, step - 80)
    search_hi = min(n - 2, step + 20)
    if search_lo > search_hi:
        search_lo = max(0, n - 2)
        search_hi = n - 2
    window = positions[search_lo:search_hi + 1]
    dists = np.linalg.norm(window - state[:2], axis=1)
    i_nearest = search_lo + int(np.argmin(dists))
    raw_err = float(dists[i_nearest - search_lo])

    # --- Adaptive lookahead ---
    # A fixed +3 lookahead continually asks the NN to steer toward a point
    # ~3 steps ahead.  When the robot catches up, the nearest-point jumps
    # forward and the lookahead target jumps with it, creating a limit cycle
    # of overshoot-and-correct.  Scale lookahead with tracking error so it
    # vanishes when the robot is on-reference.
    lookahead = int(np.clip(raw_err / 0.05, 0, 3))
    i  = min(i_nearest + lookahead, n - 2)
    i1 = min(i + 1, n - 1)
    i2 = min(i + 2, n - 1)

    # --- Reference velocity and heading ---
    dx = positions[i1][0] - positions[i][0]
    dy = positions[i1][1] - positions[i][1]
    ref_theta = np.arctan2(dy, dx)
    ref_v = np.sqrt(dx**2 + dy**2) / dt

    dx2 = positions[i2][0] - positions[i1][0]
    dy2 = positions[i2][1] - positions[i1][1]
    ref_theta_next = np.arctan2(dy2, dx2)
    dtheta_ref = (ref_theta_next - ref_theta + np.pi) % (2 * np.pi) - np.pi
    ref_omega = dtheta_ref / dt

    # --- Position-error correction (smooth, no dead-zone) ---
    ex = positions[i][0] - state[0]
    ey = positions[i][1] - state[1]
    dist_err = np.sqrt(ex**2 + ey**2)

    if dist_err > 1e-8:
        angle_to_ref = np.arctan2(ey, ex)
        heading_diff = (angle_to_ref - ref_theta + np.pi) % (2 * np.pi) - np.pi
    else:
        heading_diff = 0.0
    # Smooth ramp in [0, 0.8] — zero when on-reference, saturates to 0.8
    # around 30 cm off.  Replaces the previous hard `if dist_err > 0.01`
    # dead-zone that caused a target_theta discontinuity.
    w_dev = 0.8 * np.tanh(3.0 * dist_err)
    target_theta = ref_theta + w_dev * heading_diff

    # --- Heading correction → desired omega ---
    # Saturate the CORRECTION term with tanh so drift-induced heading errors
    # don't push omega OOD.  Only the error-correction additive is bounded;
    # ref_omega from the trajectory (max ~1.5 rad/s on lemniscate) is kept intact.
    heading_err = (target_theta - state[2] + np.pi) % (2 * np.pi) - np.pi
    heading_err_sat = 0.5 * np.tanh(2.0 * heading_err)  # ∈ (-0.5, 0.5) rad
    k_theta = 3.0
    omega_target = ref_omega + k_theta * heading_err_sat  # correction bounded ±1.5 rad/s

    # --- Damped blending: gentle near reference, firm when far ---
    # Near the reference, aggressive blending turns small NN prediction noise
    # into steering jitter → limit cycles.  Gain scales up as error grows.
    blend = 0.2 + 0.3 * np.tanh(3.0 * dist_err)   # ∈ [0.2, 0.5]
    omega_des = float(state[4] + blend * (omega_target - state[4]))
    omega_des = float(np.clip(omega_des, -8.0, 8.0))

    # --- Velocity correction (attenuated near reference) ---
    # The 1.5·dist_err feed-forward added noticeable velocity overshoot near
    # the reference.  Attenuate it with a soft gate so it only kicks in when
    # the robot is genuinely off-track (≥ few cm).
    v_corr_gain = 1.0 * np.tanh(5.0 * dist_err)   # ≈0 on ref, →1 at ≥30 cm
    v_target = np.clip(ref_v + v_corr_gain * dist_err, 0.05, 4.0)
    v_des = float(np.clip(state[3] + blend * (v_target - state[3]), 0.05, 4.0))

    # --- Desired next state: one-step forward from current ---
    # Vanilla feedforward target — keeps the (state, desired_next)
    # transition in-distribution with the offline data.  Earlier drafts
    # added a cross-track pull toward the next reference waypoint here
    # but that pushed the desired_next OOD on heavy or windy runs and
    # regressed Frozen tracking by ~5×, so it was removed.  The same
    # role is now played by the omega/v correction that goes through
    # ``target_theta`` and ``v_target`` above (heading + speed), which
    # remain bounded by tanh saturations.
    x_next = state[0] + v_des * np.cos(state[2]) * dt
    y_next = state[1] + v_des * np.sin(state[2]) * dt
    theta_next = state[2] + omega_des * dt

    return np.array([x_next, y_next, theta_next, v_des, omega_des])


def _clamp_input_to_training_range(nn_input, input_mean_np, input_std_np, k=3.0):
    """Clamp raw NN input features to [mean - k*std, mean + k*std].

    This is derived entirely from training data statistics stored in the model,
    so it generalizes across different robots/platforms without manual tuning.

    Parameters
    ----------
    nn_input : array, shape (D,)
        Raw (un-normalized) input features.
    input_mean_np : array, shape (D,)
        Per-feature mean from training data.
    input_std_np : array, shape (D,)
        Per-feature std from training data.
    k : float
        Number of standard deviations to allow. Default 3.0 covers 99.7%
        of Gaussian-distributed training data.

    Returns
    -------
    clamped : array, shape (D,)
    """
    lo = input_mean_np - k * input_std_np
    hi = input_mean_np + k * input_std_np
    return np.clip(nn_input, lo, hi)


class NNController:
    """Neural network inverse dynamics controller.

    Parameters
    ----------
    feedback_gain : float, optional
        If > 0 and ``feedback_env`` is provided, an LQR-style state
        feedback term ``-K(x - x_ref)`` is added to the NN action.  This
        gives a hybrid feedforward (NN) + feedback (linear) controller
        that closes the loop on tracking error.  Default 0.0 keeps the
        legacy purely-feedforward behaviour.
    feedback_env : env, optional
        Environment used to compute the (nominal) linearization for the
        feedback gain.  Only ``effective_mass``, ``effective_Iz``, and
        ``effective_friction`` are read at construction.
    feedback_dt : float, optional
        Timestep used by the feedback DARE.  Required when
        ``feedback_gain > 0``.
    """

    def __init__(self, model, device='cpu', is_evidential=True, feature_mode="full",
                 feedback_gain: float = 0.0, feedback_env=None, feedback_dt: float = 0.02):
        self.model = model
        self.device = device
        self.is_evidential = is_evidential
        # Auto-detect from checkpoint attribute so re-trained models self-configure
        self.feature_mode = getattr(model, 'feature_mode', feature_mode)
        self.model.to(device)
        self.model.eval()

        self.input_mean = model.input_mean.to(device)
        self.input_std = model.input_std.to(device)
        self.target_mean = model.target_mean.cpu().numpy()
        self.target_std = model.target_std.cpu().numpy()
        # Numpy copies for feature clamping
        self.input_mean_np = model.input_mean.cpu().numpy()
        self.input_std_np = model.input_std.cpu().numpy()

        self.feedback_gain = float(feedback_gain)
        if self.feedback_gain > 0.0 and feedback_env is not None:
            from controllers.state_feedback import make_state_feedback
            self._feedback = make_state_feedback(feedback_env, feedback_dt,
                                                  gain=self.feedback_gain)
        else:
            self._feedback = None

    def compute(self, env, state, positions, step, dt):
        """Compute NN control action.

        Parameters
        ----------
        env : UnicycleEnv | AUV3DEnv | Drone3DEnv
            Environment (for build_inverse_dynamics_input and the
            optional ``compute_desired_next_state`` method).
        state : array
            Current state (5-dim for unicycle, 8-dim for 3-D platforms).
        positions : array, shape (N, D)
            Reference trajectory (2-D for unicycle, 3-D for AUV / Drone).
        step : int
            Current time step.
        dt : float
            Timestep.

        Returns
        -------
        action : array
            Predicted control torques (2-dim for unicycle, 4-dim for
            3-D platforms).
        info : dict
        """
        # Dispatch to the env's own waypoint follower if it has one
        # (AUV / Drone), else fall back to the unicycle helper.
        if hasattr(env, "compute_desired_next_state"):
            desired_next = env.compute_desired_next_state(state, positions, step, dt)
        else:
            desired_next = compute_desired_next_state(state, positions, step, dt)

        nn_input = env.build_inverse_dynamics_input(state, desired_next,
                                                     feature_mode=self.feature_mode)

        # Clamp raw features to training data range (mean ± 3*std)
        nn_input = _clamp_input_to_training_range(
            nn_input, self.input_mean_np, self.input_std_np, k=3.0
        )

        # Normalize
        inp_t = torch.FloatTensor(nn_input).unsqueeze(0).to(self.device)
        inp_norm = (inp_t - self.input_mean) / self.input_std

        with torch.no_grad():
            output = self.model(inp_norm)

        if self.is_evidential:
            gamma, nu, alpha, beta = output
            action_norm = gamma.squeeze(0).cpu().numpy()
            ep = epistemic_score(nu, alpha, beta).item()
            al = aleatoric_score(nu, alpha, beta).item()
        else:
            if isinstance(output, tuple):
                action_norm = output[0].squeeze(0).cpu().numpy()
            else:
                action_norm = output.squeeze(0).cpu().numpy()
            ep = 0.0
            al = 0.0

        action = action_norm * self.target_std + self.target_mean

        if self._feedback is not None:
            action = action + self._feedback.compute(state, positions, step)

        # Per-channel clipping using the env's ACTION_LIMITS if available
        # (works for both 2-D unicycle and 4-D AUV / Drone); else fall
        # back to the unicycle defaults.
        lim = getattr(env, "ACTION_LIMITS", None)
        if lim is not None:
            for i, (lo, hi) in enumerate(lim):
                if i < len(action):
                    action[i] = float(np.clip(action[i], lo, hi))
        else:
            action[0] = np.clip(action[0], -20.0, 20.0)
            action[1] = np.clip(action[1], -10.0, 10.0)

        return action, {
            "epistemic": ep,
            "aleatoric": al,
            "desired_next": desired_next,
            "nn_input": nn_input,
        }
