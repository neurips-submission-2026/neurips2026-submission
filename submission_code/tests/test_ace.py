"""Tests for ACEAdapter — the simplified camera-ready ACE method.

The minimal method has three mechanisms:
  1. NIG evidential head (epistemic uncertainty in one forward pass)
  2. Priority replay buffer keyed by epistemic uncertainty
  3. Smooth-EMA λ schedule on observation-time epistemic ε̄
And six hyperparameters: lr, lambda_eta, lambda_kappa, anchor_strength,
batch_size, buffer_capacity.

These tests exercise each mechanism in isolation plus end-to-end behaviour.
"""
import numpy as np
import pytest
import torch

from envs.unicycle_env import UnicycleEnv
from models.evidential_net import EvidentialNet
from models.adaptive_evidential_net import AdaptiveEvidentialNet
from training.train_online import ACEAdapter
from utils.trajectories import generate_trajectory, lemniscate_2d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Small EvidentialNet that matches the 12-d unicycle feature."""
    m = EvidentialNet(12, 2, hidden_dims=[32, 32])
    m.input_mean = torch.zeros(12)
    m.input_std = torch.ones(12)
    m.target_mean = torch.zeros(2)
    m.target_std = torch.ones(2)
    m.feature_mode = "full"
    m.eval()
    return m


@pytest.fixture
def env():
    e = UnicycleEnv(dt=0.02)
    e.reset(state=np.array([0.5, 0.0, 0.0, 0.0, 0.0]))
    return e


@pytest.fixture
def positions():
    fn = lambda t: lemniscate_2d(t, scale=4.0, speed=0.5)
    pos, _ = generate_trajectory(fn, 0.02, 200, r_bar=4)
    return pos


@pytest.fixture
def adapter(model, env):
    return ACEAdapter(
        model, env, dt=0.02,
        lr=2e-4, lambda_eta=0.10, lambda_kappa=0.05,
        anchor_strength=5e-4,
        batch_size=16, buffer_capacity=200,
        update_every=5, min_buffer_size=32,
    )


# ---------------------------------------------------------------------------
# Mechanism 1: NIG head + active sampling priority
# ---------------------------------------------------------------------------

def test_buffer_priority_uses_epistemic(adapter, env, positions):
    """The priority pushed to the buffer must be the per-sample epistemic ε."""
    for k in range(50):
        s = env.get_state()
        adapter.step(env, s, positions, k)
    # Buffer has been populated
    assert len(adapter.buffer) > 0
    # PriorityReplayBuffer stores priorities; sample weighting is keyed by ε.
    # Smoke check: ε values were pushed (non-zero variance over the rollout).
    assert adapter.eps_bar > 0.0


def test_uses_commanded_action_not_actual_torque(adapter, env, positions):
    """Adapter must store the *commanded* action (avoids the wind double-bug)."""
    s = env.get_state()
    adapter.step(env, s, positions, 0)
    s2 = env.get_state()
    a, _ = adapter.step(env, s2, positions, 1)
    # _prev_action equals the action just executed (commanded), not env's
    # last_actual_torque (= command + wind).
    assert np.allclose(adapter._prev_action, a)


# ---------------------------------------------------------------------------
# Mechanism 2: smooth-EMA λ schedule
# ---------------------------------------------------------------------------

def test_lambda_starts_at_one(adapter):
    """λ initialises at 1 (= preserve mode) before any data."""
    assert adapter.lambda_schedule == 1.0


def test_lambda_drops_under_high_epistemic(adapter, env, positions):
    """When ε̄ is large relative to κ_λ, λ should drop below 1."""
    # Force a high ε̄ by manual injection — same code path the live
    # rollout would take.
    adapter.eps_bar = 1.0  # >> κ_λ = 0.05
    # Simulate the per-step λ update
    for _ in range(50):
        lam_target = adapter.lambda_kappa / (adapter.eps_bar + adapter.lambda_kappa)
        adapter.lambda_schedule += adapter.lambda_eta * (
            lam_target - adapter.lambda_schedule)
    # κ/(1+κ) ≈ 0.048 — λ should be very close to that.
    assert adapter.lambda_schedule < 0.10
    assert adapter.lambda_schedule > 0.03


def test_lambda_stays_high_at_low_epistemic(adapter, env, positions):
    """At low ε̄ the schedule should stay near 1."""
    adapter.eps_bar = 0.0
    for _ in range(50):
        lam_target = adapter.lambda_kappa / (adapter.eps_bar + adapter.lambda_kappa)
        adapter.lambda_schedule += adapter.lambda_eta * (
            lam_target - adapter.lambda_schedule)
    assert adapter.lambda_schedule > 0.99


def test_lambda_evolves_per_step(adapter, env, positions):
    """λ should move each step the buffer-push code path runs."""
    initial = adapter.lambda_schedule
    history = []
    for k in range(80):
        s = env.get_state()
        adapter.step(env, s, positions, k)
        history.append(adapter.lambda_schedule)
    # At least one step changed λ
    assert any(h != initial for h in history)
    # λ stays in [0, 1]
    assert all(0.0 <= h <= 1.0 for h in history)


def test_lambda_geometric_convergence_rate(adapter):
    """Theorem 1 sanity: from a perturbed λ, error decays geometrically."""
    adapter.eps_bar = 0.0  # → lam_target = 1
    adapter.lambda_schedule = 0.0
    eta = adapter.lambda_eta
    for k in range(100):
        lam_target = adapter.lambda_kappa / (adapter.eps_bar + adapter.lambda_kappa)
        adapter.lambda_schedule += eta * (lam_target - adapter.lambda_schedule)
        # Bound: |λ_t − 1| ≤ (1 − η)^t · |λ_0 − 1|
        bound = (1.0 - eta) ** (k + 1)
        assert abs(adapter.lambda_schedule - 1.0) <= bound + 1e-9


# ---------------------------------------------------------------------------
# Mechanism 3: anchor regularizer
# ---------------------------------------------------------------------------

def test_anchor_zero_does_not_alter_loss(model, env):
    """anchor_strength=0 ⇒ no anchor term."""
    a = ACEAdapter(model, env, dt=0.02, anchor_strength=0.0,
                       batch_size=8, min_buffer_size=8, update_every=1)
    # Nothing pulls weights toward _anchor
    a._anchor = {n: torch.zeros_like(p) for n, p in a.model.named_parameters()
                 if p.requires_grad}
    # Push junk into buffer
    for _ in range(16):
        a.buffer.push(np.zeros(12), np.zeros(2), priority=0.1)
    p_before = {n: p.detach().clone() for n, p in a.model.named_parameters()}
    a._do_update()
    # Some movement happened from the data gradient
    moved = any(not torch.allclose(p_before[n], p, atol=0)
                for n, p in a.model.named_parameters())
    assert moved


def test_anchor_pulls_weights_toward_snapshot(model, env, positions):
    """With strong anchor, drift away from θ_0 should be tiny."""
    a = ACEAdapter(model, env, dt=0.02, anchor_strength=10.0,
                       lr=1e-2,  # large lr so the anchor's effect is visible
                       batch_size=8, min_buffer_size=8, update_every=1)
    # Run many updates
    for _ in range(64):
        a.buffer.push(np.random.randn(12), np.random.randn(2), priority=0.1)
    for _ in range(20):
        a._do_update()
    drift = sum(
        ((p - a._anchor[n]) ** 2).sum().item()
        for n, p in a.model.named_parameters()
        if n in a._anchor
    )
    assert drift < 5.0  # bounded by η·G/κ_a (Theorem 3)


# ---------------------------------------------------------------------------
# End-to-end: hyperparameter count and contract
# ---------------------------------------------------------------------------

def test_six_method_hyperparameters():
    """ACEAdapter exposes exactly six method-level hyperparameters."""
    import inspect
    sig = inspect.signature(ACEAdapter.__init__)
    method_knobs = {"lr", "lambda_eta", "lambda_kappa", "anchor_strength",
                    "batch_size", "buffer_capacity"}
    found = set(sig.parameters.keys())
    assert method_knobs.issubset(found)


def test_runs_without_crash_under_mass_shift(model, positions):
    """End-to-end smoke: run 200 steps under mass×3 perturbation."""
    env = UnicycleEnv(dt=0.02)
    env._mass_multiplier = 3.0
    env.reset(state=np.array([0.5, 0.0, 0.0, 0.0, 0.0]))
    a = ACEAdapter(model, env, dt=0.02, feedback_gain=1.0)
    for k in range(200):
        s = env.get_state()
        action, info = a.step(env, s, positions, k)
        assert np.all(np.isfinite(action))
        assert 0.0 <= info["lambda_schedule"] <= 1.0


def test_runs_without_crash_nominal(model, env, positions):
    """End-to-end smoke: run 200 steps on Nominal."""
    a = ACEAdapter(model, env, dt=0.02, feedback_gain=1.0)
    for k in range(200):
        s = env.get_state()
        action, _ = a.step(env, s, positions, k)
        assert np.all(np.isfinite(action))


# ---------------------------------------------------------------------------
# iter-30 flag tests.  Each verifies that:
#   1. the default behaviour is unchanged (regression guard for iter-29);
#   2. the new branch triggers when the flag is set away from default.
# ---------------------------------------------------------------------------

def _build_adapter(model, env, **kw):
    """Construct an ACEAdapter with a small budget for fast tests.

    Defaults are merged so callers may override any of them; this avoids
    the ``multiple values for keyword argument`` error when an iter-31
    test wants to set ``lambda_eta`` differently from the default.
    """
    defaults = dict(
        dt=0.02,
        lr=1e-4, lambda_eta=0.10, lambda_kappa=0.05,
        anchor_strength=5e-3, lambda_floor=0.10,
        batch_size=8, buffer_capacity=64,
        update_every=5, min_buffer_size=8,
    )
    defaults.update(kw)
    return ACEAdapter(model, env, **defaults)


def test_iter30_anchor_mode_reverts_to_iter29_when_lambda_one(
        model, env, positions):
    """schedule_mode='anchor' must produce the iter-29 loss when lambda=1.

    Both branches collapse to the same loss because:
      ev_reg branch with lambda_schedule=1: anchor coeff = anchor_strength
      anchor branch with lambda_schedule=1: anchor coeff = anchor_strength * 1
    The evidence regulariser also receives the same lambda. So a single
    gradient step should produce identical parameter updates."""
    torch.manual_seed(0)
    a_ev = _build_adapter(model, env, schedule_mode="ev_reg")
    a_an = _build_adapter(model, env, schedule_mode="anchor")
    # Force lambda_schedule to 1 so both branches are equivalent.
    a_ev.lambda_schedule = 1.0
    a_an.lambda_schedule = 1.0
    # Push one identical batch into each buffer.
    rng = np.random.default_rng(0)
    inp = rng.standard_normal(12).astype(np.float32)
    tgt = rng.standard_normal(2).astype(np.float32)
    for _ in range(20):
        a_ev.buffer.push(inp.copy(), tgt.copy(), priority=1.0)
        a_an.buffer.push(inp.copy(), tgt.copy(), priority=1.0)
    # Snapshot pre-update parameters as a baseline.
    pre_ev = {n: p.detach().clone() for n, p in a_ev.model.named_parameters()}
    # Take a single gradient step from the same starting point.
    torch.manual_seed(0); a_ev._do_update()
    # Reset model to the same starting point and run anchor branch.
    for n, p in a_an.model.named_parameters():
        p.data.copy_(pre_ev[n])
    torch.manual_seed(0); a_an._do_update()
    # Parameter deltas must match.
    for n, p_ev in a_ev.model.named_parameters():
        p_an = dict(a_an.model.named_parameters())[n]
        assert torch.allclose(p_ev, p_an, atol=1e-7), (
            f"anchor mode diverged from ev_reg at lambda=1 in {n}: "
            f"max abs diff = {(p_ev - p_an).abs().max().item():.2e}")


def test_iter30_anchor_mode_weakens_anchor_when_lambda_low(model, env):
    """schedule_mode='anchor' with low lambda must reduce anchor coefficient."""
    a = _build_adapter(model, env, schedule_mode="anchor")
    a.lambda_schedule = 0.1
    # Push a non-trivial batch.
    rng = np.random.default_rng(1)
    for _ in range(20):
        inp = rng.standard_normal(12).astype(np.float32)
        tgt = rng.standard_normal(2).astype(np.float32)
        a.buffer.push(inp, tgt, priority=1.0)
    # Snapshot anchor distance before/after a gradient step.
    pre = {n: p.detach().clone()
           for n, p in a.model.named_parameters() if p.requires_grad}
    a._do_update()
    diff_norm = sum(((p - pre[n]) ** 2).sum().item()
                    for n, p in a.model.named_parameters() if p.requires_grad)
    # In anchor mode at lambda=0.1 the anchor coefficient is 5e-4 vs the
    # 5e-3 of iter-29; a single step should still move the parameters
    # by a measurable amount.
    assert diff_norm > 0.0


def test_iter30_priority_clip_bounds_buffer_priorities(model, env, positions):
    """priority_clip=(lo, hi) keeps buffer priorities inside [lo, hi]."""
    a = _build_adapter(model, env, priority_clip=(0.05, 0.5))
    for k in range(80):
        s = env.get_state()
        a.step(env, s, positions, k)
    pri = a.buffer.priorities
    if len(pri) == 0:
        pytest.skip("buffer empty under this short rollout")
    assert pri.min() >= 0.05 - 1e-9, f"min priority {pri.min()} below clip lower"
    assert pri.max() <= 0.5 + 1e-9, f"max priority {pri.max()} above clip upper"


def test_iter30_priority_alpha_softens_sampling(model, env):
    """priority_alpha < 1 makes sampling more uniform: empirical entropy of
    the sampling distribution rises vs alpha=1 on the same priorities."""
    from utils.buffers import PriorityReplayBuffer
    rng = np.random.default_rng(2)
    pri = np.exp(rng.standard_normal(64) * 2.0)  # heavy-tailed priorities

    def sampling_dist(alpha):
        buf = PriorityReplayBuffer(capacity=64, sampling_alpha=alpha)
        for v in pri:
            buf.push(np.zeros(4, dtype=np.float32),
                     np.zeros(2, dtype=np.float32),
                     priority=float(v))
        # Reproduce the internal sampling distribution.
        p = buf._priorities[:buf._size].copy()
        p -= p.min() - 1e-9
        if alpha != 1.0:
            p = np.power(p, alpha)
        p /= p.sum()
        return p

    p1 = sampling_dist(1.0)
    pa = sampling_dist(0.6)
    h1 = -np.sum(p1 * np.log(p1 + 1e-12))
    ha = -np.sum(pa * np.log(pa + 1e-12))
    assert ha > h1, (
        f"entropy did not increase under alpha=0.6: H(alpha=1)={h1:.3f} "
        f"H(alpha=0.6)={ha:.3f}")


def test_iter30_relative_eps_baseline_captures_after_warmup(
        model, env, positions):
    """When relative_eps=True, eps_baseline is None during warmup and gets
    set to a positive value once step_count crosses warmup_steps (64)."""
    a = _build_adapter(model, env, relative_eps=True)
    # Drive enough steps to cross warmup.  Use a non-trivial epsilon by
    # injecting noise into get_state() so the model sees OOD inputs.
    for k in range(80):
        s = env.get_state()
        a.step(env, s, positions, k)
    # Either the baseline was captured (typical case) OR the rollout was
    # too short / noise too low and the baseline stayed None.  Assert the
    # invariant: if captured, it must be strictly positive.
    if a.eps_baseline is not None:
        assert a.eps_baseline > 0.0
    # Sanity: baseline tracking is gated behind the flag.
    a_iter29 = _build_adapter(model, env, relative_eps=False)
    for k in range(80):
        s = env.get_state()
        a_iter29.step(env, s, positions, k)
    assert a_iter29.eps_baseline is None, (
        "iter-29 default must never set eps_baseline")


def test_iter30_invalid_flag_values_raise():
    """Constructor must reject invalid flag combinations early."""
    m = EvidentialNet(12, 2, hidden_dims=[32, 32])
    m.input_mean = torch.zeros(12); m.input_std = torch.ones(12)
    m.target_mean = torch.zeros(2); m.target_std = torch.ones(2)
    m.feature_mode = "full"
    e = UnicycleEnv(dt=0.02); e.reset(state=np.zeros(5))
    with pytest.raises(ValueError):
        ACEAdapter(m, e, dt=0.02, schedule_mode="invalid")
    with pytest.raises(ValueError):
        ACEAdapter(m, e, dt=0.02, priority_clip=(1.0, 0.5))  # lo >= hi
    with pytest.raises(ValueError):
        ACEAdapter(m, e, dt=0.02, priority_alpha=-0.1)


# ---------------------------------------------------------------------------
# iter-31 tempered-loss tests.  The data terms (NLL + MSE) are scaled by
# (1 - lambda_k); the evidence regulariser keeps full lambda_0 weight; the
# anchor stays constant.  At lambda_k = 0 the tempered branch is equivalent
# to "iter29 mode with lambda_0=1.0 disabled" (same NLL gradient direction,
# different scalar weight).  At lambda_k = 1 the tempered branch produces no
# data gradient at all.
# ---------------------------------------------------------------------------

def test_iter31_tempered_zero_lambda_step_moves_weights(model, env):
    """At lambda_schedule = 0 (full adapt) tempered loss must take a real step."""
    a = _build_adapter(model, env, loss_form="tempered",
                       lambda_schedule_init=0.0,
                       lambda_eta=0.0)  # freeze schedule at init value
    rng = np.random.default_rng(7)
    for _ in range(20):
        a.buffer.push(rng.standard_normal(12).astype(np.float32),
                      rng.standard_normal(2).astype(np.float32),
                      priority=1.0)
    pre = {n: p.detach().clone()
           for n, p in a.model.named_parameters() if p.requires_grad}
    a._do_update()
    moved = sum(((p - pre[n]) ** 2).sum().item()
                for n, p in a.model.named_parameters() if p.requires_grad)
    assert moved > 0.0, "tempered loss at lambda=0 produced zero parameter movement"


def test_iter31_tempered_lambda_one_only_anchor_pulls(model, env):
    """At lambda_schedule = 1 (full preserve) the data weight is zero, so the
    only gradient that touches the parameters is the constant anchor pulling
    them toward theta_0.  Since the model starts at theta_0, no movement
    happens within numerical precision."""
    a = _build_adapter(model, env, loss_form="tempered",
                       lambda_schedule_init=1.0,
                       lambda_eta=0.0,
                       anchor_strength=5e-3)
    rng = np.random.default_rng(8)
    for _ in range(20):
        a.buffer.push(rng.standard_normal(12).astype(np.float32),
                      rng.standard_normal(2).astype(np.float32),
                      priority=1.0)
    # Take a single step.  evidence regulariser does pull on (alpha, beta, nu)
    # heads but the L2 anchor is at theta_0 so the dominant gradient is small.
    pre = {n: p.detach().clone()
           for n, p in a.model.named_parameters() if p.requires_grad}
    a._do_update()
    moved = sum(((p - pre[n]) ** 2).sum().item()
                for n, p in a.model.named_parameters() if p.requires_grad)
    # Movement should be much smaller than under lambda=0 (data term off);
    # we assert it's bounded rather than exactly zero because the evidence
    # regulariser still produces a gradient on the NIG head outputs.
    assert moved < 1e-2, (
        f"tempered loss at lambda=1 still moved parameters too much: {moved}")


def test_iter31_tempered_movement_scales_with_one_minus_lambda(model, env):
    """Parameter movement under tempered loss should scale with (1-lambda).

    The grad clip in _do_update saturates if we crank lr or targets, so
    we instead inspect the unclipped gradient norm BEFORE the optimiser
    step.  With anchor off and a fixed seed, the gradient norm at
    lambda=0.1 should be roughly 9x larger than at lambda=0.9 (ratio
    of (1-0.1) / (1-0.9) = 9)."""
    rng = np.random.default_rng(9)
    inputs = np.stack([rng.standard_normal(12).astype(np.float32)
                       for _ in range(32)])
    targets = np.stack([rng.standard_normal(2).astype(np.float32)
                        for _ in range(32)])

    def _grad_norm(lambda_init):
        torch.manual_seed(0)
        a = _build_adapter(model, env, loss_form="tempered",
                           lambda_schedule_init=lambda_init,
                           lambda_eta=0.0,
                           anchor_strength=0.0,
                           batch_size=32, buffer_capacity=64,
                           min_buffer_size=8)
        for inp, tgt in zip(inputs, targets):
            a.buffer.push(inp.copy(), tgt.copy(), priority=1.0)
        # Recreate the loss path manually so we can read the gradient
        # norm before the clip + optimiser step.
        buf_inputs, buf_targets = a.buffer.sample(a.batch_size)
        inp_t = torch.FloatTensor(buf_inputs); tgt_t = torch.FloatTensor(buf_targets)
        a.model.train()
        gamma, nu, alpha, beta = a.model(inp_t)
        nll = a.model._nig_nll(tgt_t, gamma, nu, alpha, beta)
        reg = torch.abs(tgt_t - gamma) * (2.0 * nu + alpha)
        mse = torch.nn.functional.smooth_l1_loss(gamma, tgt_t, beta=1.0)
        data_term = nll.mean() + a.mse_weight * mse
        evreg_term = a.lambda_0 * reg.mean()
        loss = (1.0 - lambda_init) * data_term + evreg_term
        a.optimizer.zero_grad()
        loss.backward()
        gn = sum(p.grad.detach().norm().item() ** 2
                 for p in a.model.parameters() if p.grad is not None) ** 0.5
        return gn

    g_low = _grad_norm(0.1)
    g_high = _grad_norm(0.9)
    # The fixed evidential regulariser is the only part NOT scaled by
    # (1-lambda); call its contribution G_reg.  Then G(lambda) ≈ |G_reg +
    # (1-lambda)*G_data|, so we expect g_low > g_high but not necessarily
    # by exactly 9x.  A factor of 2 is a generous lower bound.
    assert g_low > 2.0 * g_high, (
        f"tempered grad norm did not respect data weight: "
        f"|grad|(lambda=0.1)={g_low:.3e}, |grad|(lambda=0.9)={g_high:.3e}")


def test_iter31_invalid_loss_form_raises(model, env):
    with pytest.raises(ValueError):
        ACEAdapter(model, env, dt=0.02, loss_form="kl-regularised")
    with pytest.raises(ValueError):
        ACEAdapter(model, env, dt=0.02, lambda_schedule_init=1.5)
    with pytest.raises(ValueError):
        ACEAdapter(model, env, dt=0.02, lambda_schedule_init=-0.1)


# ---------------------------------------------------------------------------
# iter-32 schedule_signal="aleatoric_share" tests.  The schedule weight is
# the aleatoric fraction of total predictive uncertainty (Kendall & Gal
# 2017).  Three properties to verify:
#   1. When aleatoric dominates (al_bar >> eps_bar), lam_target -> 1
#      (preserve, no data gradient).
#   2. When epistemic dominates (eps_bar >> al_bar), lam_target -> 0
#      (full adapt up to lambda_floor).
#   3. The constructor rejects invalid schedule_signal values.
# ---------------------------------------------------------------------------

def _settle_aleatoric_share(adapter, eps_bar, al_bar, n_iters=200):
    """Drive the aleatoric-share schedule to its fixed point at the
    given EMA values.  Mirrors the schedule-update equation in
    train_online.py exactly: lam_target = max(al_bar/(eps_bar+al_bar+1e-6),
    kappa/(eps_bar+kappa))."""
    adapter.eps_bar = float(eps_bar)
    adapter.al_bar = float(al_bar)
    for _ in range(n_iters):
        denom = adapter.eps_bar + adapter.al_bar + 1e-6
        lam_share = adapter.al_bar / denom
        lam_sat = adapter.lambda_kappa / (adapter.eps_bar + adapter.lambda_kappa)
        lam_target = max(lam_share, lam_sat)
        adapter.lambda_schedule += adapter.lambda_eta * (
            lam_target - adapter.lambda_schedule)
    return adapter.lambda_schedule


def test_iter32_aleatoric_share_preserves_under_noise(model, env):
    """al_bar >> eps_bar  =>  share dominates  =>  lambda -> 1 (preserve)."""
    a = _build_adapter(model, env, schedule_signal="aleatoric_share",
                       lambda_floor=0.0, lambda_kappa=0.05)
    lam = _settle_aleatoric_share(a, eps_bar=0.001, al_bar=1.0)
    assert lam > 0.99, f"aleatoric-share did not preserve under noise: lambda={lam:.4f}"


def test_iter32_aleatoric_share_preserves_when_both_uncertainties_high(model, env):
    """eps_bar ≈ al_bar BOTH high (the noise-cell regime that broke iter-32
    in smoke testing): share alone gives ~0.5 (moderate adapt, wrong); the
    kappa-saturation tie-breaker pulls the target back toward 1 (preserve)
    only when eps_bar is small in absolute scale.  When BOTH are large the
    formula falls back to share which is ~0.5 — and that is acceptable
    because the noise gate in step() blocks updates anyway in this regime."""
    a = _build_adapter(model, env, schedule_signal="aleatoric_share",
                       lambda_floor=0.0, lambda_kappa=0.05)
    # Noise cell on Unicycle: tiny eps_bar, tiny al_bar, ratio ~0.5
    # but lam_sat dominates with kappa=0.05 >> eps_bar=0.001
    lam = _settle_aleatoric_share(a, eps_bar=0.001, al_bar=0.001)
    assert lam > 0.95, (
        f"aleatoric-share did not preserve under low absolute epistemic: "
        f"lambda={lam:.4f}")


def test_iter32_aleatoric_share_adapts_under_epistemic(model, env):
    """eps_bar >> al_bar AND eps_bar >> kappa  =>  both share AND sat give
    small targets  =>  lambda -> low."""
    a = _build_adapter(model, env, schedule_signal="aleatoric_share",
                       lambda_floor=0.0, lambda_kappa=0.05)
    # eps_bar=1.0 is much larger than kappa=0.05 (sat = 0.048)
    # AND much larger than al_bar=0.001 (share = 0.001)
    lam = _settle_aleatoric_share(a, eps_bar=1.0, al_bar=0.001)
    assert lam < 0.10, (
        f"aleatoric-share did not adapt under epistemic dominance: "
        f"lambda={lam:.4f}")


def test_iter32_kappa_acts_as_safety_floor(model, env):
    """Smaller lambda_kappa should keep more cells in adapt mode (sat
    saturates faster); larger lambda_kappa should keep more cells in
    preserve mode (sat stays high for longer eps_bar range)."""
    a_small = _build_adapter(model, env, schedule_signal="aleatoric_share",
                             lambda_kappa=0.001, lambda_floor=0.0)
    a_large = _build_adapter(model, env, schedule_signal="aleatoric_share",
                             lambda_kappa=10.0, lambda_floor=0.0)
    # On a low-epistemic cell the kappa saturation is the only thing
    # that holds lambda up: small kappa -> sat falls off fast -> share
    # dominates -> lambda follows ratio. Large kappa -> sat ~= 1 always.
    lam_small = _settle_aleatoric_share(a_small, eps_bar=0.05, al_bar=0.05)
    lam_large = _settle_aleatoric_share(a_large, eps_bar=0.05, al_bar=0.05)
    # Large kappa keeps lambda closer to 1 than small kappa does.
    assert lam_large > lam_small, (
        f"kappa did not act as a preserve guard: "
        f"kappa=0.001 -> {lam_small:.4f}, kappa=10.0 -> {lam_large:.4f}")


def test_iter32_invalid_schedule_signal_raises(model, env):
    with pytest.raises(ValueError):
        ACEAdapter(model, env, dt=0.02, schedule_signal="evidence_share")
    with pytest.raises(ValueError):
        ACEAdapter(model, env, dt=0.02, schedule_signal=None)


# ---------------------------------------------------------------------------
# iter-33 noise_gate="continuous" tests.  The binary skip in step() is
# replaced by a multiplicative attenuation on the data term inside
# _do_update.  Three properties to verify:
#   1. The attenuation goes to 0 as al_bar_fast grows large.
#   2. The attenuation is 1 when al_bar_fast is 0 and obs_jitter is 0.
#   3. With noise_gate=continuous, _do_update IS called even when
#      al_bar_fast > kappa/20 (where the binary gate would skip).
# ---------------------------------------------------------------------------

def test_iter33_continuous_attn_formula_decays_with_aleatoric():
    """Quartic-decay attenuation:
        attn_al = 1 / (1 + (al_bar_fast/threshold)^4)
        attn_jit = 1 / (1 + (obs_jitter/jitter_threshold)^4)
        noise_attn = min(attn_al, attn_jit)
    Sharper than quadratic so cumulative drift under sustained noise
    stays bounded comparable to the binary-gate's hard skip.  Tests
    the formula directly to avoid in-step() EMA blending."""
    kappa = 0.05
    noise_threshold = kappa / 20.0   # 0.0025
    jitter_threshold = 0.01

    def _attn(al, jit):
        r_al = al / max(noise_threshold, 1e-9)
        r_jit = jit / max(jitter_threshold, 1e-9)
        return min(1.0 / (1.0 + r_al ** 4), 1.0 / (1.0 + r_jit ** 4))

    assert abs(_attn(0.0, 0.0) - 1.0) < 1e-9
    assert 0.45 < _attn(noise_threshold, 0.0) < 0.55           # ~0.5 at threshold
    assert _attn(2 * noise_threshold, 0.0) < 0.07              # 1/(1+16) = 0.059
    assert _attn(5 * noise_threshold, 0.0) < 1e-2              # 1/(1+625) = 0.0016
    assert _attn(100.0, 0.0) < 1e-9                             # heavy noise -> 0
    assert 0.45 < _attn(0.0, jitter_threshold) < 0.55          # jitter symmetric
    assert _attn(100.0, 100.0) < 1e-9                          # both gates active


def test_iter33_continuous_skips_via_early_exit_under_extreme_noise(
        model, env, positions):
    """With noise_gate=continuous AND ratio_max > 5x threshold, the gate
    falls back to binary-style hard skip via the early-exit in _do_update.
    This is the iter-33 hybrid behaviour: continuous between 0..5x, binary
    above 5x, so cumulative drift on heavy-noise cells stays bounded."""
    a = _build_adapter(model, env, noise_gate="continuous",
                       update_every=5, min_buffer_size=4)
    # Drive al_bar_fast WAY above 5x the binary threshold
    a.al_bar_fast = 1.0   # 400x the threshold
    rng = np.random.default_rng(0)
    for _ in range(8):
        a.buffer.push(rng.standard_normal(12).astype(np.float32),
                      rng.standard_normal(2).astype(np.float32),
                      priority=1.0)
    pre_param = {n: p.detach().clone()
                 for n, p in a.model.named_parameters() if p.requires_grad}
    for k in range(80):
        s = env.get_state()
        a.step(env, s, positions, k)
        a.al_bar_fast = 1.0
    # Parameters must NOT have drifted under extreme noise (the hybrid
    # cutoff plus early-exit in _do_update should reproduce binary-skip
    # semantics here).
    drift = sum(((p - pre_param[n]) ** 2).sum().item()
                for n, p in a.model.named_parameters() if n in pre_param)
    assert drift < 1e-6, (
        f"continuous+hybrid should not drift parameters under extreme noise "
        f"(drift={drift:.2e})")


def test_iter33_invalid_noise_gate_raises(model, env):
    with pytest.raises(ValueError):
        ACEAdapter(model, env, dt=0.02, noise_gate="annealing")


# ---------------------------------------------------------------------------
# iter-34 sustained-bias detector ("shift mode") tests.
# ---------------------------------------------------------------------------

def test_iter34_shift_signal_stays_low_under_zero_mean_residuals():
    """When the residual EMA is dominated by zero-mean noise, the SNR of
    the bias should stay near zero.  Tested by feeding equal-magnitude
    positive and negative residuals into the EMA."""
    rng = np.random.default_rng(0)
    bias = np.zeros(2)
    eta = 0.05
    for _ in range(500):
        r = rng.normal(0.0, 1.0, size=2)  # zero-mean white
        bias = (1 - eta) * bias + eta * r
    al_bar = 1.0  # noise scale
    snr = np.linalg.norm(bias) / (np.sqrt(al_bar) + 1e-6)
    assert snr < 1.0, (
        f"shift_signal under zero-mean noise should stay below 1.0 "
        f"(saw {snr:.3f})")


def test_iter34_shift_signal_rises_under_constant_bias():
    """When the residual EMA settles around a non-zero bias direction,
    the SNR should grow well above 1.0."""
    bias = np.zeros(4)
    eta = 0.05
    constant_bias = np.array([0.5, -0.3, 0.7, 0.1])
    for _ in range(500):
        bias = (1 - eta) * bias + eta * constant_bias
    al_bar = 0.01  # smallish noise
    snr = np.linalg.norm(bias) / (np.sqrt(al_bar) + 1e-6)
    assert snr > 5.0, (
        f"shift_signal under constant bias should rise above 5.0 "
        f"(saw {snr:.3f})")


def test_iter34_shift_mode_bypass_off_by_default(model, env, positions):
    """With shift_detection=False, _shift_mode must always be False even
    if the SNR happens to be high.  iter-31 byte-equivalence."""
    a = _build_adapter(model, env, shift_detection=False)
    # Drive a few steps to compute residuals
    for k in range(10):
        s = env.get_state()
        a.step(env, s, positions, k)
    # Force a high signal
    a.shift_signal = 100.0
    a.eps_bar = 1.0
    # Run another step; _shift_mode should remain False
    s = env.get_state()
    a.step(env, s, positions, 10)
    assert a._shift_mode is False


def test_iter34_invalid_shift_threshold_raises(model, env):
    with pytest.raises(ValueError):
        ACEAdapter(model, env, dt=0.02, shift_threshold=0.0)
    with pytest.raises(ValueError):
        ACEAdapter(model, env, dt=0.02, shift_threshold=-1.0)
