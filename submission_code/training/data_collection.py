"""Self-supervised transition data collection for inverse dynamics."""
import numpy as np
from envs.unicycle_env import UnicycleEnv
from controllers.lqr_controller import LQRController
from utils.trajectories import generate_trajectory, lemniscate_2d
from utils.exciting_trajectories import generate_exciting_trajectory


def collect_transitions(env, positions, dt, n_steps,
                        action_source="lqr", ctrl=None,
                        noise_std=0.0, rng=None, feature_mode="invariant"):
    """Collect (state_t, state_{t+1}, action_t) transitions.

    The target is always the ACTUAL applied torque — self-supervised.
    """
    inputs, targets = [], []

    for k in range(n_steps - 1):
        state_t = env.get_state()

        if action_source == "lqr" and ctrl is not None:
            action, _ = ctrl.compute(state_t, positions, k)
        elif action_source == "random" and rng is not None:
            t = k * dt
            action = np.array([
                5.0 * np.sin(0.5 * t) + 3.0 * np.sin(1.7 * t) + rng.randn() * 2.0,
                2.0 * np.cos(0.8 * t) + 1.5 * np.sin(2.3 * t) + rng.randn() * 1.0,
            ])
        elif action_source == "chirp":
            t = k * dt
            f = 0.1 + 2.0 * t / (n_steps * dt)
            action = np.array([
                8.0 * np.sin(2 * np.pi * f * t),
                4.0 * np.cos(2 * np.pi * f * t * 0.7),
            ])
        else:
            action = np.zeros(2)

        if noise_std > 0 and rng is not None:
            action[0] += rng.randn() * noise_std * 3.0
            action[1] += rng.randn() * noise_std * 1.5

        action = np.clip(action, [-20, -10], [20, 10])
        state_tp1 = env.step(action)
        nn_input = env.build_inverse_dynamics_input(state_t, state_tp1,
                                                     feature_mode=feature_mode)
        inputs.append(nn_input)
        targets.append(action.copy())

    return np.array(inputs), np.array(targets)


def collect_diverse_dataset(cfg, seed=42, feature_mode="invariant", fast=False,
                            demo=False, data_mode="legacy",
                            domain_rand=False):
    """Collect diverse transitions from multiple exploration strategies.

    Parameters
    ----------
    fast : bool
        If True, collect ~10x fewer transitions for quick testing (~2 min).
    demo : bool
        If True, collect minimal data for a smoke-test (~30 s).  Implies fast.
    data_mode : {"legacy", "pe", "dr"}
        ``legacy`` -- the original lemniscate-heavy mix (single trajectory family).
        ``pe``     -- replace lemniscate configs with composite-PE trajectories
                      (sum-of-incommensurate-frequencies + chirp + slow wandering).
        ``dr``     -- DR-only baseline: lemniscate configs but per-rollout
                      sampling of mass / friction / wind.  See ``domain_rand``.
    domain_rand : bool
        If True, before each Strategy-1 rollout sample
        ``mass_mult ~ U(0.7, 1.5)``, ``friction_mult ~ U(0.5, 2.0)``,
        ``wind_x ~ N(0, 1.5)`` and apply via env attributes.  Forces the
        inverse-dynamics model to see a wider parameter envelope at
        training time so the online adapter does not have to repair from
        scratch under shift.
    """
    if demo:
        fast = True
    dt = cfg.sim.dt
    rng = np.random.RandomState(seed)
    r_bar = 4
    if demo:
        steps_per = 100
    elif fast:
        steps_per = 500
    else:
        steps_per = 2000

    all_inputs, all_targets = [], []

    # ── Strategy 1: LQR-guided trajectories ──
    traj_configs = []
    # Strategy-1 trajectory family is data-mode dependent.
    #   legacy : original lemniscate-heavy mix
    #   pe     : every config swapped to composite-PE (broadband)
    #   dr     : lemniscate configs, but per-rollout DR turned on (caller
    #            must also pass ``domain_rand=True``).
    if demo:
        if data_mode == "pe":
            traj_configs.append(('composite', 4.0, 0.5, 42))
        else:
            traj_configs.append(('lemniscate', 4.0, 0.5, 0))
    elif fast:
        if data_mode == "pe":
            for s in [42, 123, 456, 789]:
                traj_configs.append(('composite', 4.0, 0.5, s))
            traj_configs.append(('multi_freq', 4.0, 0.5, 0))
            traj_configs.append(('chirp', 3.5, 0.5, 0))
        else:
            for scale in [3.0, 5.0]:
                for speed in [0.3, 0.7]:
                    traj_configs.append(('lemniscate', scale, speed, 0))
            traj_configs.append(('multi_freq', 4.0, 0.5, 0))
            traj_configs.append(('chirp', 3.5, 0.5, 0))
            traj_configs.append(('composite', 4.0, 0.5, 42))
    else:
        if data_mode == "pe":
            # 16 composite-PE configs, 6 multi_freq, 4 chirp.
            for s in [11, 22, 33, 42, 55, 66, 77, 88,
                      99, 111, 123, 156, 222, 333, 444, 555]:
                traj_configs.append(('composite', 4.0, 0.5, s))
            for scale in [2.5, 4.0, 5.5]:
                for speed in [0.3, 0.7]:
                    traj_configs.append(('multi_freq', scale, speed, 0))
            for scale in [2.0, 3.0, 4.0, 5.0]:
                traj_configs.append(('chirp', scale, 0.5, 0))
        else:   # legacy + dr both reuse the original mix
            for scale in [2.0, 3.0, 4.0, 5.0, 6.0]:
                for speed in [0.3, 0.5, 0.7]:
                    traj_configs.append(('lemniscate', scale, speed, 0))
            for scale in [2.5, 4.0, 5.5]:
                for speed in [0.3, 0.5, 0.7]:
                    traj_configs.append(('multi_freq', scale, speed, 0))
            for scale in [2.0, 3.5, 5.0]:
                traj_configs.append(('chirp', scale, 0.5, 0))
            for s in [42, 123, 456, 789]:
                traj_configs.append(('composite', 4.0, 0.5, s))

    noise_levels = [0.0] if demo else ([0.0, 1.0] if fast else [0.0, 0.5, 2.0])
    print(f"  Strategy 1: {len(traj_configs)} trajectories x {len(noise_levels)} noise")

    for i, (ttype, scale, speed, tseed) in enumerate(traj_configs):
        if ttype == 'lemniscate':
            fn = lambda t, s=scale, sp=speed: lemniscate_2d(t, scale=s, speed=sp)
            positions, _ = generate_trajectory(fn, dt, steps_per, r_bar)
        else:
            positions, _ = generate_exciting_trajectory(
                ttype, dt, steps_per, r_bar, scale=scale, speed=speed, seed=tseed)

        x0, y0 = positions[0]
        ddx = positions[min(2, len(positions)-1)][0] - positions[0][0]
        ddy = positions[min(2, len(positions)-1)][1] - positions[0][1]
        theta0 = np.arctan2(ddy, ddx) if (abs(ddx)+abs(ddy)) > 1e-8 else 0.0

        for noise in noise_levels:
            env = UnicycleEnv(dt=dt)
            env.reset(state=np.array([x0, y0, theta0, 0.0, 0.0]))
            # Domain randomisation (PE+DR / DR-only modes).  Sampled
            # once per rollout so the trajectory sees a coherent set of
            # parameters.  K-feedback inside LQRController re-uses
            # env.effective_* so it tracks the perturbed dynamics
            # while the recorded transitions reflect the action that
            # actually achieved the realised state change.
            if domain_rand:
                env._mass_multiplier = float(rng.uniform(0.7, 1.5))
                env._friction_multiplier = float(rng.uniform(0.5, 2.0))
                env.set_wind(float(rng.normal(0.0, 1.5)),
                             float(rng.normal(0.0, 1.5)))
            ctrl = LQRController(env, dt)
            inp, tgt = collect_transitions(
                env, positions, dt, steps_per,
                action_source="lqr", ctrl=ctrl, noise_std=noise, rng=rng,
                feature_mode=feature_mode)
            all_inputs.append(inp)
            all_targets.append(tgt)

        if (i + 1) % 10 == 0 or i == len(traj_configs) - 1:
            n = sum(len(x) for x in all_inputs)
            print(f"    [{i+1}/{len(traj_configs)}] {n:,} transitions")

    # ── Strategy 2: Random sinusoidal ──
    n_random = 1 if demo else (4 if fast else 20)
    print(f"  Strategy 2: {n_random} random sinusoidal rollouts")
    for _ in range(n_random):
        env = UnicycleEnv(dt=dt)
        env.reset(state=np.array([0, 0, rng.uniform(-np.pi, np.pi),
                                   rng.uniform(-1, 1), rng.uniform(-2, 2)]))
        positions = np.zeros((steps_per, 2))
        inp, tgt = collect_transitions(
            env, positions, dt, steps_per, action_source="random", rng=rng,
            feature_mode=feature_mode)
        all_inputs.append(inp)
        all_targets.append(tgt)

    # ── Strategy 3: Chirp sweep ──
    n_chirp = 0 if demo else (2 if fast else 10)
    print(f"  Strategy 3: {n_chirp} chirp rollouts")
    for _ in range(n_chirp):
        env = UnicycleEnv(dt=dt)
        env.reset(state=np.array([0, 0, rng.uniform(-np.pi, np.pi),
                                   rng.uniform(-0.5, 0.5), rng.uniform(-1, 1)]))
        positions = np.zeros((steps_per, 2))
        inp, tgt = collect_transitions(
            env, positions, dt, steps_per, action_source="chirp",
            feature_mode=feature_mode)
        all_inputs.append(inp)
        all_targets.append(tgt)

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    idx = rng.permutation(len(inputs))
    inputs, targets = inputs[idx], targets[idx]

    print(f"\n  Total: {len(inputs):,} transitions  "
          f"(input_dim={inputs.shape[1]}, output_dim={targets.shape[1]})")
    return inputs, targets


# ----------------------------------------------------------------------
# 3-D platform offline data collection (AUV / Drone).
#
# Mirrors `collect_diverse_dataset` but for the 3-D envs.  Uses LQR3D
# to drive the env along a mix of:
#   - exciting_3d sum-of-sinusoids (PE waypoints; the headline driver)
#   - lemniscate_3d (legacy lemniscate trajectory, kept for back-compat)
#
# Domain-randomisation samples mass / drag / wind per rollout to give
# the inverse-dynamics model a wide parameter envelope at training
# time — same idea as the Unicycle PE+DR mode, ported to 3-D.
#
# This is the function the user's "AUV PE+DR retraining" depends on.
# ----------------------------------------------------------------------


def collect_diverse_dataset_3d(make_env, kind: str, dt: float,
                                 feature_mode: str = "invariant",
                                 fast: bool = False, demo: bool = False,
                                 data_mode: str = "pe",
                                 domain_rand: bool = True,
                                 seed: int = 42):
    """Collect 3-D inverse-dynamics transitions with PE+DR.

    First-attempt design (using `exciting_3d` setpoints + LQR3D
    tracking) failed on AUV/Drone because the high-bandwidth setpoint
    trajectory pushes LQR3D's hover linearisation outside its valid
    region — the resulting actions saturate at the action limits and
    the recorded (state, action) pairs are not a faithful sample of
    the inverse-dynamics map.  This redesign uses:

      Strategy 1 (slow, smooth setpoints):
        - lemniscate_3d at multiple scales / slow speeds matched to
          platform bandwidth.
        - smooth_waypoints_3d (exists in utils/trajectories.py).
        Both are within LQR3D's tracking authority.

      Strategy 2 (PE source — open-loop random smooth actions):
        Apply random-frequency sinusoids directly as the action,
        observe the resulting state transitions.  This is the actual
        "persistent-excitation" data — the action signal directly
        excites all input dimensions, no closed-loop needed.

      DR: Per-rollout perturbation in mass / drag / wind for both
      strategies.

    Returns (inputs, targets) ndarrays.
    """
    from controllers.lqr_3d import LQR3D
    from utils.trajectories import lemniscate_3d, smooth_waypoints_3d, exciting_3d

    rng = np.random.RandomState(seed)
    if demo:
        steps_per = 200
    elif fast:
        steps_per = 800
    else:
        steps_per = 3000

    all_inputs, all_targets = [], []

    # Probe one env instance to read action limits + state dim from the
    # platform definition (so quadrotor's torque-thrust action and
    # 12-state interface flow through automatically).
    _probe = make_env()
    action_lo = np.array([lo for lo, _ in _probe.ACTION_LIMITS], dtype=np.float64)
    action_hi = np.array([hi for _, hi in _probe.ACTION_LIMITS], dtype=np.float64)
    state_dim = int(getattr(_probe, "STATE_DIM", 8))

    if kind == "auv":
        z_off = -3.0
        lem_speeds = [0.04, 0.07, 0.10] if not fast else [0.05, 0.08]
        lem_scales = [0.6, 1.0, 1.4] if not fast else [0.8, 1.2]
        # Symmetric DR (mean = 1.0) — avoids systematic NN bias from
        # E[mass·g] > nominal that the previous asymmetric range produced.
        # AUV uses a wider mass range so the model can still extrapolate
        # to MassShift's mass_mult=1.5 scenario (narrower [0.85, 1.15]
        # caused a -39% regression on AUV/Mass at iter-27 first try).
        dr_mass_lo, dr_mass_hi = 0.75, 1.25
        dr_drag_lo, dr_drag_hi = 0.70, 1.30
        dr_wind_sigma = 0.0
    else:  # drone (quadrotor)
        z_off = 3.0
        lem_speeds = [0.06, 0.10, 0.14] if not fast else [0.08, 0.12]
        lem_scales = [0.3, 0.5, 0.7] if not fast else [0.4, 0.6]
        dr_mass_lo, dr_mass_hi = 0.85, 1.15
        dr_drag_lo, dr_drag_hi = 0.80, 1.20
        dr_wind_sigma = 0.10

    def _init_state(p0, psi0):
        """Build an initial state vector matching the env's state shape."""
        if state_dim == 12:
            # Quadrotor: hover at (p0_x, p0_y, p0_z), level attitude, zero vels.
            return np.array([p0[0], p0[1], p0[2], 0.0, 0.0, psi0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # 8-state body-frame model (AUV).
        return np.array([p0[0], p0[1], p0[2], psi0, 0.0, 0.0, 0.0, 0.0])

    # --- Strategy 1: LQR-tracked smooth trajectories ---
    # Each lemniscate / smooth-waypoint is generated in BOTH CW and CCW
    # directions (and with both signs of psi-init) so the (tau_x, tau_y)
    # label distribution is zero-mean.  Without this, an offline NN
    # trained on always-CCW lemniscates learns a biased "tilt forward-
    # right" mean — visible in the iter-27 diagnostic as
    # tau_x +0.037 / tau_y -0.027 NN bias on a nominal lemniscate.
    if data_mode in ("pe", "lemniscate"):
        traj_configs = []
        for scale in lem_scales:
            for speed in lem_speeds:
                for direction in (+1.0, -1.0):
                    traj_configs.append(("lemniscate", scale, speed, direction))
        # Smooth-waypoints: gentler swept loops.
        if not demo:
            for scale in lem_scales[::2]:
                for direction in (+1.0, -1.0):
                    traj_configs.append(("smooth", scale, 0.0, direction))

        noise_levels = [0.0] if demo else ([0.0, 0.3] if fast else [0.0, 0.3, 0.8])
        print(f"  3-D Strategy 1 ({kind}): {len(traj_configs)} smooth-traj × "
              f"{len(noise_levels)} noise")

        for i, (ttype, scale, speed, direction) in enumerate(traj_configs):
            positions = np.zeros((steps_per, 3), dtype=np.float64)
            for k in range(steps_per):
                if ttype == "lemniscate":
                    # Stronger Z excitation combined with x/y motion: 5x
                    # the previous scale_z so the inverse-dynamics
                    # network sees a wider range of vertical thrust
                    # demands during training.
                    p = lemniscate_3d(k, scale_xy=scale,
                                      scale_z=1.5 if kind == "auv" else 1.0,
                                      z_offset=z_off, speed=speed)
                else:
                    p = smooth_waypoints_3d(
                        k, scale_xy=scale,
                        scale_z=1.5 if kind == "auv" else 1.0,
                        period=400.0,
                        z_offset=z_off)
                # Mirror y to flip the trajectory direction (CW vs CCW)
                # without changing scale or x range.
                p[1] = direction * p[1]
                positions[k] = p
            p0 = positions[0]
            p2 = positions[min(2, steps_per - 1)]
            psi0 = float(np.arctan2(p2[1] - p0[1], p2[0] - p0[0]))
            init_state = _init_state(p0, psi0)
            for noise in noise_levels:
                env = make_env()
                if hasattr(env, "reset"):
                    env.reset(state=init_state.copy())
                if domain_rand and hasattr(env, "set_perturbation"):
                    if kind == "auv":
                        env.set_perturbation(
                            mass_mult=float(rng.uniform(dr_mass_lo, dr_mass_hi)),
                            drag_mult=float(rng.uniform(dr_drag_lo, dr_drag_hi)))
                    else:
                        env.set_perturbation(
                            mass_mult=float(rng.uniform(dr_mass_lo, dr_mass_hi)),
                            drag_mult=float(rng.uniform(dr_drag_lo, dr_drag_hi)),
                            wind_x=float(rng.normal(0.0, dr_wind_sigma)),
                            wind_y=float(rng.normal(0.0, dr_wind_sigma)),
                            wind_z=float(rng.normal(0.0, dr_wind_sigma * 0.3)))
                ctrl = LQR3D(env, kind=kind)
                inp_buf, tgt_buf = [], []
                for k in range(steps_per - 1):
                    state_t = env.get_state()
                    # CRITICAL: build the feature with `desired_next`,
                    # NOT the realised `state_tp1`.  The deployed NN
                    # controller calls `build_inverse_dynamics_input(
                    # state, desired_next)` at inference, so the
                    # training feature distribution must match.
                    desired_next = env.compute_desired_next_state(
                        state_t, positions, k, dt)
                    ctrl.setpoint = positions[k + 1, :3]
                    action = ctrl.compute(state_t)
                    if noise > 0.0:
                        action = action + rng.randn(4) * noise * (action_hi - action_lo) * 0.02
                    action = np.clip(action, action_lo, action_hi)
                    env.step(action)
                    feat = env.build_inverse_dynamics_input(
                        state_t, desired_next, feature_mode=feature_mode)
                    inp_buf.append(feat)
                    tgt_buf.append(action.copy())
                all_inputs.append(np.array(inp_buf))
                all_targets.append(np.array(tgt_buf))
            if (i + 1) % 8 == 0 or i == len(traj_configs) - 1:
                n = sum(len(x) for x in all_inputs)
                print(f"    [{i+1}/{len(traj_configs)}] {n:,} transitions")

    # --- Strategy 2: Closed-loop PE-setpoint tracking ---
    # Replaces the previous open-loop sinusoidal-action sweep, which
    # produced biased labels: the open-loop drone tilts and drifts
    # within ~1 s of rollout start (action_lo,hi has |tau| ~ 0.05 N·m
    # but the strategy injected ±0.012 N·m sinusoids), but the action
    # labels remain near {m·g, 0, 0, 0} — disconnected from the rapidly
    # diverging state.  The NN trained on those pairs learns "for any
    # state, output ≈ m·g for thrust, ≈ 0 for torques," which the
    # iter-27 diagnostic confirmed (T residual_std = 1074 % of LQR std).
    #
    # The new strategy uses exciting_3d sum-of-sinusoids as a position
    # SETPOINT, then has LQR3D close the loop.  Benefits:
    #   - drone stays inside the hover linearisation region
    #   - features are built from desired_next, matching inference
    #   - labels are physically-grounded inverse-dynamics actions
    n_pe = 2 if demo else (8 if fast else 24)
    if kind == "auv":
        pe_scales = [0.5, 0.8, 1.2]
        pe_speeds = [0.04, 0.07, 0.10]
    else:
        pe_scales = [0.25, 0.40, 0.55]
        pe_speeds = [0.20, 0.35, 0.50]
    print(f"  3-D Strategy 2 ({kind}): {n_pe} closed-loop PE-setpoint rollouts")
    for j in range(n_pe):
        pe_scale = float(rng.choice(pe_scales))
        pe_speed = float(rng.choice(pe_speeds))
        pe_seed  = int(rng.randint(0, 1_000_000))
        positions = np.zeros((steps_per, 3), dtype=np.float64)
        for k in range(steps_per):
            p = exciting_3d(k, scale=pe_scale, speed=pe_speed, seed=pe_seed)
            # exciting_3d centres z at +3.0; shift to platform z_off.
            p[2] += z_off - 3.0
            positions[k] = p
        p0 = positions[0]
        p2 = positions[min(2, steps_per - 1)]
        psi0 = float(np.arctan2(p2[1] - p0[1], p2[0] - p0[0]))
        init_state = _init_state(p0, psi0)
        env = make_env()
        if hasattr(env, "reset"):
            env.reset(state=init_state.copy())
        if domain_rand and hasattr(env, "set_perturbation"):
            if kind == "auv":
                env.set_perturbation(
                    mass_mult=float(rng.uniform(dr_mass_lo, dr_mass_hi)),
                    drag_mult=float(rng.uniform(dr_drag_lo, dr_drag_hi)))
            else:
                env.set_perturbation(
                    mass_mult=float(rng.uniform(dr_mass_lo, dr_mass_hi)),
                    drag_mult=float(rng.uniform(dr_drag_lo, dr_drag_hi)),
                    wind_x=float(rng.normal(0.0, dr_wind_sigma)),
                    wind_y=float(rng.normal(0.0, dr_wind_sigma)),
                    wind_z=float(rng.normal(0.0, dr_wind_sigma * 0.3)))
        ctrl = LQR3D(env, kind=kind)
        inp_buf, tgt_buf = [], []
        for k in range(steps_per - 1):
            state_t = env.get_state()
            desired_next = env.compute_desired_next_state(
                state_t, positions, k, dt)
            ctrl.setpoint = positions[k + 1, :3]
            action = ctrl.compute(state_t)
            action = np.clip(action, action_lo, action_hi)
            env.step(action)
            feat = env.build_inverse_dynamics_input(
                state_t, desired_next, feature_mode=feature_mode)
            inp_buf.append(feat)
            tgt_buf.append(action.copy())
        all_inputs.append(np.array(inp_buf))
        all_targets.append(np.array(tgt_buf))
        if (j + 1) % 4 == 0 or j == n_pe - 1:
            n = sum(len(x) for x in all_inputs)
            print(f"    [{j+1}/{n_pe}] {n:,} cumulative transitions")

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    idx = rng.permutation(len(inputs))
    inputs, targets = inputs[idx], targets[idx]
    print(f"\n  Total ({kind}): {len(inputs):,} transitions  "
          f"(input_dim={inputs.shape[1]}, output_dim={targets.shape[1]})")
    return inputs, targets
