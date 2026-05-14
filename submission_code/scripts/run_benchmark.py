"""Multi-platform benchmark suite for the paper.

Runs LQR, NN-Frozen, ER, EWC, Evidential (fixed-λ), and ACE on three
platforms (Unicycle, AUV-3D, Drone-3D) under five scenarios (Nominal,
MassShift, Disturbance, SensorNoise, ActuatorNoise) and writes a CSV
of per-run mean / max / RMS tracking error plus a wide-form summary
and per-trajectory JSON traces.

LQR uses only the nominal linearisation (same physics knowledge as
the NN's K-feedback), so the comparison is symmetric.

Usage
-----
    python scripts/run_benchmark.py                  # 3 seeds (default)
    python scripts/run_benchmark.py --quick          # 1 seed
    python scripts/run_benchmark.py --seeds 5

Outputs (default --out-dir = results/benchmark)
-----------------------------------------------
    multi_platform_runs.csv     long-form rows
    multi_platform_summary.csv  wide-form mean ± std
    multi_platform_traces.json  per-(platform, method, scenario) trajectories
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.unicycle_env import UnicycleEnv
from envs.auv3d_env import AUV3DEnv
from envs.drone3d_env import Drone3DEnv
from controllers.lqr_controller import LQRController
from controllers.lqr_3d import LQR3D
from controllers.nn_controller import NNController
from training.train_online import UniformCLAdapter, ACEAdapter
from training.ewc import EWCAdapter
from utils.trajectories import (
    generate_trajectory, lemniscate_2d, lemniscate_3d,
)
from models.evidential_net import EvidentialNet


# ----------------------------------------------------------------------
# Platform table: (env factory, trajectory factory, model checkpoint,
# default dt, mass-shift magnitude, kind for LQR3D)
# ----------------------------------------------------------------------

PLATFORMS = ["unicycle", "auv", "drone"]
PLATFORM_PRETTY = {"unicycle": "Unicycle", "auv": "AUV", "drone": "Drone"}

PLATFORM_CFG = {
    # Each platform's total_time is chosen so the rollout covers ≥ 1 full
    # lemniscate period (period ≈ 2π / speed in the parameterisation of
    # `lemniscate_3d` / `lemniscate_2d`).  This makes the trajectory
    # plots show a complete closed loop instead of a fragment.
    #   Unicycle  speed=0.50 → T≈12.6 s, 30 s ⇒ 2.4 periods (kept).
    #   AUV       speed=0.08 → T≈78.5 s, 90 s ⇒ 1.15 periods.
    #   Drone     speed=0.12 → T≈52.4 s, 60 s ⇒ 1.15 periods.
    "unicycle": dict(dt=0.02, ckpt="pretrained/unicycle.pth",
                     mass_mult=2.0, drag_mult=1.4,
                     total_time=60.0),
    "auv":      dict(dt=0.05, ckpt="pretrained/auv.pth",
                     mass_mult=1.10, drag_mult=1.4,
                     total_time=90.0),
    "drone":    dict(dt=0.02, ckpt="pretrained/drone.pth",
                     mass_mult=1.10, drag_mult=1.3,
                     total_time=60.0),
}


def _make_nominal_env(platform: str):
    """Fresh env with default parameters — no perturbation."""
    if platform == "unicycle":
        return UnicycleEnv(dt=PLATFORM_CFG["unicycle"]["dt"])
    if platform == "auv":
        return AUV3DEnv(dt=PLATFORM_CFG["auv"]["dt"])
    if platform == "drone":
        return Drone3DEnv(dt=PLATFORM_CFG["drone"]["dt"])
    raise ValueError(platform)


# Per-platform sensor noise σ.  Calibrated so the perturbation is
# of the same fractional scale as the unicycle's std_pos=0.02 m on a
# state with O(1 m) excursions (i.e. ≈ 2 % of trajectory amplitude).
_SENSOR_NOISE = {
    "unicycle": dict(std_pos=0.02, std_vel=0.03),
    "auv":      dict(std_pos=0.05, std_vel=0.05),
    "drone":    dict(std_pos=0.02, std_vel=0.03, std_ori=0.02),
}
# Per-platform actuator noise σ (added to commanded action).  Either
# a scalar (broadcast to every channel) or an array matching the
# action dimension.  The drone uses a per-channel vector because
# (T, tau_x, tau_y, tau_z) span very different magnitudes.
_ACTUATOR_NOISE = {
    "unicycle": 0.5,
    "auv":      2.0,
    "drone":    np.array([0.3, 0.10, 0.10, 0.05]),
}


def _wrap_actuator_noise(env, sigma, seed: int):
    """Monkey-patch env.step to add Gaussian noise to the commanded
    action before the underlying dynamics integrate it.

    ``sigma`` is broadcast to the action's shape so a scalar applies
    uniformly while an array gives per-channel std.
    """
    rng = np.random.default_rng(seed)
    orig_step = env.step
    sigma_vec = np.asarray(sigma, dtype=np.float64)
    def noisy_step(action):
        a = np.asarray(action, dtype=np.float64).copy()
        a = a + rng.normal(0.0, 1.0, size=a.shape) * sigma_vec
        return orig_step(a)
    env.step = noisy_step
    return env


def _make_perturbed_env(platform: str, scenario: str, seed: int):
    """Construct a perturbed env according to the scenario."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = _make_nominal_env(platform)
    if scenario == "Nominal":
        return env
    if scenario == "MassShift":
        mm = PLATFORM_CFG[platform]["mass_mult"]
        if hasattr(env, "set_uncertainty"):
            env.set_uncertainty(True, mass_multiplier=mm)
        else:
            env._mass_multiplier = mm
        return env
    if scenario == "Disturbance":
        if platform == "unicycle":
            # Match the cold-start unicycle Disturbance scenario.
            env.set_wind(1.5, 0.75)
        elif platform == "auv":
            # AUV has no exogenous wind; we instead apply heavier
            # hydrodynamic drag — a slowly-varying disturbance the
            # NN can in principle adapt to but LQR cannot.
            dm = PLATFORM_CFG["auv"]["drag_mult"]
            env.set_perturbation(mass_mult=1.0, drag_mult=dm)
        elif platform == "drone":
            # Wind gust only — keep mass nominal so the disturbance
            # is purely exogenous (LQR's model has no notion of
            # wind, exactly the regime where the adaptive NN feed-
            # forward should win).
            env.set_perturbation(mass_mult=1.0, drag_mult=1.0,
                                 wind_x=0.4, wind_y=0.2, wind_z=-0.2)
        return env
    if scenario == "SensorNoise":
        env.set_noise(**_SENSOR_NOISE[platform])
        return env
    if scenario == "ActuatorNoise":
        return _wrap_actuator_noise(env, _ACTUATOR_NOISE[platform], seed)
    raise ValueError(f"Unknown scenario: {scenario}")


SCENARIOS = ["Nominal", "MassShift", "Disturbance",
             "SensorNoise", "ActuatorNoise"]
SCENARIO_PRETTY = {"Nominal": "Nominal", "MassShift": "Mass shift",
                   "Disturbance": "Disturbance",
                   "SensorNoise": "Sensor noise",
                   "ActuatorNoise": "Actuator noise"}


# ----------------------------------------------------------------------
# Trajectory + initial state
# ----------------------------------------------------------------------


def _make_trajectory(platform: str, total_time: float):
    dt = PLATFORM_CFG[platform]["dt"]
    n_steps = int(total_time / dt)
    if platform == "unicycle":
        fn = lambda t: lemniscate_2d(t, scale=4.0, speed=0.5)
        positions, _ = generate_trajectory(fn, dt, n_steps, r_bar=4)
        x0, y0 = positions[0]
        ddx = positions[2][0] - positions[0][0]
        ddy = positions[2][1] - positions[0][1]
        theta0 = np.arctan2(ddy, ddx)
        init = np.array([x0, y0, theta0, 0.0, 0.0])
        return positions, n_steps, init, dt
    if platform == "auv":
        fn = lambda t: lemniscate_3d(t, scale_xy=1.2, scale_z=0.3,
                                     z_offset=-3.0, speed=0.08)
    else:  # drone (quadrotor)
        fn = lambda t: lemniscate_3d(t, scale_xy=0.5, scale_z=0.2,
                                     z_offset=3.0, speed=0.12)
    positions, _ = generate_trajectory(fn, dt, n_steps, r_bar=4)
    p0 = positions[0]
    p2 = positions[min(2, len(positions) - 1)]
    psi0 = float(np.arctan2(p2[1] - p0[1], p2[0] - p0[0]))
    if platform == "drone":
        # Quadrotor: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        init = np.array([p0[0], p0[1], p0[2],
                         0.0, 0.0, psi0,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0], dtype=np.float64)
    else:
        # AUV: [x, y, z, psi, u, v, w, r]
        init = np.array([p0[0], p0[1], p0[2], psi0,
                         0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return positions, n_steps, init, dt


# ----------------------------------------------------------------------
# Method factory
# ----------------------------------------------------------------------


METHODS = ["LQR", "Frozen", "ER", "EWC", "EVIDENTIAL", "ACE"]
METHOD_PRETTY = {
    "LQR":         "LQR (nominal)",
    "Frozen":      "NN Frozen",
    "ER":          "NN ER",
    "EWC":         "NN EWC",
    "EVIDENTIAL":  "NN Evidential (fixed λ)",
    "ACE":         "NN ACE (ours)",
}


def _load_model(platform: str) -> EvidentialNet:
    path = os.path.join(ROOT, PLATFORM_CFG[platform]["ckpt"])
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    use_ln = bool(ckpt.get("use_layernorm", False))
    model = EvidentialNet(ckpt["input_dim"], ckpt["output_dim"],
                          hidden_dims=ckpt.get("hidden_dims", [256, 256, 256, 256]),
                          use_layernorm=use_ln)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.input_mean  = ckpt["input_mean"]
    model.input_std   = ckpt["input_std"]
    model.target_mean = ckpt["target_mean"]
    model.target_std  = ckpt["target_std"]
    model.feature_mode = ckpt.get("feature_mode", "full")
    model.eval()
    return model


def _make_lqr_baseline(platform: str, nominal_env, dt: float):
    """LQR from nominal linearisation only — no privileged knowledge.

    Symmetric K: the LQR baseline uses the SAME Q matrix as the K
    inside NN methods, ensuring a fair comparison.
    """
    if platform == "unicycle":
        return LQRController(nominal_env, dt)
    kind = "auv" if platform == "auv" else "drone"
    return LQR3D(nominal_env, kind=kind)


def _make_controller(method: str, platform: str, model: EvidentialNet,
                     env, dt: float):
    """Construct the controller / adapter for the requested method.

    K-feedback is computed from the *nominal* env so all NN methods get
    the same physics knowledge as LQR (a symmetric comparison).
    """
    nominal_env = _make_nominal_env(platform)
    feature_mode = getattr(model, "feature_mode", "full")
    fb_gain = 1.0

    if method == "LQR":
        return _make_lqr_baseline(platform, nominal_env, dt)
    if method == "Frozen":
        return NNController(model, device="cpu", feature_mode=feature_mode,
                            feedback_gain=fb_gain, feedback_env=nominal_env,
                            feedback_dt=dt)
    if method in ("CL", "ER"):
        return UniformCLAdapter(model, env, dt, lr=3e-4, lambda_reg=0.001,
                                update_every=5, min_buffer_size=64,
                                batch_size=32, buffer_capacity=1000,
                                update_threshold=0.02, feature_mode=feature_mode,
                                feedback_gain=fb_gain, feedback_env=nominal_env)
    if method == "EWC":
        return EWCAdapter(model, env, dt, lr=3e-4,
                          update_every=5, min_buffer_size=64,
                          batch_size=32, buffer_capacity=1000,
                          update_threshold=0.02, feature_mode=feature_mode,
                          feedback_gain=fb_gain, feedback_env=nominal_env,
                          lambda_ewc=1.0, num_fisher_samples=2048)

    # ACE family (fixed-λ DER baseline vs full ACE with adaptive
    # schedule + sustained-bias detector). Shared hyperparameters:
    common = dict(
        lr=1e-4, lambda_kappa=0.05, lambda_floor=0.10,
        anchor_strength=5e-3,
        batch_size=32, buffer_capacity=1000,
        update_every=5, min_buffer_size=64,
        feature_mode=feature_mode,
        feedback_gain=fb_gain, feedback_env=nominal_env,
        loss_form="tempered",
    )
    if method in ("EDL", "ACE_FIXED", "EVIDENTIAL"):
        # Fixed-λ DER baseline (lambda_eta=0, no shift detector).
        return ACEAdapter(model, env, dt,
                          lambda_eta=0.0,
                          lambda_schedule_init=0.5,
                          shift_detection=False,
                          **common)
    if method == "ACE":
        # Full ACE: adaptive λ schedule + L2 anchor + priority replay
        # + sustained-bias detector.
        return ACEAdapter(model, env, dt,
                          lambda_eta=0.10,
                          lambda_schedule_init=1.0,
                          shift_detection=True,
                          shift_threshold=2.0,
                          shift_lr_boost=3.0,
                          **common)
    raise ValueError(method)


# ----------------------------------------------------------------------
# Single-rollout runner
# ----------------------------------------------------------------------


def _state_xy(state, platform):
    """Position component of the state matching the trajectory dimension."""
    if platform == "unicycle":
        return state[:2]
    return state[:3]


def _step_lqr(controller, env, state, positions, k, platform, n_steps):
    """Compute and apply LQR action.  Different APIs for 2-D vs 3-D."""
    if platform == "unicycle":
        action, _ = controller.compute(state, positions, k)
        env.step(action)
    else:
        # LQR3D: aim for the next reference waypoint
        sp = positions[min(k + 3, n_steps - 1)]
        action = controller.compute(state, setpoint=sp)
        env.step(action)
    return action


def run_one(platform: str, method: str, scenario: str, model, seed: int,
            total_time: float):
    positions, n_steps, init_state, dt = _make_trajectory(platform, total_time)
    env = _make_perturbed_env(platform, scenario, seed)
    # Seed the env (initial-state perturbation + observation noise stream)
    # AND honour the explicit init state used to align with the reference
    # trajectory. The init perturbation lives inside reset() only when no
    # explicit state is supplied; here we add the perturbation manually
    # so the seed actually affects LQR / Frozen / ER trajectories.
    env.reset(seed=seed)
    init_perturbed = init_state.copy()
    if hasattr(env, "_rng"):
        # Small initial-state offset, scaled to platform geometry.
        if platform == "unicycle":
            init_perturbed[:3] += env._rng.uniform(-0.05, 0.05, size=3)
        elif platform == "auv":
            init_perturbed[:3] += env._rng.uniform(-0.10, 0.10, size=3)
            init_perturbed[3]  += env._rng.uniform(-0.087, 0.087)
        elif platform == "drone":
            init_perturbed[:3] += env._rng.uniform(-0.10, 0.10, size=3)
            init_perturbed[3:6] += env._rng.uniform(-0.087, 0.087, size=3)
    env.reset(state=init_perturbed)
    ctrl = _make_controller(method, platform, model, env, dt)

    errs = np.empty(n_steps, dtype=np.float64)
    epist = np.zeros(n_steps, dtype=np.float64)
    lam_sched = np.full(n_steps, np.nan, dtype=np.float64)
    actions = np.zeros((n_steps, len(getattr(env, "ACTION_LIMITS", [(0, 0), (0, 0)]))),
                       dtype=np.float64) if hasattr(env, "ACTION_LIMITS") else None
    xy = np.zeros((n_steps, positions.shape[1]), dtype=np.float64)

    # Under SensorNoise the controller must observe the noisy state;
    # tracking error is still measured against the noise-free truth.
    use_obs = (scenario == "SensorNoise" and hasattr(env, "get_observation"))

    t0 = time.perf_counter()
    for k in range(n_steps):
        state = env.get_observation() if use_obs else env.get_state()
        if method == "LQR":
            a = _step_lqr(ctrl, env, state, positions, k, platform, n_steps)
            info = {}
        elif method == "Frozen":
            a, info = ctrl.compute(env, state, positions, k, dt)
            env.step(a)
        else:
            a, info = ctrl.step(env, state, positions, k)
        if actions is not None and len(a) == actions.shape[1]:
            actions[k] = a
        true_state = env.get_state()
        xy[k] = _state_xy(true_state, platform)
        errs[k] = float(np.linalg.norm(xy[k] - positions[k]))
        if "epistemic" in info:
            epist[k] = info["epistemic"]
        if "lambda_schedule" in info:
            lam_sched[k] = info["lambda_schedule"]
        if not np.isfinite(errs[k]) or errs[k] > 50.0:
            errs[k:] = errs[k]
            xy[k:] = xy[k]
            break
    runtime = (time.perf_counter() - t0) / n_steps * 1000.0

    skip = int(2.0 / dt)
    return {
        "platform": platform, "method": method, "scenario": scenario,
        "seed": seed,
        "mean_err": float(errs[skip:].mean()),
        "max_err":  float(errs[skip:].max()),
        "rms_err":  float(np.sqrt((errs[skip:] ** 2).mean())),
        "runtime_ms_per_step": runtime,
        "errs":  errs.tolist(),
        "epist": epist.tolist(),
        "lambda_schedule": lam_sched.tolist(),
        "xy":    xy.tolist(),
        "positions": positions.tolist(),
        "dt":    dt,
    }


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=42,
                   help="First seed value (seeds run from offset to offset+seeds-1)")
    p.add_argument("--quick", action="store_true", help="single seed")
    p.add_argument("--platforms", nargs="+", default=PLATFORMS)
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--out-dir", default="results/benchmark")
    p.add_argument("--ckpt-dir", default=None,
                   help="Override directory for the platform checkpoints "
                        "(default: pretrained/).")
    args = p.parse_args(argv)

    if args.quick:
        args.seeds = 1

    if args.ckpt_dir is not None:
        for plat, cfg in PLATFORM_CFG.items():
            old = cfg["ckpt"]
            base = os.path.basename(old)
            cfg["ckpt"] = os.path.join(args.ckpt_dir, base)
        print(f"[ckpt-dir override] reading checkpoints from {args.ckpt_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nMulti-platform benchmark: {len(args.platforms)} platforms × "
          f"{len(args.scenarios)} scenarios × {len(args.methods)} methods × "
          f"{args.seeds} seeds = "
          f"{len(args.platforms)*len(args.scenarios)*len(args.methods)*args.seeds} runs")

    rows: List[Dict] = []
    traces: List[Dict] = []

    for plat in args.platforms:
        print(f"\n=== Platform: {PLATFORM_PRETTY[plat]} ===")
        model = _load_model(plat)
        total_time = PLATFORM_CFG[plat]["total_time"]
        for sc in args.scenarios:
            print(f"-- Scenario: {sc} --")
            for m in args.methods:
                mean_errs = []
                for seed in range(args.seeds):
                    r = run_one(plat, m, sc, model,
                                args.seed_offset + seed, total_time)
                    mean_errs.append(r["mean_err"])
                    rows.append({
                        "platform": plat, "method": m, "scenario": sc,
                        "seed": r["seed"],
                        "mean_err": r["mean_err"],
                        "max_err":  r["max_err"],
                        "rms_err":  r["rms_err"],
                        "runtime_ms_per_step": r["runtime_ms_per_step"],
                    })
                    if seed == 0:
                        traces.append({
                            "platform": plat, "method": m, "scenario": sc,
                            "errs": r["errs"], "epist": r["epist"],
                            "lambda_schedule": r["lambda_schedule"],
                            "xy":   r["xy"],   "positions": r["positions"],
                            "dt":   r["dt"],
                        })
                mean = float(np.mean(mean_errs))
                std  = float(np.std(mean_errs))
                print(f"   {METHOD_PRETTY[m]:>20s}: {mean:.4f} ± {std:.4f} m")

    # ---- CSV ----
    long_path = os.path.join(args.out_dir, "multi_platform_runs.csv")
    with open(long_path, "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nWrote {long_path}")

    # ---- Wide summary: rows = (platform, method), columns = scenarios ----
    summary_path = os.path.join(args.out_dir, "multi_platform_summary.csv")
    with open(summary_path, "w") as f:
        cols = args.scenarios
        f.write("platform,method," + ",".join(cols) + "\n")
        for plat in args.platforms:
            for m in args.methods:
                vals = []
                for sc in args.scenarios:
                    seq = [r["mean_err"] for r in rows
                           if r["platform"] == plat and r["method"] == m
                           and r["scenario"] == sc]
                    if seq:
                        vals.append(f"{np.mean(seq):.4f}±{np.std(seq):.4f}")
                    else:
                        vals.append("-")
                f.write(f"{plat},{METHOD_PRETTY[m]}," + ",".join(vals) + "\n")
    print(f"Wrote {summary_path}")

    # ---- JSON traces ----
    traces_path = os.path.join(args.out_dir, "multi_platform_traces.json")
    with open(traces_path, "w") as f:
        json.dump({"traces": traces, "platforms": args.platforms,
                   "scenarios": args.scenarios, "methods": args.methods}, f)
    print(f"Wrote {traces_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
