"""Render five 30-second MP4 clips for the website's demo section.

Each clip animates three controllers (LQR, Frozen, ACE) on the unicycle
under one disturbance regime. Output: website/assets/animations/<name>.mp4.

Usage:
    python3 scripts/render_demo_clips.py                 # render all five
    python3 scripts/render_demo_clips.py --scenario wind # render one
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.unicycle_env import UnicycleEnv
from utils.trajectories import generate_trajectory, lemniscate_2d


def build_trajectory(total_time: float, dt: float):
    """Build the lemniscate reference and matching init state.

    Returns
    -------
    positions : ndarray, shape (n_steps, 2)
        Desired x, y of the unicycle at each timestep.
    n_steps : int
        Number of simulation steps in the rollout.
    init : ndarray, shape (5,)
        Initial unicycle state [x, y, theta, v, omega], aligned so the
        unicycle starts on the reference and heading along the curve.
    """
    n_steps = int(total_time / dt)
    fn = lambda t: lemniscate_2d(t, scale=4.0, speed=0.5)
    positions, _ = generate_trajectory(fn, dt, n_steps, r_bar=4)
    # Trim to exactly n_steps (generate_trajectory returns extra lookahead points)
    positions = positions[:n_steps]
    x0, y0 = positions[0]
    ddx = positions[2][0] - positions[0][0]
    ddy = positions[2][1] - positions[0][1]
    theta0 = float(np.arctan2(ddy, ddx))
    init = np.array([x0, y0, theta0, 0.0, 0.0])
    return positions, n_steps, init


from models.evidential_net import EvidentialNet


def load_unicycle_model(ckpt_path: str | None = None) -> EvidentialNet:
    """Load the pretrained unicycle inverse-dynamics network.

    Mirrors run_benchmark.py's loader so the demo clips show exactly the
    same controller behaviour the benchmark numbers report.
    """
    if ckpt_path is None:
        ckpt_path = str(ROOT / "pretrained" / "unicycle.pth")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model = EvidentialNet(
        ckpt["input_dim"], ckpt["output_dim"],
        hidden_dims=ckpt.get("hidden_dims", [256, 256, 256, 256]),
        use_layernorm=bool(ckpt.get("use_layernorm", False)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.input_mean  = ckpt["input_mean"]
    model.input_std   = ckpt["input_std"]
    model.target_mean = ckpt["target_mean"]
    model.target_std  = ckpt["target_std"]
    model.feature_mode = ckpt.get("feature_mode", "full")
    model.eval()
    return model


from controllers.lqr_controller import LQRController
from controllers.nn_controller import NNController
from training.train_online import ACEAdapter


def make_controller(method: str, model: EvidentialNet, env, dt: float):
    """Instantiate one of (LQR, Frozen, ACE) on the unicycle.

    Returns
    -------
    (controller, kind) where kind is 'lqr' | 'nn' | 'adapter'.  The kind
    string tells the runner how to dispatch the per-step call.
    """
    feature_mode = getattr(model, "feature_mode", "full")
    nominal_env = UnicycleEnv(dt=dt)
    if method == "LQR":
        return LQRController(env, dt), "lqr"
    if method == "Frozen":
        return NNController(
            model, device="cpu", feature_mode=feature_mode,
            feedback_gain=1.0, feedback_env=nominal_env, feedback_dt=dt,
        ), "nn"
    if method == "ACE":
        return ACEAdapter(
            model, env, dt,
            lr=1e-4, lambda_eta=0.10, lambda_kappa=0.05,
            lambda_floor=0.10, anchor_strength=5e-3,
            batch_size=32, buffer_capacity=1000,
            update_every=5, min_buffer_size=64,
            feature_mode=feature_mode,
            feedback_gain=1.0, feedback_env=nominal_env,
            loss_form="tempered",
            lambda_schedule_init=1.0,
            shift_detection=True,
            shift_threshold=2.0,
            shift_lr_boost=3.0,
        ), "adapter"
    raise ValueError(f"Unknown method: {method!r}")

SCENARIOS = {
    "nominal":  dict(
        title="Nominal",
        caption="No disturbance. ACE preserves the offline solution; λ stays near 1.",
        shift={},
    ),
    "mass_shift": dict(
        title="Mass shift",
        caption="Payload doubles. ε rises, λ falls; ACE adapts back. LQR drifts.",
        shift=dict(mass=2.0),
    ),
    "disturbance": dict(
        title="Disturbance",
        caption="Cross-wind. LQR has no integral action and drifts; ACE compensates.",
        shift=dict(wind_x=1.5, wind_y=0.75),
    ),
    "sensor_noise": dict(
        title="Sensor noise",
        caption="Position σ = 2 cm. Aleatoric rises; ACE keeps λ high and does not adapt to noise.",
        shift=dict(sensor_pos=0.02, sensor_vel=0.03),
    ),
    "actuator_noise": dict(
        title="Actuator noise",
        caption="Action σ = 0.5 injected. The residual is noise, not bias; ACE holds back.",
        shift=dict(actuator_sigma=0.5),
    ),
}

# Per-scenario ACE-vs-Frozen mean-error gate.  ("le", k) means
# ACE <= k * Frozen — a "true win" gate; ("within", k) means
# ACE <= k * Frozen — a "no harm" gate.  Both forms reduce to the
# same assertion but the kind label documents intent for readers.
SCENARIO_GATES = {
    "nominal":        ("within", 1.5),
    "mass_shift":     ("le",     1.0),
    "disturbance":    ("le",     0.6),
    "sensor_noise":   ("within", 1.1),
    "actuator_noise": ("within", 2.0),
}


def _wrap_actuator_noise(env, sigma, seed: int):
    """Monkey-patch env.step to add Gaussian noise to the commanded action.

    Mirrors scripts/run_benchmark.py:_wrap_actuator_noise so the demo and the
    paper benchmark see the same disturbance.
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


def _apply_shift(env: UnicycleEnv, shift: dict, seed: int) -> UnicycleEnv:
    """Apply a scenario's shift kwargs to a fresh unicycle env.

    Returns the (possibly wrapped) env so callers always use the post-shift
    reference — the actuator-noise branch wraps env.step.
    """
    if "mass" in shift or "friction" in shift:
        env.set_uncertainty(
            True,
            mass_multiplier=shift.get("mass", 1.0),
            friction_multiplier=shift.get("friction", 1.0),
        )
    if "wind_x" in shift or "wind_y" in shift:
        env.set_wind(shift.get("wind_x", 0.0), shift.get("wind_y", 0.0))
    if "sensor_pos" in shift or "sensor_vel" in shift:
        env.set_noise(
            std_pos=shift.get("sensor_pos", 0.0),
            std_vel=shift.get("sensor_vel", 0.0),
        )
    if "actuator_sigma" in shift:
        env = _wrap_actuator_noise(env, shift["actuator_sigma"], seed)
    return env


def run_scenario(scenario_name: str, model: EvidentialNet,
                 total_time: float = 30.0, dt: float = 0.02,
                 seed: int = 42) -> dict:
    """Run all three methods on one scenario; return per-step data
    needed by the renderer (positions, errors, xy, eps, ale, lambda).
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_name!r}")
    spec = SCENARIOS[scenario_name]
    positions, n_steps, init = build_trajectory(total_time, dt)
    out = {"positions": positions, "dt": dt, "title": spec["title"],
           "caption": spec["caption"]}
    for method in ("LQR", "Frozen", "ACE"):
        np.random.seed(seed); torch.manual_seed(seed)
        env = UnicycleEnv(dt=dt)
        env = _apply_shift(env, spec["shift"], seed)
        env.reset(seed=seed); env.reset(state=init)
        ctrl, kind = make_controller(method, model, env, dt)
        errs  = np.empty(n_steps); xy = np.zeros((n_steps, 2))
        epist = np.zeros(n_steps); aleat = np.zeros(n_steps)
        lamb  = np.full(n_steps, np.nan)
        shift_sig = np.zeros(n_steps); updates = np.zeros(n_steps)
        use_obs = (scenario_name == "sensor_noise"
                   and hasattr(env, "get_observation"))
        for k in range(n_steps):
            state = env.get_observation() if use_obs else env.get_state()
            if kind == "lqr":
                a, _ = ctrl.compute(state, positions, k); env.step(a); info = {}
            elif kind == "nn":
                a, info = ctrl.compute(env, state, positions, k, dt); env.step(a)
            else:
                a, info = ctrl.step(env, state, positions, k)
            truth = env.get_state()
            xy[k] = truth[:2]
            errs[k] = float(np.linalg.norm(xy[k] - positions[k]))
            epist[k] = info.get("epistemic", 0.0) or 0.0
            aleat[k] = info.get("aleatoric", 0.0) or 0.0
            lamb[k]  = info.get("lambda_schedule", np.nan)
            # ACE-only diagnostics straight off the adapter object.
            if kind == "adapter":
                shift_sig[k] = float(getattr(ctrl, "shift_signal", 0.0))
                updates[k]   = int(getattr(ctrl, "update_count", 0))
            if not np.isfinite(errs[k]) or errs[k] > 50.0:
                errs[k:] = errs[k]; xy[k:] = xy[k]; break
        out[method] = dict(errs=errs, xy=xy, epist=epist, aleat=aleat,
                           lamb=lamb, shift_sig=shift_sig, updates=updates)
    return out


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from scripts._demo_palette import (
    apply_style, METHOD_COLORS, METHOD_ORDER,
    COL_BG, COL_PANEL, COL_REF, COL_TEXT, COL_MUTED, COL_ACE,
)


def build_animation(data: dict, render_fps: int = 30) -> tuple:
    """Build the three-panel matplotlib figure + FuncAnimation for a scenario.

    Returns (fig, anim).  Save with anim.save(path, writer=FFMpegWriter(fps=render_fps)).
    """
    apply_style()
    dt = data["dt"]
    n_steps = data["positions"].shape[0]
    sim_fps = int(round(1.0 / dt))
    stride = max(1, sim_fps // render_fps)   # subsample so we don't render every sim step
    frame_idx = np.arange(0, n_steps, stride)
    n_frames = len(frame_idx)
    t_full = np.arange(n_steps) * dt

    # --- figure layout ----------------------------------------------------
    fig = plt.figure(figsize=(14.4, 5.6), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.8, 1.0, 1.0], wspace=0.30)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.16)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_err  = fig.add_subplot(gs[0, 1])
    inner_gs = gs[0, 2].subgridspec(2, 1, hspace=0.30, height_ratios=[1.0, 1.0])
    ax_unc = fig.add_subplot(inner_gs[0, 0])
    ax_eng = fig.add_subplot(inner_gs[1, 0], sharex=ax_unc)

    # Title and caption — set once.
    fig.suptitle(data["title"], y=0.96, fontsize=20, weight="bold", color=COL_TEXT)
    fig.text(0.5, 0.02, data["caption"], ha="center",
             fontsize=13, style="italic", color=COL_MUTED)

    # --- trajectory panel -------------------------------------------------
    pos = data["positions"]
    ax_traj.plot(pos[:, 0], pos[:, 1], "--", color="#1f2937", lw=2.2,
                 alpha=0.9, zorder=1, label="reference")
    ax_traj.set_aspect("equal"); ax_traj.set_xlim(-4.8, 4.8); ax_traj.set_ylim(-2.7, 2.7)
    ax_traj.set_xticks([]); ax_traj.set_yticks([])
    ax_traj.set_title("Trajectory", color=COL_MUTED)
    for spine in ax_traj.spines.values():
        spine.set_color(COL_PANEL)
    LW_MAP = {"LQR": 2.4, "Frozen": 2.4, "ACE": 3.6}
    MS_MAP = {"LQR": 10,  "Frozen": 10,  "ACE": 14}
    trails = {m: ax_traj.plot([], [], color=METHOD_COLORS[m], lw=LW_MAP[m],
                              alpha=0.95, zorder=3, label=m)[0]
              for m in METHOD_ORDER}
    dots   = {m: ax_traj.plot([], [], "o", color=METHOD_COLORS[m], ms=MS_MAP[m],
                              mec="white", mew=1.0, zorder=4)[0]
              for m in METHOD_ORDER}
    handles, labels = ax_traj.get_legend_handles_labels()
    # put ACE first in the legend even though it was plotted last.
    order = [labels.index("ACE"), labels.index("Frozen"), labels.index("LQR"),
             labels.index("reference")]
    ax_traj.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="lower right", fontsize=12, frameon=False,
                   labelcolor=COL_TEXT)

    # --- error panel ------------------------------------------------------
    err_ymax = max(float(np.nanmax([data[m]["errs"] for m in METHOD_ORDER])),
                   0.5) * 1.1
    ax_err.set_xlim(0, t_full[-1]); ax_err.set_ylim(0, err_ymax)
    ax_err.set_xlabel("time (s)", color=COL_MUTED)
    ax_err.set_ylabel("tracking error (m)", color=COL_MUTED)
    ax_err.set_title("Tracking error", color=COL_MUTED)
    ax_err.grid(alpha=0.4)
    err_lines = {m: ax_err.plot([], [], color=METHOD_COLORS[m], lw=LW_MAP[m],
                                label=m)[0] for m in METHOD_ORDER}

    # --- ACE diagnostics: predictive uncertainty + shift detector --------
    eps_arr   = data["ACE"]["epist"]
    ale_arr   = data["ACE"]["aleat"]
    sig_arr   = data["ACE"]["shift_sig"]
    upd_arr   = data["ACE"]["updates"]
    # Cap shift_signal display for readability (max can be 100s; threshold is 2).
    SHIFT_THR = 2.0
    SHIFT_CAP = float(np.clip(np.nanmax(sig_arr) * 1.05, 5.0, 50.0))
    # Uncertainty axes — auto-scaled to per-scenario maxima.
    eps_max = max(float(np.nanmax(eps_arr)), 1e-4)
    ale_max = max(float(np.nanmax(ale_arr)), 1e-4)
    upd_max = max(float(np.nanmax(upd_arr)), 1.0)

    # Top subplot — raw ε and a on a shared right-axis (independent scales).
    ax_unc.set_xlim(0, t_full[-1])
    ax_unc.set_ylim(0, eps_max * 1.10)
    ax_unc.set_ylabel("ε (epistemic)", color=COL_ACE)
    ax_unc.tick_params(axis="y", labelcolor=COL_ACE)
    ax_unc.set_title("Predictive uncertainty", color=COL_MUTED)
    ax_unc.grid(alpha=0.4)
    ax_unc.tick_params(axis="x", labelbottom=False)
    eps_line = ax_unc.plot([], [], color=COL_ACE, lw=2.0, label="ε")[0]
    # Twin axis for aleatoric a (different scale).
    ax_unc_r = ax_unc.twinx()
    ax_unc_r.set_ylim(0, ale_max * 1.10)
    ax_unc_r.set_ylabel("a (aleatoric)", color="#b45309")
    ax_unc_r.tick_params(axis="y", labelcolor="#b45309")
    ale_line = ax_unc_r.plot([], [], color="#b45309", lw=2.0, label="a")[0]

    # Bottom subplot — shift_signal (the actual decision rule) +
    # cumulative gradient updates on a right axis.
    ax_eng.set_xlim(0, t_full[-1])
    ax_eng.set_ylim(0, SHIFT_CAP)
    ax_eng.set_xlabel("time (s)", color=COL_MUTED)
    ax_eng.set_ylabel("shift signal", color=COL_ACE)
    ax_eng.tick_params(axis="y", labelcolor=COL_ACE)
    ax_eng.set_title("Shift detector + learning steps", color=COL_MUTED)
    ax_eng.grid(alpha=0.4)
    ax_eng.axhline(SHIFT_THR, ls="--", lw=1.5, color=COL_ACE, alpha=0.6,
                   zorder=1)
    ax_eng.text(t_full[-1] * 0.99, SHIFT_THR + SHIFT_CAP * 0.02,
                f"threshold = {SHIFT_THR:.1f}",
                ha="right", va="bottom", fontsize=10, color=COL_ACE,
                style="italic")
    sig_line = ax_eng.plot([], [], color=COL_ACE, lw=2.0)[0]
    ax_eng_r = ax_eng.twinx()
    ax_eng_r.set_ylim(0, upd_max * 1.10)
    ax_eng_r.set_ylabel("updates", color="#6b7280")
    ax_eng_r.tick_params(axis="y", labelcolor="#6b7280")
    upd_line = ax_eng_r.plot([], [], color="#6b7280", lw=2.0, ls=":")[0]

    TAIL_STEPS = int(round(1.0 / dt))   # 1.0 s of motion at simulation dt
    def update(i: int):
        k = int(frame_idx[i])
        # trails — last TAIL_STEPS samples only, full opacity hard cut
        for m in METHOD_ORDER:
            lo = max(0, k - TAIL_STEPS + 1)
            xy_seg = data[m]["xy"][lo:k + 1]
            if xy_seg.shape[0] > 0:
                trails[m].set_data(xy_seg[:, 0], xy_seg[:, 1])
                dots[m].set_data([xy_seg[-1, 0]], [xy_seg[-1, 1]])
        # error
        for m in METHOD_ORDER:
            err_lines[m].set_data(t_full[:k + 1], data[m]["errs"][:k + 1])
        # uncertainty panel — raw ε (left axis) + a (right axis)
        eps_line.set_data(t_full[:k + 1], eps_arr[:k + 1])
        ale_line.set_data(t_full[:k + 1], ale_arr[:k + 1])
        # shift-detector + updates panel
        sig_clipped = np.clip(sig_arr[:k + 1], 0.0, SHIFT_CAP)
        sig_line.set_data(t_full[:k + 1], sig_clipped)
        upd_line.set_data(t_full[:k + 1], upd_arr[:k + 1])
        return list(trails.values()) + list(dots.values()) + \
               list(err_lines.values()) + [eps_line, ale_line, sig_line, upd_line]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / render_fps,
                         blit=False)
    return fig, anim


import shutil


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ffmpeg is required to render MP4 clips but was not found on PATH.\n"
            "Install it with `apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)."
        )


def render_one(scenario_name: str, model: EvidentialNet, out_dir: Path,
               total_time: float, dt: float, render_fps: int,
               bitrate_kbps: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario_name}.mp4"
    t0 = time.perf_counter()
    data = run_scenario(scenario_name, model, total_time=total_time, dt=dt)
    fig, anim = build_animation(data, render_fps=render_fps)
    writer = FFMpegWriter(fps=render_fps, codec="libx264",
                          bitrate=bitrate_kbps,
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    skip = int(2.0 / dt)
    ace_mean = float(data["ACE"]["errs"][skip:].mean())
    frozen_mean = float(data["Frozen"]["errs"][skip:].mean())
    lqr_mean = float(data["LQR"]["errs"][skip:].mean())
    print(f"  [{scenario_name}] rendered in {time.perf_counter() - t0:.1f}s "
          f"-> {out_path.name}  "
          f"(LQR={lqr_mean:.3f} Frozen={frozen_mean:.3f} ACE={ace_mean:.3f} m)")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=list(SCENARIOS) + ["all"],
                   default="all")
    p.add_argument("--out-dir",
                   default=str(ROOT.parent / "assets" / "animations"))
    p.add_argument("--total-time", type=float, default=30.0)
    p.add_argument("--dt",         type=float, default=0.02)
    p.add_argument("--fps",        type=int,   default=60)
    p.add_argument("--bitrate",    type=int,   default=1000,
                   help="MP4 target bitrate in kbps.")
    args = p.parse_args(argv)
    _check_ffmpeg()
    out_dir = Path(args.out_dir)
    model = load_unicycle_model()
    targets = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    print(f"Rendering {len(targets)} clip(s) to {out_dir}")
    for name in targets:
        render_one(name, model, out_dir,
                   total_time=args.total_time, dt=args.dt,
                   render_fps=args.fps, bitrate_kbps=args.bitrate)


if __name__ == "__main__":
    main()
