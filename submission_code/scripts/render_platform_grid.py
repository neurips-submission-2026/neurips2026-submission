"""2x3 grid: 2 conditions × 3 platforms in one animated MP4.

Rows (conditions):
    1. Parametric shift   — mass × mass_mult (per-platform from run_benchmark)
    2. Sensor noise       — platform-specific σ (from run_benchmark)
Columns (platforms):
    1. Unicycle  (ground)
    2. AUV       (underwater, 2-D x-y projection)
    3. Drone     (aerial,     2-D x-y projection)

Each cell shows the lemniscate (or 2-D projection of it) with three
method tails — Frozen (dotted square), ER (dashed diamond), ACE (solid
circle) — plus a gold goal star and a small text annotation with the
final-15-s tracking error for each method.

Output:
    website/assets/animations/platform_grid.mp4
"""
from __future__ import annotations
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.trajectories import generate_trajectory, lemniscate_2d, lemniscate_3d
from scripts.run_benchmark import (
    PLATFORM_CFG, _SENSOR_NOISE, _load_model, _make_perturbed_env,
    _make_trajectory, _make_controller, _make_nominal_env,
)
from scripts._demo_palette import (
    apply_style, METHOD_COLORS, COL_BG, COL_PANEL, COL_TEXT, COL_MUTED,
    COL_ACE, COL_ER, COL_FROZEN,
)
from training.train_online import UniformCLAdapter, ACEAdapter
from controllers.nn_controller import NNController


# ---------------------------------------------------------------------------
# Grid configuration
# ---------------------------------------------------------------------------

PLATFORMS = ["unicycle", "auv", "drone"]
PLATFORM_PRETTY = {"unicycle": "Unicycle", "auv": "AUV", "drone": "Drone"}

CONDITIONS = ["Disturbance", "SensorNoise"]
CONDITION_PRETTY = {
    "Disturbance": "Disturbance (ε says LEARN)",
    "SensorNoise": "Sensor noise (a says HOLD)",
}

# Enhanced sensor noise for the demo — the benchmark's σ on unicycle
# (0.02 m) sits below ER's update gate so the contrast is invisible.
# These values are 5-10× the benchmark's; per the Appendix-C sensor-σ
# sweep ER's drift compounds at this scale.
_DEMO_SENSOR_NOISE = {
    "unicycle": dict(std_pos=0.10, std_vel=0.03),
    "auv":      dict(std_pos=0.10, std_vel=0.05),
    "drone":    dict(std_pos=0.03, std_vel=0.03, std_ori=0.02),
}

METHOD_ORDER = ["LQR", "Frozen", "ER", "ACE"]

# Number of lemniscate laps to animate per platform.  Each platform
# has its own lemniscate period; we pick a fixed lap count so all
# three cells in a row complete their laps in roughly comparable wall
# time, scaled by the per-platform total_time below.
N_CYCLES_PER_PLATFORM = {"unicycle": 4, "auv": 3, "drone": 4}

# Per-platform lemniscate period (seconds).  Re-derived from
# scripts.run_benchmark trajectory generation so what we draw matches
# what the benchmark scores.
_LEM_PERIOD = {"unicycle": 2 * np.pi / 0.5,
               "auv":      2 * np.pi / 0.08,
               "drone":    2 * np.pi / 0.12}

# View limits (metres) for each platform's 2-D projection.  Chosen so
# the lemniscate fills the cell.
_VIEW = {
    "unicycle": dict(xlim=(-4.6, 4.6), ylim=(-2.55, 2.55)),
    "auv":      dict(xlim=(-1.45, 1.45), ylim=(-0.80, 0.80)),
    "drone":    dict(xlim=(-0.60, 0.60), ylim=(-0.35, 0.35)),
}

# Lookahead in steps for the goal-star position so the star aligns
# with the well-tracking methods on each platform.
_GOAL_LOOKAHEAD = {
    ("unicycle", "Disturbance"):   3,
    ("unicycle", "SensorNoise"):  12,
    ("auv",      "Disturbance"):   3,
    ("auv",      "SensorNoise"):   6,
    ("drone",    "Disturbance"):   3,
    ("drone",    "SensorNoise"):   4,
}

# Animation params.
RENDER_FPS    = 15
DPI           = 110
PLAYBACK_SEC  = 30.0    # target playback duration per clip


# ---------------------------------------------------------------------------
# State → (x, y) projection
# ---------------------------------------------------------------------------


def _state_xy(state: np.ndarray, platform: str) -> np.ndarray:
    """Pull the (x, y) projection out of a platform state vector."""
    return state[:2]


def _pos_xy(pos: np.ndarray, platform: str) -> np.ndarray:
    """Pull (x, y) out of a reference position vector (2-D for unicycle,
    3-D for AUV/drone)."""
    return pos[:, :2]


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _make_demo_env(platform: str, scenario: str, seed: int):
    """Variant of run_benchmark._make_perturbed_env that uses the
    inflated demo sensor noise so unicycle and AUV show visible ER
    drift on a single-clip horizon.  Disturbance / nominal pass
    through unchanged."""
    if scenario != "SensorNoise":
        return _make_perturbed_env(platform, scenario, seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = _make_nominal_env(platform)
    env.set_noise(**_DEMO_SENSOR_NOISE[platform])
    return env


def _make_demo_controller(method: str, platform: str, model, env, dt: float):
    """Use run_benchmark's standard controller config (matches the paper)."""
    return _make_controller(method, platform, model, env, dt)


def run_method(platform: str, scenario: str, method: str, model,
               total_time: float, seed: int = 42) -> dict:
    """Run one method on one (platform, scenario) and return the per-step
    trajectory + tracking error."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = _make_demo_env(platform, scenario, seed)
    positions, n_steps, init, dt = _make_trajectory(platform, total_time)
    env.reset(seed=seed)
    env.reset(state=init)
    ctrl = _make_demo_controller(method, platform, model, env, dt)

    xy   = np.zeros((n_steps, 2))
    errs = np.zeros(n_steps)
    use_obs = (scenario == "SensorNoise") and hasattr(env, "get_observation")

    for k in range(n_steps):
        state = env.get_observation() if use_obs else env.get_state()
        if method == "LQR":
            if platform == "unicycle":
                action, _ = ctrl.compute(state, positions, k)
            else:
                # LQR3D has a different API: aim for the next reference
                # waypoint via the `setpoint` kwarg (matches
                # run_benchmark._step_lqr).
                sp = positions[min(k + 3, n_steps - 1)]
                action = ctrl.compute(state, setpoint=sp)
            env.step(action)
        elif method == "Frozen":
            action, _ = ctrl.compute(env, state, positions, k, dt)
            env.step(action)
        else:
            action, _ = ctrl.step(env, state, positions, k)
        truth = env.get_state()
        xy[k]   = _state_xy(truth, platform)
        ref_xy  = positions[k][:2]
        errs[k] = float(np.linalg.norm(xy[k] - ref_xy))
        # numerical-blowup guard
        if not np.isfinite(errs[k]) or errs[k] > 50.0:
            errs[k:] = errs[k]
            xy[k:]   = xy[k]
            break
    return dict(xy=xy, errs=errs, positions=_pos_xy(positions[:n_steps], platform),
                dt=dt)


def run_cell(platform: str, scenario: str, models: dict,
             seed: int = 42) -> dict:
    """Run all three methods for one (platform, scenario) cell."""
    period = _LEM_PERIOD[platform]
    n_cyc  = N_CYCLES_PER_PLATFORM[platform]
    total  = period * n_cyc
    out: dict = {"platform": platform, "scenario": scenario,
                 "total_time": total, "period": period,
                 "n_cycles": n_cyc, "dt": PLATFORM_CFG[platform]["dt"]}
    for m in METHOD_ORDER:
        out[m] = run_method(platform, scenario, m, models[platform],
                            total_time=total, seed=seed)
    return out


# ---------------------------------------------------------------------------
# Per-cell drawing
# ---------------------------------------------------------------------------

LW = {"LQR": 2.2, "Frozen": 2.4, "ER": 2.6, "ACE": 3.4}
LS = {"LQR": "-.", "Frozen": ":", "ER": "--", "ACE": "-"}
MS = {"LQR": 9,    "Frozen": 9,  "ER": 11, "ACE": 14}
MARK = {"LQR": "^", "Frozen": "s", "ER": "D", "ACE": "o"}
TAIL_ALPHA = {"LQR": 0.45, "Frozen": 0.45, "ER": 0.50, "ACE": 0.70}
DOT_ALPHA  = {"LQR": 0.75, "Frozen": 0.75, "ER": 0.80, "ACE": 0.92}
Z_TRAIL    = {"LQR": 2.8, "Frozen": 3.0, "ER": 3.2, "ACE": 3.4}
Z_DOT      = {"LQR": 3.8, "Frozen": 4.0, "ER": 4.2, "ACE": 4.4}


def setup_cell(ax, cell: dict, platform: str, scenario: str):
    """Draw the static reference path and create the per-method artists.

    Returns (trails dict, dots dict, goal_dot, frame_idx, period_steps,
             pos_xy_arr, summary_text artist).
    """
    pos_xy = cell["ACE"]["positions"]
    view = _VIEW[platform]
    ax.set_aspect("equal")
    ax.set_xlim(*view["xlim"])
    ax.set_ylim(*view["ylim"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfd")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
    title = f"{PLATFORM_PRETTY[platform]}  —  {CONDITION_PRETTY[scenario]}"
    ax.set_title(title, color=COL_TEXT, fontsize=12, weight="semibold",
                 loc="left")
    # Thin dashed reference path.
    ax.plot(pos_xy[:, 0], pos_xy[:, 1], "--", color="#9ca3af",
            lw=1.0, alpha=0.6, zorder=1)

    trails, dots = {}, {}
    for m in METHOD_ORDER:
        c = METHOD_COLORS[m]
        trails[m] = ax.plot([], [], color=c, lw=LW[m], ls=LS[m],
                            alpha=TAIL_ALPHA[m], zorder=Z_TRAIL[m])[0]
        dots[m]   = ax.plot([], [], MARK[m], color=c, ms=MS[m],
                            mec="white", mew=1.2, alpha=DOT_ALPHA[m],
                            zorder=Z_DOT[m])[0]
    goal_dot, = ax.plot([], [], "*", color="#f59e0b", ms=18,
                        mec="#7c2d12", mew=1.4, zorder=5)

    # Final tracking-error summary, displayed in the corner.
    dt = cell["dt"]
    last_w = int(min(15.0, cell["total_time"] * 0.35) / dt)
    means = {m: float(cell[m]["errs"][-last_w:].mean()) for m in METHOD_ORDER}
    ace = means["ACE"]
    lines = []
    for m in METHOD_ORDER:
        if m == "ACE":
            lines.append(f"{m:<6} {means[m]:.3f} m")
        else:
            ratio = means[m] / ace if ace > 0 else float("inf")
            lines.append(f"{m:<6} {means[m]:.3f} m  ({ratio:.2f}× ACE)")
    summary = ax.text(
        0.02, 0.02, "\n".join(lines),
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.5, color=COL_TEXT, family="monospace",
        bbox=dict(boxstyle="round,pad=0.30",
                  facecolor="white", edgecolor="#d1d5db", alpha=0.92,
                  linewidth=0.8),
    )
    return trails, dots, goal_dot, pos_xy, summary


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def build_animation(cells: list[dict]) -> tuple:
    """cells is a length-6 list ordered row-major:
        [(plat, scen) for scen in CONDITIONS for plat in PLATFORMS].
    """
    apply_style()
    # figsize chosen so both dimensions are even at DPI=110 (libx264 needs even pixels).
    fig = plt.figure(figsize=(15, 9), dpi=DPI)
    gs = fig.add_gridspec(2, 3, wspace=0.10, hspace=0.30)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.045)

    fig.suptitle("See ACE adapt",
                 y=0.965, fontsize=20, weight="bold", color=COL_TEXT)
    fig.text(0.5, 0.926,
             "Top row: disturbance  (ε says LEARN).   "
             "Bottom row: sensor noise  (a says HOLD).   "
             "LQR (green triangle, dash-dot) · Frozen (gray square, dotted) · "
             "ER (red diamond, dashed) · ACE (blue circle, solid) · gold star = goal.",
             ha="center", va="top", fontsize=10, color=COL_MUTED, style="italic")

    # Tail seconds per platform → matches hero-pair's TAIL_SECONDS but
    # scaled so it covers roughly the same fraction of the lap.
    tail_steps = {p: int(round(_LEM_PERIOD[p] * 0.20 / PLATFORM_CFG[p]["dt"]))
                  for p in PLATFORMS}

    # Per-cell artists.
    cell_artists = []
    longest_total = max(c["total_time"] for c in cells)
    for i, cell in enumerate(cells):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        trails, dots, goal_dot, pos_xy, summary = setup_cell(
            ax, cell, cell["platform"], cell["scenario"])
        n_steps = len(cell["ACE"]["xy"])
        dt = cell["dt"]
        # Each platform may have a different dt and total_time; we
        # parameterise animation by NORMALISED time fraction so all
        # cells advance proportionally.  Each cell maps the global
        # frame-time → its own local step index.
        cell_artists.append({
            "cell": cell, "ax": ax, "trails": trails, "dots": dots,
            "goal": goal_dot, "pos_xy": pos_xy, "summary": summary,
            "n_steps": n_steps, "dt": dt,
            "tail_steps": tail_steps[cell["platform"]],
            "lookahead": _GOAL_LOOKAHEAD[
                (cell["platform"], cell["scenario"])],
        })

    # We render PLAYBACK_SEC seconds at RENDER_FPS.  Each cell's local
    # step advances by its own n_steps / total_frames per frame.
    n_frames = int(PLAYBACK_SEC * RENDER_FPS)

    def update(frame: int):
        artists: list = []
        frac = (frame + 1) / n_frames
        for ca in cell_artists:
            k = min(int(round(frac * (ca["n_steps"] - 1))), ca["n_steps"] - 1)
            # Goal star
            goal_idx = min(k + ca["lookahead"], len(ca["pos_xy"]) - 1)
            ca["goal"].set_data([ca["pos_xy"][goal_idx, 0]],
                                [ca["pos_xy"][goal_idx, 1]])
            artists.append(ca["goal"])
            for m in METHOD_ORDER:
                lo = max(0, k - ca["tail_steps"] + 1)
                seg = ca["cell"][m]["xy"][lo:k + 1]
                if seg.shape[0] > 0:
                    ca["trails"][m].set_data(seg[:, 0], seg[:, 1])
                    ca["dots"][m].set_data([seg[-1, 0]], [seg[-1, 1]])
                    artists.append(ca["trails"][m])
                    artists.append(ca["dots"][m])
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0 / RENDER_FPS, blit=False)
    return fig, anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",
                   default=str(ROOT.parent / "assets" / "animations"))
    p.add_argument("--bitrate", type=int, default=1800)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--no-render", action="store_true",
                   help="Skip MP4 save; emit a PNG snapshot of the final frame.")
    args = p.parse_args(argv)
    if not args.no_render:
        _check_ffmpeg()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all three models once.
    print("Loading models ...")
    models = {p: _load_model(p) for p in PLATFORMS}

    # Run 6 cells (row-major: top row = MassShift, bottom = SensorNoise).
    cells: list[dict] = []
    t0 = time.perf_counter()
    for scenario in CONDITIONS:
        for platform in PLATFORMS:
            print(f"  rollout {platform:<8} / {scenario:<12} ...",
                  end=" ", flush=True)
            cell = run_cell(platform, scenario, models, seed=args.seed)
            cells.append(cell)
            last_w = int(min(15.0, cell["total_time"] * 0.35) / cell["dt"])
            mns = {m: float(cell[m]["errs"][-last_w:].mean())
                   for m in METHOD_ORDER}
            ace = mns["ACE"]
            parts = []
            for m in METHOD_ORDER:
                if m == "ACE":
                    parts.append(f"ACE {ace:.3f}")
                else:
                    parts.append(f"{m} {mns[m]:.3f} ({mns[m]/ace:.2f}× ACE)")
            print(f"final-15s mean — " + ", ".join(parts))
    print(f"  6 rollouts done in {time.perf_counter() - t0:.1f}s")

    fig, anim = build_animation(cells)
    if args.no_render:
        snap = out_dir / "platform_grid_snapshot.png"
        # Force last frame
        try:
            anim._func(int(PLAYBACK_SEC * RENDER_FPS) - 1)
        except Exception:
            pass
        fig.savefig(str(snap), dpi=DPI, facecolor="white")
        plt.close(fig)
        print(f"  snapshot -> {snap}")
        return

    out_path = out_dir / "platform_grid.mp4"
    writer = FFMpegWriter(fps=RENDER_FPS, codec="libx264",
                          bitrate=args.bitrate,
                          extra_args=["-pix_fmt", "yuv420p"])
    t1 = time.perf_counter()
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"  rendered in {time.perf_counter() - t1:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
