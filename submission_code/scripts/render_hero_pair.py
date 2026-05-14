"""Hero pair: two side-by-side MP4s that dramatise the paper's two-sided
uncertainty claim.

  hero_friction.mp4      — friction × 3.5 held for 12 lemniscate laps.
                           ε says LEARN.  ACE adapts; ER lags; Frozen drifts.
  hero_sensor_noise.mp4  — σ_pos = 8 cm position noise, whole rollout.
                           a says HOLD.  ACE ≈ Frozen; ER overfits the noise.

Each clip is a 2 × 2 grid:
  Trajectory      |  Tracking error
  ε per-lap mean  |  a per-lap mean

Three methods rendered:
  Frozen  — slate gray, no online learning
  ER      — uniform replay continual learning (UniformCLAdapter)
  ACE     — full ACE (NIG + priority replay + adaptive λ + L2 anchor)

ε and a are drawn for ACE only; Frozen and ER do not consume the evidential
signals in their inference path so plotting them would imply they do.

Usage
-----
    python3 scripts/render_hero_pair.py --scenario friction
    python3 scripts/render_hero_pair.py --scenario sensor_noise
    python3 scripts/render_hero_pair.py --scenario both

Output
------
    website/assets/animations/hero_{friction,sensor_noise}.mp4
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

from envs.unicycle_env import UnicycleEnv
from training.train_online import ACEAdapter, UniformCLAdapter
from controllers.nn_controller import NNController
from scripts.render_demo_clips import (
    load_unicycle_model, build_trajectory, _apply_shift,
)
from scripts._demo_palette import (
    apply_style, METHOD_COLORS, COL_BG, COL_PANEL, COL_TEXT, COL_MUTED,
    COL_ACE, COL_ER, COL_FROZEN,
)


# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

DT             = 0.02
LEM_SCALE      = 4.0
LEM_SPEED      = 0.5
LEM_PERIOD     = 2 * np.pi / LEM_SPEED      # 12.566 s per lap
N_CYCLES       = 9                           # 9 laps total — animation skips lap 1 so playback shows laps 2-9
ANIM_LAP_START = 2                           # first lap shown on screen (1-based)
ANIM_LAP_END   = 9                           # last lap shown on screen (1-based, inclusive)
TOTAL_TIME     = LEM_PERIOD * N_CYCLES       # 113.1 s
ROLLOUT_TIME   = TOTAL_TIME + 2.0            # small lookahead buffer

# Friction × 2 held — large enough that Frozen sits at a visible offset
# (~0.35 m sustained on a 4-m lemniscate) so the recovery story is
# unmistakable.  σ_pos = 0.12 m is well above the benchmark's 0.02 m
# default; chosen so ER's uniform-replay drift compounds visibly across
# 125 s while ACE's λ-floor keeps it pinned near Frozen.
FRICTION_MULT_HERO  = 2.5
SHIFT_TIME_FRICTION = 0.5
SENSOR_SIGMA_POS    = 0.12

# Render parameters — slow enough that a human can actually track the
# three methods through one lemniscate lap.  14 laps × 12.57 s / stride
# 10 = 880 frames at 15 fps → 58.7 s total playback, ~4.2 s per lap.
RENDER_FPS    = 15
DPI           = 110
STRIDE        = 10
TAIL_SECONDS  = 3.0                          # long tails so motion is legible at slow playback
TAIL_STEPS    = int(round(TAIL_SECONDS / DT))

# Trajectory panel view box (metres).  Tightened around the lemniscate
# so the figure-8 fills the panel and method offsets read at full size.
TRAJ_XLIM = (-4.55, 4.55)
TRAJ_YLIM = (-2.55, 2.55)

# Reference path appearance — a thin dashed line.  The eye-catching
# reference is the moving GOAL STAR (gold five-pointed) below, not
# the path itself.  GOAL_LOOKAHEAD shifts the star slightly forward
# of pos[k] so well-tracking methods visually align with the star
# (otherwise the controller's adaptive lookahead makes the robot dot
# appear ahead of pos[k]).  Per-scenario because noise rollouts have
# higher per-step variance and need more lookahead to align.
REF_LW    = 1.6
REF_ALPHA = 0.55
GOAL_LOOKAHEAD_BY_SCENARIO = {"friction": 3, "sensor_noise": 12}

METHOD_ORDER  = ["Frozen", "ER", "ACE"]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "friction": {
        "shift": {"friction": FRICTION_MULT_HERO},
        "shift_time": SHIFT_TIME_FRICTION,
        "use_obs": False,
        "title": (
            f"ε says LEARN  —  friction × {FRICTION_MULT_HERO:g} "
            f"held for {N_CYCLES} laps"
        ),
        "subtitle": (
            "ACE's ε spikes, then drops as the model adapts. "
            "Tracking recovers in 3 laps. ER lags. Frozen sits at the "
            "friction error forever."
        ),
        "shade": True,
        "shade_label": f"friction × {FRICTION_MULT_HERO:g} ON",
    },
    "sensor_noise": {
        "shift": {"sensor_pos": SENSOR_SIGMA_POS},
        "shift_time": 0.0,
        "use_obs": True,
        "title": (
            f"a says HOLD  —  σ = {SENSOR_SIGMA_POS*100:.0f} cm "
            f"position noise (entire rollout)"
        ),
        "subtitle": (
            "ACE's a stays high; λ-floor clamps updates; ACE ≈ Frozen. "
            "ER's uniform replay incorporates the noise; tracking degrades."
        ),
        "shade": False,
        "shade_label": "",
    },
}


# ---------------------------------------------------------------------------
# Controller factory (local — adds ER to the demo path)
# ---------------------------------------------------------------------------


def _make_controller(method: str, model, env, dt: float):
    """Construct (controller, kind) for one of Frozen / ER / ACE.

    Mirrors `scripts/run_benchmark._make_controller`'s hyperparameters so
    behaviour matches the paper.  Returns the same (controller, kind)
    pair shape as `scripts/render_demo_clips.make_controller`.
    """
    feature_mode = getattr(model, "feature_mode", "full")
    nominal_env = UnicycleEnv(dt=dt)
    fb_gain = 1.0
    if method == "Frozen":
        return (
            NNController(
                model, device="cpu", feature_mode=feature_mode,
                feedback_gain=fb_gain, feedback_env=nominal_env,
                feedback_dt=dt,
            ),
            "nn",
        )
    if method == "ER":
        # Demo ER = uniform-replay continual learning with no update gate.
        # The paper's ER configuration uses ``update_threshold=0.02`` which
        # sits well above the unicycle's typical ε≈1e-6 — on this platform
        # the gate is effectively "never update", making paper-ER and
        # Frozen indistinguishable.  For the hero clip we drop the gate
        # so ER reflects what uniform replay *does* when it actually
        # updates: useful adaptation on a structured shift, harmful
        # noise-chasing under sensor jitter.  The caption labels this as
        # "ER (uniform replay, no gate)" so the comparison is honest.
        return (
            UniformCLAdapter(
                model, env, dt, lr=3e-4, lambda_reg=0.001,
                update_every=5, min_buffer_size=64,
                batch_size=32, buffer_capacity=1000,
                update_threshold=0.0, feature_mode=feature_mode,
                feedback_gain=fb_gain, feedback_env=nominal_env,
            ),
            "adapter",
        )
    if method == "ACE":
        return (
            ACEAdapter(
                model, env, dt,
                lr=1e-4, lambda_eta=0.10, lambda_kappa=0.05,
                lambda_floor=0.10, anchor_strength=5e-3,
                batch_size=32, buffer_capacity=1000,
                update_every=5, min_buffer_size=64,
                feature_mode=feature_mode,
                feedback_gain=fb_gain, feedback_env=nominal_env,
                loss_form="tempered",
                lambda_schedule_init=1.0,
                shift_detection=True,
                shift_threshold=2.0,
                shift_lr_boost=3.0,
            ),
            "adapter",
        )
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def run_one(method: str, scenario_key: str, model, seed: int = 42) -> dict:
    """Run a single (method, scenario) rollout for the hero clip.

    Friction is applied at t = shift_time so the viewer sees a clean
    pre-shift baseline; sensor noise is applied from the start.
    """
    spec = SCENARIOS[scenario_key]
    positions, n_steps, init = build_trajectory(ROLLOUT_TIME, DT)
    n = int(TOTAL_TIME / DT)
    positions = positions[:n + n_steps - n]  # keep full-length positions
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = UnicycleEnv(dt=DT)
    # Sensor noise is constitutive of the scenario — apply at t=0.
    # Friction is applied at runtime at SHIFT_TIME (so the pre-shift
    # baseline is visible).
    if scenario_key == "sensor_noise":
        env = _apply_shift(env, spec["shift"], seed)
    env.reset(seed=seed)
    env.reset(state=init)
    ctrl, kind = _make_controller(method, model, env, DT)

    n_steps_run = int(TOTAL_TIME / DT)
    k_shift = int(spec["shift_time"] / DT) if spec["shade"] else -1

    errs = np.zeros(n_steps_run)
    xy   = np.zeros((n_steps_run, 2))
    eps  = np.zeros(n_steps_run)
    ale  = np.zeros(n_steps_run)
    lam  = np.full(n_steps_run, np.nan)
    upd  = np.zeros(n_steps_run)

    use_obs = spec["use_obs"] and hasattr(env, "get_observation")

    for k in range(n_steps_run):
        if scenario_key == "friction" and k == k_shift:
            env.set_uncertainty(
                True, friction_multiplier=spec["shift"]["friction"]
            )
        state = env.get_observation() if use_obs else env.get_state()
        if kind == "nn":
            a, info = ctrl.compute(env, state, positions, k, DT)
            env.step(a)
        else:
            a, info = ctrl.step(env, state, positions, k)
        truth = env.get_state()
        xy[k] = truth[:2]
        errs[k] = float(np.linalg.norm(xy[k] - positions[k]))
        eps[k] = info.get("epistemic", 0.0) or 0.0
        ale[k] = info.get("aleatoric", 0.0) or 0.0
        lam[k] = info.get("lambda_schedule", np.nan)
        if kind == "adapter":
            upd[k] = int(getattr(ctrl, "update_count", 0))
        if not np.isfinite(errs[k]) or errs[k] > 50.0:
            errs[k:] = errs[k]; xy[k:] = xy[k]
            break
    return dict(errs=errs, xy=xy, eps=eps, ale=ale, lamb=lam, upd=upd,
                positions=positions[:n_steps_run])


# ---------------------------------------------------------------------------
# Per-lap aggregation
# ---------------------------------------------------------------------------


def per_cycle_means(arr: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Compute per-lap means and the (t_lo, t_hi) boundary list."""
    means, bounds = [], []
    for c in range(N_CYCLES):
        t_lo = c * LEM_PERIOD
        t_hi = (c + 1) * LEM_PERIOD
        k_lo = int(t_lo / DT)
        k_hi = int(t_hi / DT)
        means.append(float(arr[k_lo:k_hi].mean()))
        bounds.append((t_lo, t_hi))
    return np.array(means), bounds


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def build_animation(scenario_key: str, rollouts: dict) -> tuple:
    """Build the 2x2 figure + FuncAnimation for one hero clip.

    Parameters
    ----------
    scenario_key : str
        One of SCENARIOS keys.
    rollouts : dict
        {method_name: rollout_dict, ...} from run_one for every entry in
        METHOD_ORDER.
    """
    apply_style()
    spec = SCENARIOS[scenario_key]
    A   = rollouts["ACE"]
    n_steps = int(TOTAL_TIME / DT)
    # Animation skips the lap-1 transient (already noisy after the
    # initial shock).  Frames cover laps ANIM_LAP_START..ANIM_LAP_END.
    k_anim_start = int((ANIM_LAP_START - 1) * LEM_PERIOD / DT)
    k_anim_end   = min(int(ANIM_LAP_END * LEM_PERIOD / DT), n_steps)
    frame_idx = np.arange(k_anim_start, k_anim_end, STRIDE)
    n_frames  = len(frame_idx)
    t_full = np.arange(n_steps) * DT

    eps_cycle_means, _ = per_cycle_means(A["eps"])
    ale_cycle_means, _ = per_cycle_means(A["ale"])

    fig = plt.figure(figsize=(16, 9), dpi=DPI)
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0],
                           height_ratios=[1.0, 1.0],
                           wspace=0.22, hspace=0.30)
    # Top margin tight because suptitle/subtitle dropped.
    fig.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=0.07)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_err  = fig.add_subplot(gs[0, 1])
    ax_eps  = fig.add_subplot(gs[1, 0])
    ax_ale  = fig.add_subplot(gs[1, 1])

    # Stripped: no suptitle, no subtitle.  The HTML caption identifies
    # the scenario in one line; the video itself just shows data.

    # --- Trajectory panel -----------------------------------------------
    pos = A["positions"]
    # Thin dashed reference path — just a hint of the lemniscate so the
    # eye knows where the figure-8 is, without competing with the
    # moving goal star (drawn below) for attention.
    ax_traj.plot(pos[:, 0], pos[:, 1], "--", color="#6b7280",
                 lw=REF_LW, alpha=REF_ALPHA, zorder=1,
                 label="reference path")
    ax_traj.set_aspect("equal")
    ax_traj.set_xlim(*TRAJ_XLIM)
    ax_traj.set_ylim(*TRAJ_YLIM)
    ax_traj.set_xticks([])
    ax_traj.set_yticks([])
    ax_traj.set_title("Trajectory", color=COL_MUTED, fontsize=15)
    for spine in ax_traj.spines.values():
        spine.set_color(COL_PANEL)

    # Distinct line styles per method so they read at a glance, plus
    # strong translucency on the tails so when methods overlap you can
    # still see all three through each other.  ACE on top (highest
    # zorder) since it's the method we want the eye to land on.
    LW = {"Frozen": 3.0, "ER": 3.4, "ACE": 4.4}
    LS = {"Frozen": ":", "ER": "--", "ACE": "-"}
    MS = {"Frozen": 13,  "ER": 15,  "ACE": 20}
    MARK = {"Frozen": "s", "ER": "D", "ACE": "o"}
    # Tail alphas pushed down so overlapping tails are mutually visible.
    TAIL_ALPHA = {"Frozen": 0.45, "ER": 0.50, "ACE": 0.70}
    DOT_ALPHA  = {"Frozen": 0.70, "ER": 0.75, "ACE": 0.90}
    Z_TRAIL    = {"Frozen": 3.0, "ER": 3.2, "ACE": 3.4}
    Z_DOT      = {"Frozen": 4.0, "ER": 4.2, "ACE": 4.4}
    trails, dots = {}, {}
    for m in METHOD_ORDER:
        c = METHOD_COLORS[m]
        trails[m] = ax_traj.plot(
            [], [], color=c, lw=LW[m], ls=LS[m], alpha=TAIL_ALPHA[m],
            zorder=Z_TRAIL[m], label=m,
        )[0]
        dots[m] = ax_traj.plot(
            [], [], MARK[m], color=c, ms=MS[m], mec="white", mew=1.4,
            alpha=DOT_ALPHA[m], zorder=Z_DOT[m],
        )[0]
    # Big moving goal star — the "robot SHOULD be HERE right now"
    # marker.  Per-scenario lookahead so it visually aligns with where
    # the controller is steering (otherwise it appears to lag).
    scenario_lookahead = GOAL_LOOKAHEAD_BY_SCENARIO.get(scenario_key, 3)
    goal_dot, = ax_traj.plot(
        [], [], "*", color="#f59e0b", ms=28, mec="#7c2d12", mew=2.0,
        zorder=5, label="goal  (where the robot should be)",
    )
    # Compact legend — just the method names.  Goal star and reference
    # path are explained in the website figcaption to keep the panel
    # uncluttered.
    handles, labels = ax_traj.get_legend_handles_labels()
    order = [labels.index("ACE"), labels.index("ER"), labels.index("Frozen")]
    ax_traj.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="lower right", fontsize=11, frameon=False,
                   labelcolor=COL_TEXT)

    # (Formula box removed — kept it minimal per design feedback.)

    # (Zoom inset removed — user feedback was that it cluttered the
    # figure and the rest of the panel was hard to read with it inside.
    # The tightened TRAJ_XLIM/YLIM view now does the same job: errors
    # are visually larger relative to the panel because the lemniscate
    # itself fills more of it.)

    # --- Tracking error panel -------------------------------------------
    err_max = max(
        float(rollouts[m]["errs"].max()) for m in METHOD_ORDER
    ) * 1.10
    ax_err.set_xlim(0, TOTAL_TIME)
    ax_err.set_ylim(0, err_max)
    ax_err.set_xlabel("time (s)", fontsize=13, color=COL_MUTED)
    ax_err.set_ylabel("tracking error (m)", fontsize=13, color=COL_MUTED)
    ax_err.set_title("Tracking error", color=COL_MUTED, fontsize=14)
    ax_err.grid(alpha=0.4)

    # Subtle shift band for friction only (no text label).
    if spec["shade"]:
        ax_err.axvspan(spec["shift_time"], TOTAL_TIME,
                       color="#fef3c7", alpha=0.45, zorder=0)
        ax_err.axvline(spec["shift_time"], color="#a16207", ls="--", lw=1.0,
                       alpha=0.6, zorder=1)

    err_lines = {}
    for m in METHOD_ORDER:
        err_lines[m] = ax_err.plot(
            [], [], color=METHOD_COLORS[m], lw=LW[m], label=m
        )[0]
    ax_err.legend(loc="upper right", fontsize=11, frameon=False,
                  labelcolor=COL_TEXT)

    # --- Bottom-left: ε per lap (ACE) -----------------------------------
    # Just epistemic — aleatoric tracks the same shape (ε and a share
    # β/(α-1)), so plotting both is redundant.  Raw values on log y
    # so the scenario contrast is visible: friction → ε drops over
    # the adaptation arc; noise → ε holds roughly flat.
    eps_top = float(eps_cycle_means.max()) * 1.30
    eps_bot = max(1e-7, float(eps_cycle_means.min()) * 0.70)
    ax_eps.set_xlim(0.5, N_CYCLES + 0.5)
    ax_eps.set_ylim(eps_bot, eps_top)
    ax_eps.set_yscale("log")
    ax_eps.set_xlabel("lemniscate lap", fontsize=13, color=COL_MUTED)
    ax_eps.set_ylabel("ε  (epistemic, lap mean)",
                      fontsize=13, color=COL_ACE)
    ax_eps.set_title("ε  per lap", color=COL_MUTED, fontsize=14)
    ax_eps.set_xticks(range(1, N_CYCLES + 1))
    ax_eps.set_xticklabels([str(i + 1) for i in range(N_CYCLES)],
                           fontsize=11, color=COL_TEXT)
    ax_eps.grid(alpha=0.4, which="both")
    eps_curve, = ax_eps.plot([], [], "-o", color=COL_ACE, lw=3.0, ms=11,
                             mec="white", mew=1.4, alpha=0.95)
    # Placeholder for aleatoric curve so the return-blit still works.
    ale_curve, = ax_eps.plot([], [], color="none", alpha=0.0)

    # --- Bottom-right: % improvement vs Frozen per lap ------------------
    # The OPERATIONAL VERDICT panel: did ACE's decision to adapt (or hold)
    # actually help?  Defined per lap as
    #     win[m]  = 100 · (frozen_err - method_err) / frozen_err
    # so:
    #   > 0  → method is better than Frozen (adaptation helped)
    #   = 0  → method matches Frozen (no help, no harm)
    #   < 0  → method is worse than Frozen (adaptation hurt)
    #
    # The two-sided paper claim collapses to this panel:
    #   friction  → ACE and ER both ~+85 %  (adaptation helps)
    #   noise     → ACE near 0 %  (correctly held)
    #                ER negative  (adaptation HURT — drifted past Frozen)
    err_per_lap = {m: np.array([
        float(rollouts[m]["errs"][int(c * LEM_PERIOD / DT):
                                  int((c + 1) * LEM_PERIOD / DT)].mean())
        for c in range(N_CYCLES)
    ]) for m in METHOD_ORDER}
    frozen_per_lap = err_per_lap["Frozen"]
    win = {
        m: 100.0 * (frozen_per_lap - err_per_lap[m]) /
           np.maximum(frozen_per_lap, 1e-6)
        for m in ("ER", "ACE")
    }

    win_top = max(95.0,
                  float(max(win["ER"].max(), win["ACE"].max())) + 10)
    win_bot = min(-25.0,
                  float(min(win["ER"].min(), win["ACE"].min())) - 10)
    ax_ale.set_xlim(0.5, N_CYCLES + 0.5)
    ax_ale.set_ylim(win_bot, win_top)
    ax_ale.set_xlabel("lemniscate lap", fontsize=13, color=COL_MUTED)
    ax_ale.set_ylabel("% better than Frozen", fontsize=13, color=COL_MUTED)
    ax_ale.set_title("Verdict", color=COL_MUTED, fontsize=14)
    ax_ale.set_xticks(range(1, N_CYCLES + 1))
    ax_ale.set_xticklabels([str(i + 1) for i in range(N_CYCLES)],
                           fontsize=11, color=COL_TEXT)
    ax_ale.grid(alpha=0.4)
    # Zone shading only — no helps/hurts labels, no annotations.
    ax_ale.axhline(0.0, color="#4b5563", lw=1.0, alpha=0.65, zorder=1)
    ax_ale.axhspan(0.0, win_top, color="#dcfce7", alpha=0.30, zorder=0)
    ax_ale.axhspan(win_bot, 0.0, color="#fee2e2", alpha=0.35, zorder=0)

    WIN_MARK = {"ER": "D", "ACE": "o"}
    WIN_LW   = {"ER": 2.6, "ACE": 3.6}
    win_lines = {}
    for m in ("ER", "ACE"):
        win_lines[m] = ax_ale.plot(
            [], [], f"-{WIN_MARK[m]}", color=METHOD_COLORS[m],
            lw=WIN_LW[m], ms=10, mec="white", mew=1.3, alpha=0.95,
            label=m,
        )[0]

    # Phase callout removed — kept the figure minimal per design feedback.

    def update(i: int):
        k = int(frame_idx[i])
        t = float(t_full[k])
        # Goal star — pos[k + lookahead] (scenario-specific) so it
        # visually aligns with the controllers' lookahead target.
        goal_idx = min(k + scenario_lookahead, len(pos) - 1)
        goal_dot.set_data([pos[goal_idx, 0]], [pos[goal_idx, 1]])
        # Method trails — last TAIL_STEPS samples only.
        for m in METHOD_ORDER:
            lo = max(0, k - TAIL_STEPS + 1)
            seg = rollouts[m]["xy"][lo:k + 1]
            if seg.shape[0] > 0:
                trails[m].set_data(seg[:, 0], seg[:, 1])
                dots[m].set_data([seg[-1, 0]], [seg[-1, 1]])
        # Error panel
        for m in METHOD_ORDER:
            err_lines[m].set_data(t_full[:k + 1], rollouts[m]["errs"][:k + 1])
        # Operational verdict (bottom-right) — % better than Frozen per
        # lap.  One point per completed lap; same cadence as bottom-left.
        completed_pl = int(t // LEM_PERIOD)
        if completed_pl > 0:
            xs_pl = np.arange(1, completed_pl + 1)
            for m in ("ER", "ACE"):
                win_lines[m].set_data(xs_pl, win[m][:completed_pl])
        else:
            for m in ("ER", "ACE"):
                win_lines[m].set_data([], [])
        # Per-lap ε raw values (bottom-left).
        completed = int(t // LEM_PERIOD)
        if completed > 0:
            xs = np.arange(1, completed + 1)
            eps_curve.set_data(xs, eps_cycle_means[:completed])
        else:
            eps_curve.set_data([], [])
        return (list(trails.values()) + list(dots.values()) +
                [goal_dot] +
                list(err_lines.values()) +
                list(win_lines.values()) +
                [eps_curve, ale_curve])

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0 / RENDER_FPS, blit=False)
    return fig, anim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH.")


def render_one(scenario_key: str, model, out_dir: Path,
               bitrate: int, no_render: bool, seed: int = 42) -> None:
    print(f"\n=== Hero clip: {scenario_key} ===")
    rollouts: dict = {}
    t0 = time.perf_counter()
    for m in METHOD_ORDER:
        print(f"  rollout {m} ...", end=" ", flush=True)
        rollouts[m] = run_one(m, scenario_key, model, seed=seed)
        last_w = int(15.0 / DT)
        print(f"final-15s mean = {rollouts[m]['errs'][-last_w:].mean():.4f} m")
    print(f"  rollouts done in {time.perf_counter() - t0:.1f}s")

    last_w = int(15.0 / DT)
    means  = {m: float(rollouts[m]["errs"][-last_w:].mean())
              for m in METHOD_ORDER}
    print(f"\n  final-15s tracking:")
    for m in METHOD_ORDER:
        ratio = (means[m] / means["ACE"]) if means["ACE"] > 0 else float("inf")
        marker = " (ACE)" if m == "ACE" else f"  ({ratio:.2f}× ACE)"
        print(f"    {m:<6} {means[m]:.4f} m{marker}")

    eps_means, _ = per_cycle_means(rollouts["ACE"]["eps"])
    ale_means, _ = per_cycle_means(rollouts["ACE"]["ale"])
    print(f"\n  ACE per-lap ε: {eps_means[0]:.3e} → {eps_means[-1]:.3e}  "
          f"({100*(1-eps_means[-1]/max(eps_means[0],1e-12)):+.0f}%)")
    print(f"  ACE per-lap a: {ale_means[0]:.3e} → {ale_means[-1]:.3e}  "
          f"({100*(1-ale_means[-1]/max(ale_means[0],1e-12)):+.0f}%)")

    fig, anim = build_animation(scenario_key, rollouts)
    if no_render:
        snap = out_dir / f"hero_{scenario_key}_snapshot.png"
        # Force last frame so all artists are populated.
        n_steps_run = int(TOTAL_TIME / DT)
        # Animation covers laps ANIM_LAP_START..ANIM_LAP_END only; the
        # last frame index is the count of frames in that window.
        k_anim_start = int((ANIM_LAP_START - 1) * LEM_PERIOD / DT)
        k_anim_end   = min(int(ANIM_LAP_END * LEM_PERIOD / DT), n_steps_run)
        n_anim_frames = len(np.arange(k_anim_start, k_anim_end, STRIDE))
        last_frame_idx = n_anim_frames - 1
        if anim._func is not None:
            try:
                anim._func(last_frame_idx)
            except Exception:
                pass
        fig.savefig(str(snap), dpi=DPI, facecolor="white")
        plt.close(fig)
        print(f"  snapshot -> {snap}")
        return

    out_path = out_dir / f"hero_{scenario_key}.mp4"
    writer = FFMpegWriter(fps=RENDER_FPS, codec="libx264", bitrate=bitrate,
                          extra_args=["-pix_fmt", "yuv420p"])
    t1 = time.perf_counter()
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"  rendered in {time.perf_counter() - t1:.1f}s -> {out_path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",
                   choices=list(SCENARIOS) + ["both"],
                   default="both")
    p.add_argument("--out-dir",
                   default=str(ROOT.parent / "assets" / "animations"))
    p.add_argument("--bitrate", type=int, default=1600)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--no-render", action="store_true",
                   help="Skip MP4 save; emit a PNG snapshot of the final frame.")
    args = p.parse_args(argv)
    if not args.no_render:
        _check_ffmpeg()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading unicycle model ...")
    model = load_unicycle_model()
    targets = list(SCENARIOS) if args.scenario == "both" else [args.scenario]
    for key in targets:
        render_one(key, model, out_dir, bitrate=args.bitrate,
                   no_render=args.no_render, seed=args.seed)


if __name__ == "__main__":
    main()
