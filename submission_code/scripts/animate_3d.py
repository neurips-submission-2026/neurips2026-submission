"""3-D animation for AUV / drone using matplotlib's Axes3D.

Renders a proper 3-D scene with depth, height, and a rotatable camera.
The drone is drawn as a 4-rotor quadcopter mesh and the AUV as a
cylinder hull, both transformed each frame from their body frame to
the world frame.  Four controllers run in parallel: LQR, NN Frozen,
NN CL, NN ACE.  The checkpoints loaded are the same paper models used
in ``run_multiplatform_benchmark.py``.

Usage:
    python scripts/animate_3d.py --platform drone
    python scripts/animate_3d.py --platform auv
    python scripts/animate_3d.py --platform drone --save out.mp4
    python scripts/animate_3d.py --platform auv --max-frames 400 --fps 30
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from config import Config
from scripts.animate import (
    CONTROLLERS,
    PLATFORM_CFG,
    PLATFORM_TITLES,
    SCENARIOS,
    ControllerState,
    load_model,
    make_trajectory,
    state_xyz,
    state_heading,
)


# ----------------------------------------------------------------------
# Scenario application (subset of animate.apply_perturbations — drops
# the slider plumbing since we drive perturbations from preset dicts).
# ----------------------------------------------------------------------

def apply_scenario_to_env(platform: str, env, params: dict) -> None:
    m = float(params.get("mass", 1.0))
    f = float(params.get("friction", 1.0))
    wx = float(params.get("wind_x", 0.0))
    wy = float(params.get("wind_y", 0.0))
    sn = float(params.get("state_noise", 0.0))
    an = float(params.get("noise", 0.0))
    if platform == "auv":
        env.set_perturbation(mass_mult=m, drag_mult=f)
        env.set_noise(std_pos=sn * 0.05, std_vel=sn * 0.05)
        env._act_noise_sigma = np.full(4, an * 2.0) if an > 0 else None
        return
    if platform == "drone":
        env.set_perturbation(mass_mult=m, drag_mult=f,
                             wind_x=wx, wind_y=wy, wind_z=0.0)
        env.set_noise(std_pos=sn * 0.02, std_vel=sn * 0.03,
                      std_ori=sn * 0.02)
        env._act_noise_sigma = (an * np.array([0.3, 0.10, 0.10, 0.05])
                                if an > 0 else None)


# ----------------------------------------------------------------------
# Body-frame meshes
# ----------------------------------------------------------------------

def _circle(radius: float, n: int = 20) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n + 1)
    return np.stack([radius * np.cos(t), radius * np.sin(t),
                     np.zeros_like(t)], axis=0)


def quadcopter_mesh(arm: float = 0.18) -> list[np.ndarray]:
    """Body-frame line segments: 4 rotor disks, X arms, body cap, nose."""
    rotor_r = arm * 0.40
    parts: list[np.ndarray] = []
    rotor_centres = [(arm, arm), (-arm, arm), (-arm, -arm), (arm, -arm)]
    for cx, cy in rotor_centres:
        c = _circle(rotor_r, 18)
        c[0] += cx
        c[1] += cy
        parts.append(c)
        parts.append(np.array([[cx, cx], [cy, cy], [0.0, rotor_r * 0.45]]))
    parts.append(np.array([[arm, -arm], [arm, -arm], [0.0, 0.0]]))
    parts.append(np.array([[-arm, arm], [arm, -arm], [0.0, 0.0]]))
    s = arm * 0.25
    parts.append(np.array([[s, -s, -s, s, s],
                           [s, s, -s, -s, s],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]))
    parts.append(np.array([[0.0, arm * 1.5], [0.0, 0.0], [0.0, 0.0]]))
    return parts


def cylinder_mesh(length: float = 0.55, radius: float = 0.12,
                  n_theta: int = 18, n_ribs: int = 6) -> list[np.ndarray]:
    """Body-frame line segments for an AUV cylinder."""
    t = np.linspace(0.0, 2.0 * math.pi, n_theta + 1)
    cy = radius * np.cos(t)
    cz = radius * np.sin(t)
    half = length / 2.0
    parts: list[np.ndarray] = []
    parts.append(np.stack([np.full_like(t, +half), cy, cz], axis=0))
    parts.append(np.stack([np.full_like(t, -half), cy, cz], axis=0))
    for ang in np.linspace(0.0, 2.0 * math.pi, n_ribs, endpoint=False):
        y = radius * math.cos(ang)
        z = radius * math.sin(ang)
        parts.append(np.array([[+half, -half], [y, y], [z, z]]))
    parts.append(np.array([[-half * 0.85, -half * 0.15],
                           [0.0, 0.0],
                           [+radius, +radius * 2.6]]))
    parts.append(np.array([[-half * 0.85, -half * 0.15],
                           [0.0, 0.0],
                           [+radius * 2.6, +radius * 2.6]]))
    parts.append(np.array([[+half, +half * 1.25],
                           [0.0, 0.0], [0.0, 0.0]]))
    return parts


def transform_body(parts: list[np.ndarray], x: float, y: float, z: float,
                   yaw: float) -> list[np.ndarray]:
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    out: list[np.ndarray] = []
    for p in parts:
        q = Rz @ p
        q[0] += x
        q[1] += y
        q[2] += z
        out.append(q)
    return out


def merge_parts(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate body parts into one (3, K) array with NaN-column gaps
    so a single ``Line3D`` can render the whole body discontinuously.

    matplotlib breaks the line at any NaN, which lets us draw the entire
    body with one artist — far cheaper than one Line3D per part."""
    if not parts:
        return np.zeros((3, 0))
    nan_col = np.full((3, 1), np.nan)
    pieces: list[np.ndarray] = []
    for i, p in enumerate(parts):
        if i > 0:
            pieces.append(nan_col)
        pieces.append(p)
    return np.concatenate(pieces, axis=1)


def transform_merged(merged: np.ndarray, x: float, y: float, z: float,
                     yaw: float) -> np.ndarray:
    """Yaw + translate a merged body in one matmul, preserving NaN gaps."""
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    out = Rz @ merged
    out[0] += x
    out[1] += y
    out[2] += z
    return out


def tune_torch_threads() -> None:
    """Pin torch to a single CPU thread.  These nets are tiny — the
    multi-thread launch overhead is larger than any parallelism win."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------

def run(platform: str, save_path: str | None = None,
        max_frames: int | None = None, fps: int = 25,
        speed: int = 4, body_scale_frac: float | None = None) -> None:
    if platform not in ("auv", "drone"):
        raise SystemExit("animate_3d only supports auv / drone — "
                         "use scripts/animate.py for unicycle.")

    cfg = Config()
    cfg.nn.hidden_dims = [256, 256, 256, 256]

    tune_torch_threads()
    print(f"Loading {platform} model ...")
    nn_model = load_model(platform)
    if nn_model is None:
        raise SystemExit(f"Could not load checkpoint for {platform}.")

    positions, n_steps, dt = make_trajectory(
        platform, traj_key=1,
        total_time=PLATFORM_CFG[platform]["total_time"])
    if max_frames is not None:
        n_steps = min(n_steps, max_frames)

    ctrl_states = [ControllerState(i, nn_model, dt, device=cfg.device,
                                   positions=positions, platform=platform)
                   for i in range(len(CONTROLLERS))]

    px = np.array([p[0] for p in positions[:n_steps]])
    py = np.array([p[1] for p in positions[:n_steps]])
    pz = np.array([p[2] for p in positions[:n_steps]])
    extent_xy = max(float(px.max() - px.min()),
                    float(py.max() - py.min()),
                    1e-3)
    # Per-platform default body fraction; drone bodies look chunkier
    # because they span 2*arm, so use a smaller fraction by default.
    default_frac = {"drone": 0.025, "auv": 0.05}.get(platform, 0.04)
    frac = body_scale_frac if body_scale_frac is not None else default_frac
    body_scale = extent_xy * frac
    if platform == "drone":
        body_template = quadcopter_mesh(arm=body_scale)
    else:
        body_template = cylinder_mesh(length=body_scale * 3.0,
                                      radius=body_scale * 0.7)

    fig = plt.figure(figsize=(11.5, 8.0))
    try:
        fig.canvas.manager.set_window_title(
            f"ACE 3-D — {PLATFORM_TITLES[platform]}")
    except Exception:
        pass
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0.10, 0.10, 0.14))
    fig.patch.set_facecolor((0.10, 0.10, 0.14))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.12, 0.12, 0.18, 1.0))
        axis.label.set_color((0.85, 0.85, 0.9))
        axis.line.set_color((0.5, 0.5, 0.6))
        for tk in axis.get_ticklabels():
            tk.set_color((0.8, 0.8, 0.85))

    ax.plot(px, py, pz, color=(0.85, 0.85, 0.30), linestyle="--",
            linewidth=1.2, alpha=0.75, label="Reference")

    pad_xy = max(extent_xy * 0.12, body_scale * 2.0)
    extent_z = max(float(pz.max() - pz.min()), 1e-3)
    pad_z = max(extent_z * 0.20, body_scale * 2.0)
    ax.set_xlim(float(px.min()) - pad_xy, float(px.max()) + pad_xy)
    ax.set_ylim(float(py.min()) - pad_xy, float(py.max()) + pad_xy)
    ax.set_zlim(float(pz.min()) - pad_z, float(pz.max()) + pad_z)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(PLATFORM_TITLES[platform], color=(0.95, 0.95, 1.0))
    ax.view_init(elev=22, azim=-55)

    target_dot, = ax.plot([], [], [], marker="o", color=(1.0, 0.86, 0.20),
                          markersize=7, linestyle="None", alpha=0.9)

    def _color01(rgb_int):
        return (rgb_int[0] / 255.0, rgb_int[1] / 255.0, rgb_int[2] / 255.0)

    # Single Line3D per body using NaN-separated mesh — drastically
    # cuts matplotlib redraw time vs one Line3D per part.
    body_template_merged = merge_parts(body_template)
    body_lines: list = []
    trail_lines: list = []
    for cs in ctrl_states:
        col = _color01(cs.info["color"])
        ln, = ax.plot([], [], [], color=col, linewidth=1.6)
        body_lines.append(ln)
        tr, = ax.plot([], [], [], color=col, linewidth=1.0, alpha=0.45)
        trail_lines.append(tr)

    handles = [plt.Line2D([0], [0], color=_color01(c["color"]),
                          linewidth=2.5, label=c["short"])
               for c in CONTROLLERS]
    handles.append(plt.Line2D([0], [0], color=(0.85, 0.85, 0.30),
                              linestyle="--", label="Ref"))
    leg = ax.legend(handles=handles, loc="upper left", framealpha=0.85,
                    facecolor=(0.18, 0.18, 0.22),
                    edgecolor=(0.4, 0.4, 0.5), fontsize=9)
    for txt in leg.get_texts():
        txt.set_color((0.95, 0.95, 1.0))

    # Per-method live error readout: anchored top-right so it never
    # collides with the legend.
    info_text = fig.text(0.985, 0.965, "", color=(0.95, 0.95, 1.0),
                         fontsize=10, family="monospace",
                         ha="right", va="top",
                         bbox=dict(facecolor=(0.18, 0.18, 0.22),
                                   edgecolor=(0.4, 0.4, 0.5),
                                   boxstyle="round,pad=0.4", alpha=0.85))
    # Scenario status (mass / friction / wind / noise) at bottom-left.
    scenario_text = fig.text(0.015, 0.04, "", color=(1.0, 0.86, 0.20),
                             fontsize=10, family="monospace",
                             ha="left", va="bottom",
                             bbox=dict(facecolor=(0.18, 0.18, 0.22),
                                       edgecolor=(0.4, 0.4, 0.5),
                                       boxstyle="round,pad=0.4", alpha=0.9))
    # Keybindings hint at bottom-right.
    fig.text(0.985, 0.04,
             "5:Nominal  6:Wind  7:Heavy+Fast  8:Noisy   space:pause   q:quit",
             color=(0.7, 0.7, 0.8), fontsize=8, ha="right", va="bottom")

    state = {"step": 0, "paused": False, "scenario_key": 5}
    trail_max = 80

    def _format_scenario(scen_key: int) -> str:
        sc = SCENARIOS[scen_key]
        p = sc["params"]
        wind_line = (f"  wind=({p['wind_x']:+.1f}, {p['wind_y']:+.1f}) N"
                     if platform == "drone" else "")
        return (f"Scenario {scen_key} — {sc['name']}\n"
                f"  mass×{p['mass']:.2f}   drag×{p['friction']:.2f}\n"
                f"  sensor σ={p['state_noise']:.2f}   "
                f"actuator σ={p['noise']:.2f}{wind_line}")

    def _set_scenario(scen_key: int) -> None:
        if scen_key not in SCENARIOS:
            return
        state["scenario_key"] = scen_key
        params = SCENARIOS[scen_key]["params"]
        for cs in ctrl_states:
            apply_scenario_to_env(platform, cs.env, params)
        scenario_text.set_text(_format_scenario(scen_key))

    def _on_key(event):
        if event.key in {"5", "6", "7", "8"}:
            _set_scenario(int(event.key))
        elif event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _set_scenario(5)

    sim_steps_per_frame = max(1, int(speed))

    def update(_frame_idx: int):
        if state["paused"]:
            return ()
        # Run several sim ticks per render frame so playback feels fast
        # without dragging the matplotlib canvas down to per-tick redraws.
        for _ in range(sim_steps_per_frame):
            k = state["step"]
            if k >= n_steps:
                break
            for cs in ctrl_states:
                cs.step_sim(positions, k, dt, n_steps)
            state["step"] += 1
        k = max(0, state["step"] - 1)

        ri = min(k + 3, n_steps - 1)
        target_dot.set_data_3d([positions[ri][0]],
                               [positions[ri][1]],
                               [positions[ri][2]])

        for cs, body_ln, trail in zip(ctrl_states, body_lines, trail_lines):
            true_state = cs.env.get_state()
            x, y, z = state_xyz(true_state, platform)
            yaw = state_heading(platform, true_state)
            v = transform_merged(body_template_merged, x, y, z, yaw)
            body_ln.set_data_3d(v[0], v[1], v[2])
            if cs.trail:
                tail = np.asarray(cs.trail[-trail_max:])
                trail.set_data_3d(tail[:, 0], tail[:, 1], tail[:, 2])

        info_lines = [f"step  {k:4d}/{n_steps}",
                      f"t     {k * dt:5.2f} s"]
        for cs in ctrl_states:
            short = cs.info["short"]
            err = cs.errors[-1] if cs.errors else float("nan")
            info_lines.append(f"{short:>6s}  {err * 1000:5.0f} mm")
        info_text.set_text("\n".join(info_lines))

        artists = [target_dot, info_text, scenario_text]
        artists.extend(body_lines)
        artists.extend(trail_lines)
        return tuple(artists)

    interval_ms = max(1, int(1000.0 / fps))
    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval_ms,
                        blit=False, repeat=False)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".",
                    exist_ok=True)
        print(f"Saving animation to {save_path} ...")
        if save_path.endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=fps))
        else:
            try:
                anim.save(save_path,
                          writer=FFMpegWriter(fps=fps, bitrate=3000))
            except Exception as e:
                fallback = save_path.rsplit(".", 1)[0] + ".gif"
                print(f"FFMpeg unavailable ({e}); writing {fallback}")
                anim.save(fallback, writer=PillowWriter(fps=fps))
        print("Done.")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["auv", "drone"], default="drone")
    parser.add_argument("--save", default=None,
                        help="If set, write to this file (.mp4 or .gif) "
                             "instead of showing the window.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--speed", type=int, default=4,
                        help="Sim ticks per render frame (default 4). "
                             "Higher = faster playback.")
    parser.add_argument("--body-scale", type=float, default=None,
                        help="Body size as a fraction of trajectory xy "
                             "extent. Default 0.025 for drone, 0.05 for "
                             "AUV. Lower = smaller body.")
    args = parser.parse_args()
    run(args.platform, save_path=args.save,
        max_frames=args.max_frames, fps=args.fps,
        speed=args.speed, body_scale_frac=args.body_scale)


if __name__ == "__main__":
    main()
