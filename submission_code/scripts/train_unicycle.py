"""Offline training of the unicycle inverse-dynamics model.

Two-phase training:
  Phase 1: `InverseModelMLP` with MSE loss.
  Phase 2: distil into an `EvidentialNet` with the NIG loss.

The evidential checkpoint is saved to ``pretrained/unicycle.pth``.

Usage:
    python scripts/train_unicycle.py
    python scripts/train_unicycle.py --fast    # ~2 min
    python scripts/train_unicycle.py --demo    # ~30 s smoke-test
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from envs.unicycle_env import UnicycleEnv
from controllers.lqr_controller import LQRController
from controllers.nn_controller import NNController, compute_desired_next_state
from training.data_collection import collect_diverse_dataset
from training.train_offline import train_two_phase
from utils.trajectories import generate_trajectory, lemniscate_2d


def run_closed_loop(model, cfg, model_type="evidential"):
    """Evaluate model as controller on lemniscate."""
    dt = cfg.sim.dt
    n_steps = cfg.sim.n_steps
    r_bar = 4
    traj_fn = lambda t: lemniscate_2d(t, scale=cfg.traj.scale, speed=cfg.traj.speed)
    positions, _ = generate_trajectory(traj_fn, dt, n_steps, r_bar)

    x0, y0 = positions[0]
    dx = positions[2][0] - positions[0][0]
    dy = positions[2][1] - positions[0][1]
    theta0 = np.arctan2(dy, dx)

    env = UnicycleEnv(dt=dt)
    env.reset(state=np.array([x0, y0, theta0, 0.0, 0.0]))

    is_evid = (model_type == "evidential")
    nn_ctrl = NNController(model, device=cfg.device, is_evidential=is_evid)
    actual_xy, errors_l2, actions_log = [], [], []

    for k in range(n_steps):
        state = env.get_state()
        action, info = nn_ctrl.compute(env, state, positions, k, dt)
        actions_log.append(action.copy())
        env.step(action)
        p = env.get_state()[:2]
        actual_xy.append(p)
        errors_l2.append(np.linalg.norm(p - positions[k]))

    return {"actual_xy": np.array(actual_xy), "desired_xy": positions[:n_steps],
            "errors_l2": np.array(errors_l2), "actions": np.array(actions_log), "dt": dt}


def run_expert(cfg):
    dt = cfg.sim.dt
    n_steps = cfg.sim.n_steps
    r_bar = 4
    traj_fn = lambda t: lemniscate_2d(t, scale=cfg.traj.scale, speed=cfg.traj.speed)
    positions, _ = generate_trajectory(traj_fn, dt, n_steps, r_bar)

    x0, y0 = positions[0]
    dx = positions[2][0] - positions[0][0]
    dy = positions[2][1] - positions[0][1]
    theta0 = np.arctan2(dy, dx)

    env = UnicycleEnv(dt=dt)
    env.reset(state=np.array([x0, y0, theta0, 0.0, 0.0]))
    ctrl = LQRController(env, dt)

    actual_xy, errors_l2, actions_log = [], [], []
    for k in range(n_steps):
        state = env.get_state()
        action, _ = ctrl.compute(state, positions, k)
        actions_log.append(action.copy())
        env.step(action)
        p = env.get_state()[:2]
        actual_xy.append(p)
        errors_l2.append(np.linalg.norm(p - positions[k]))

    return {"actual_xy": np.array(actual_xy), "desired_xy": positions[:n_steps],
            "errors_l2": np.array(errors_l2), "actions": np.array(actions_log), "dt": dt}


def plot_results(expert, mlp_logs, evid_logs, train_losses, val_losses,
                 inputs, targets, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    dt = expert["dt"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, "b-", lw=0.8, label="Train")
    ax.plot(val_losses, "r--", lw=0.8, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training: MSE → Evidential"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, "step2_training_curve.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    des = expert["desired_xy"]
    ax.plot(des[:, 0], des[:, 1], "k--", lw=1.5, label="Reference")
    ax.plot(expert["actual_xy"][:, 0], expert["actual_xy"][:, 1], "g-", lw=1.2, alpha=0.7, label="LQR")
    ax.plot(mlp_logs["actual_xy"][:, 0], mlp_logs["actual_xy"][:, 1], "b-", lw=1.2, alpha=0.7, label="MLP")
    ax.plot(evid_logs["actual_xy"][:, 0], evid_logs["actual_xy"][:, 1], "r-", lw=1, alpha=0.7, label="Evidential")
    ax.legend(); ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Step 2: Inverse Dynamics Tracking"); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, "step2_trajectory.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for logs, name, c in [(expert, "LQR", "g"), (mlp_logs, "MLP", "b"), (evid_logs, "Evidential", "r")]:
        t = np.arange(len(logs["errors_l2"])) * dt
        ax.plot(t, logs["errors_l2"], color=c, lw=0.8, alpha=0.7, label=name)
    ax.legend(); ax.set_xlabel("Time (s)"); ax.set_ylabel("L2 Error (m)")
    ax.set_title("Tracking Error"); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, "step2_error.png"), dpi=150); plt.close(fig)

    print(f"  Plots saved to {save_dir}/step2_*.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: fewer transitions, smaller model, fewer epochs (~2 min)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: minimal data/epochs for quick smoke-test (~30 s)")
    parser.add_argument("--data-mode", choices=["legacy", "pe", "dr"],
                        default="legacy",
                        help="Offline trajectory family. legacy = lemniscate "
                             "mix (default); pe = composite persistent "
                             "excitation; dr = lemniscate with per-rollout "
                             "domain randomisation (use with --domain-rand).")
    parser.add_argument("--domain-rand", action="store_true",
                        help="Per-rollout sampling of mass/friction/wind.")
    parser.add_argument("--out-dir", default="pretrained",
                        help="Where to write the unicycle checkpoint and the "
                             "training-diagnostic plots.")
    args = parser.parse_args()
    if args.demo:
        args.fast = True  # demo implies fast

    cfg = Config()
    cfg.sim.total_time = 10.0 if args.demo else 30.0
    cfg.traj.speed = 0.5
    cfg.traj.scale = 4.0
    if args.demo:
        cfg.nn.hidden_dims = [64, 64]
    elif args.fast:
        cfg.nn.hidden_dims = [128, 128]
    else:
        cfg.nn.hidden_dims = [256, 256, 256, 256]

    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    mode_tag = " [DEMO]" if args.demo else (" [FAST]" if args.fast else "")
    print("=" * 60)
    print(f"  Step 2: Inverse Dynamics Model (Self-Supervised){mode_tag}")
    print("=" * 60)

    print("\n[1/5] Collecting transitions...")
    feature_mode = "full"
    inputs, targets = collect_diverse_dataset(cfg, seed=cfg.seed, fast=args.fast,
                                              demo=args.demo, feature_mode=feature_mode,
                                              data_mode=args.data_mode,
                                              domain_rand=args.domain_rand)

    print("\n[2/5] Training...")
    train_kwargs = dict(device=cfg.device, fast=args.fast)
    if args.demo:
        train_kwargs.update(phase1_epochs=10, phase2_epochs=5)
    mlp, evid, tl, vl = train_two_phase(inputs, targets, **train_kwargs)

    print("\n[3/5] Closed-loop: MLP...")
    mlp_logs = run_closed_loop(mlp, cfg, "mlp")
    skip = int(min(2.0, cfg.sim.total_time * 0.1) / mlp_logs["dt"])
    print(f"  MLP  mean={np.mean(mlp_logs['errors_l2'][skip:]):.4f} m")

    print("\n[4/5] Closed-loop: Evidential...")
    evid_logs = run_closed_loop(evid, cfg, "evidential")
    print(f"  Evid mean={np.mean(evid_logs['errors_l2'][skip:]):.4f} m")

    print("\n[5/5] LQR baseline...")
    expert_logs = run_expert(cfg)
    print(f"  LQR  mean={np.mean(expert_logs['errors_l2'][skip:]):.4f} m")

    plot_results(expert_logs, mlp_logs, evid_logs, tl, vl, inputs, targets,
                 save_dir=args.out_dir)

    print(f"\n{'='*60}")
    for name, logs in [("LQR Expert", expert_logs), ("MLP", mlp_logs), ("Evidential", evid_logs)]:
        err = logs["errors_l2"][skip:]
        print(f"  {name:<15} mean={np.mean(err):.4f}  max={np.max(err):.4f}")
    print(f"{'='*60}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    hdims = list(cfg.nn.hidden_dims)
    for name, model, extra in [
        ("unicycle", evid, {"hidden_dims": hdims}),
        ("unicycle_mlp", mlp, {"model_class": "InverseModelMLP",
                               "hidden_dim": hdims[0], "n_blocks": len(hdims)}),
    ]:
        d = {
            "model_state": model.state_dict(),
            "model_type": "inverse_dynamics",
            "input_mean": model.input_mean, "input_std": model.input_std,
            "target_mean": model.target_mean, "target_std": model.target_std,
            "input_dim": inputs.shape[1], "output_dim": targets.shape[1],
            "feature_mode": feature_mode,
            "data_mode": args.data_mode,
            "domain_rand": bool(args.domain_rand),
            "use_layernorm": bool(getattr(model, "use_layernorm", False)),
        }
        model.feature_mode = feature_mode
        d.update(extra)
        out_path = os.path.join(args.out_dir, f"{name}.pth")
        torch.save(d, out_path)
        print(f"  Saved {out_path}")
