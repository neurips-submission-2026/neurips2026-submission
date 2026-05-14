#!/usr/bin/env python3
"""Offline training for the 3-D platforms (AUV or drone).

Mirrors `train_unicycle.py` but targets the 3-D environments. The
pipeline is:

  1. Collect transitions via `collect_diverse_dataset_3d`
     (PE = persistent excitation = sum-of-sinusoids + lemniscate_3d).
  2. Phase 1: train an `InverseModelMLP` with MSE loss.
  3. Phase 2: distil into an `EvidentialNet` with the NIG loss.
  4. Save the evidential checkpoint to ``pretrained/<platform>.pth``.

Usage:
    python scripts/train_3d.py --platform auv
    python scripts/train_3d.py --platform drone --fast
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.auv3d_env import AUV3DEnv
from envs.drone3d_env import Drone3DEnv
from training.data_collection import collect_diverse_dataset_3d
from training.train_offline import train_two_phase


PLATFORM_INFO = {
    "auv":   dict(env_cls=AUV3DEnv,    dt=0.05, ckpt_name="auv.pth"),
    "drone": dict(env_cls=Drone3DEnv,  dt=0.02, ckpt_name="drone.pth"),
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--platform", choices=["auv", "drone"], required=True)
    p.add_argument("--data-mode", choices=["pe", "lemniscate"],
                   default="pe",
                   help="`pe` = broadband persistent excitation + "
                        "lemniscate_3d (default); `lemniscate` = lemniscate-only.")
    p.add_argument("--domain-rand", action="store_true",
                   help="Per-rollout sampling of mass / drag / wind.")
    p.add_argument("--feature-mode", choices=["invariant", "full"],
                   default="invariant",
                   help="Inverse-dynamics input encoding. `invariant` drops "
                        "absolute psi/z and trains in body frame; `full` keeps them.")
    p.add_argument("--fast", action="store_true",
                   help="Smaller dataset and model (~3 min).")
    p.add_argument("--demo", action="store_true",
                   help="Smallest smoke-test (~1 min).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="pretrained",
                   help="Where to write {auv,drone}.pth")
    p.add_argument("--phase2-lambda-reg", type=float, default=0.005,
                   help="NIG regulariser strength.")
    args = p.parse_args(argv)

    info = PLATFORM_INFO[args.platform]
    if args.demo:
        args.fast = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print(f"  Step 2 (3-D): {args.platform.upper()} inverse dynamics")
    print(f"  data_mode={args.data_mode}  domain_rand={args.domain_rand}  "
          f"feature_mode={args.feature_mode}  fast={args.fast}  demo={args.demo}")
    print("=" * 60)

    def make_env():
        env = info["env_cls"](dt=info["dt"])
        env.reset()
        return env

    print("\n[1/3] Collecting 3-D transitions ...")
    t0 = time.perf_counter()
    inputs, targets = collect_diverse_dataset_3d(
        make_env, kind=args.platform, dt=info["dt"],
        feature_mode=args.feature_mode,
        data_mode=args.data_mode, domain_rand=args.domain_rand,
        fast=args.fast, demo=args.demo, seed=args.seed)
    print(f"  collected {len(inputs):,} samples in {time.perf_counter() - t0:.1f}s")

    if args.demo:
        hidden_dims = [64, 64]
        train_kwargs = dict(phase1_epochs=10, phase2_epochs=5,
                             device="cpu", fast=True)
    elif args.fast:
        hidden_dims = [128, 128]
        train_kwargs = dict(device="cpu", fast=True)
    else:
        hidden_dims = [256, 256, 256, 256]
        train_kwargs = dict(device="cpu")

    print(f"\n[2/3] Training ({hidden_dims}) ...")
    t0 = time.perf_counter()
    # train_two_phase takes the architecture as keyword `hidden_dims` if
    # we route it through; otherwise we patch the model after construction.
    # Simpler: pass `model_kwargs` through; or rebuild the trainer.
    # The existing train_two_phase signature uses hidden_dims via cfg,
    # so we pass it directly if supported.  Check signature.
    mlp, evid, tl, vl = train_two_phase(
        inputs, targets,
        hidden_dim=hidden_dims[0],
        n_blocks=len(hidden_dims),
        evid_hidden_dims=tuple(hidden_dims),
        phase2_lambda_reg=args.phase2_lambda_reg,
        **train_kwargs)
    print(f"  trained in {time.perf_counter() - t0:.1f}s")

    print(f"\n[3/3] Saving checkpoints to {args.out_dir} ...")
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = Path(args.out_dir) / info["ckpt_name"]
    state_dict = evid.state_dict()
    ckpt = {
        "model_state":  state_dict,
        "model_type":   "inverse_dynamics",
        "input_mean":   evid.input_mean,
        "input_std":    evid.input_std,
        "target_mean":  evid.target_mean,
        "target_std":   evid.target_std,
        "input_dim":    int(inputs.shape[1]),
        "output_dim":   int(targets.shape[1]),
        "feature_mode": args.feature_mode,
        "hidden_dims":  hidden_dims,
        "data_mode":    args.data_mode,
        "domain_rand":  bool(args.domain_rand),
        "platform":     args.platform,
        "dt":           info["dt"],
        "use_layernorm": bool(getattr(evid, "use_layernorm", False)),
    }
    evid.feature_mode = args.feature_mode
    torch.save(ckpt, out_path)
    print(f"  wrote {out_path}")

    # Also save the Phase 1 MLP for fall-back at inference (used when
    # Phase 2's NIG distillation distorts the mean prediction; see
    # `docs/iter-28-mu-mse-analysis.md`).
    mlp_path = Path(args.out_dir) / info["ckpt_name"].replace(".pth", "_mlp.pth")
    mlp_ckpt = {
        "model_state":  mlp.state_dict(),
        "model_type":   "inverse_dynamics_mlp",
        "input_mean":   mlp.input_mean,
        "input_std":    mlp.input_std,
        "target_mean":  mlp.target_mean,
        "target_std":   mlp.target_std,
        "input_dim":    int(inputs.shape[1]),
        "output_dim":   int(targets.shape[1]),
        "feature_mode": args.feature_mode,
        "hidden_dim":   hidden_dims[0],
        "n_blocks":     len(hidden_dims),
        "platform":     args.platform,
        "dt":           info["dt"],
    }
    torch.save(mlp_ckpt, mlp_path)
    print(f"  wrote {mlp_path}")

    return out_path


if __name__ == "__main__":
    main()
