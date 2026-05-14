"""1-D evidential-regression primer: a textbook ε/a disentanglement demo.

Trains a small Normal-Inverse-Gamma network from scratch on a synthetic
1-D regression task with three deliberately-designed training regions:

  Region 1  x ∈ [-3.0, -1.5]   y = sin(x)              clean
  Region 2  x ∈ [-1.5,  0.0]   y = sin(x) + N(0, 0.4)  NOISY targets
  GAP       x ∈ [ 0.0,  1.5]   no training data         OOD
  Region 3  x ∈ [ 1.5,  3.0]   y = sin(x)              clean

With this dataset the NIG head has separate signals to anchor each
uncertainty channel: aleatoric grows in region 2 (irreducible target
noise), epistemic grows in the gap (no training coverage). The clean
regions show neither.

This is the demonstration Amini et al. (NeurIPS 2020) actually run.
The unicycle-controller offline model in this repo doesn't disentangle
because it was trained on noise-free synthetic rollouts — the aleatoric
head has no noisy regime to anchor to. This standalone primer shows
what evidential regression looks like when the training data IS
designed to support disentanglement.

Output: website/assets/animations/evidential_primer.mp4
"""
from __future__ import annotations
import argparse, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.evidential_net import (
    EvidentialNet, evidential_loss, epistemic_score, aleatoric_score,
)
from scripts._demo_palette import (
    apply_style, COL_BG, COL_PANEL, COL_TEXT, COL_MUTED, COL_ACE,
)

# --- problem definition ----------------------------------------------------
REGIONS = [
    dict(lo=-3.0, hi=-1.5, sigma=0.0, kind="clean low"),
    dict(lo=-1.5, hi= 0.0, sigma=0.4, kind="NOISY targets"),
    dict(lo= 0.0, hi= 1.5, sigma=0.0, kind="OOD gap (no data)", drop=True),
    dict(lo= 1.5, hi= 3.0, sigma=0.0, kind="clean high"),
]
REGION_COL = ("#f0fdf4", "#fef3c7", "#fee2e2", "#f0fdf4")
N_PER_REGION = 80
TRAIN_EPOCHS = 4000
SEED = 0


def make_data(seed: int = SEED):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for r in REGIONS:
        if r.get("drop"):
            continue   # OOD gap — no training data
        x = rng.uniform(r["lo"], r["hi"], N_PER_REGION)
        y = np.sin(x) + rng.normal(0.0, r["sigma"], N_PER_REGION)
        xs.append(x); ys.append(y)
    x = np.concatenate(xs); y = np.concatenate(ys)
    order = np.argsort(x)
    return x[order].astype(np.float32), y[order].astype(np.float32)


def train_model(x: np.ndarray, y: np.ndarray, epochs: int = TRAIN_EPOCHS):
    """Train a small NIG net (1 → [64,64,64] → 4 NIG params). Returns
    (model, loss_history)."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EvidentialNet(1, 1, hidden_dims=[64, 64, 64], use_layernorm=False)
    # Disable LayerNorm/BN; for a tiny 1-D problem they hurt.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x_t = torch.from_numpy(x).unsqueeze(-1)
    y_t = torch.from_numpy(y).unsqueeze(-1)
    losses = []
    for epoch in range(epochs):
        gamma, nu, alpha, beta = model(x_t)
        loss = evidential_loss(gamma, nu, alpha, beta, y_t, lambda_reg=0.01)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.item()))
    model.eval()
    return model, np.array(losses)


def evaluate(model: EvidentialNet, x_grid: np.ndarray):
    with torch.no_grad():
        x_t = torch.from_numpy(x_grid.astype(np.float32)).unsqueeze(-1)
        gamma, nu, alpha, beta = model(x_t)
        eps = epistemic_score(nu, alpha, beta).numpy()
        ale = aleatoric_score(nu, alpha, beta).numpy()
        mu  = gamma.squeeze(-1).numpy()
    return mu, eps, ale


# --- animation ------------------------------------------------------------
RENDER_FPS = 60
DPI        = 110
PROBE_PERIOD_SEC = 8.0   # one left-to-right sweep takes this long
N_SWEEPS         = 2      # total clip length = PROBE_PERIOD * N_SWEEPS


def build_animation(x_grid, mu, eps, ale, x_train, y_train):
    apply_style()
    sigma_tot = np.sqrt(np.clip(eps + ale, 1e-8, None))
    sigma_epi = np.sqrt(np.clip(eps, 1e-8, None))
    fig = plt.figure(figsize=(15, 8.5), dpi=DPI)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.7, 1.0, 1.0], hspace=0.30)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.08)
    ax_fit = fig.add_subplot(gs[0, 0])
    ax_eps = fig.add_subplot(gs[1, 0], sharex=ax_fit)
    ax_ale = fig.add_subplot(gs[2, 0], sharex=ax_fit)

    fig.suptitle("Deep Evidential Regression — disentangling epistemic vs aleatoric",
                 y=0.965, fontsize=20, weight="bold", color=COL_TEXT)
    fig.text(0.5, 0.918,
             "1-D demo: y = sin(x)   •   training data has THREE regimes   •   "
             "ε rises in OOD gap   •   a rises where targets were noisy",
             ha="center", fontsize=12, color=COL_MUTED, style="italic")

    # shading
    def shade(ax):
        for r, c in zip(REGIONS, REGION_COL):
            ax.axvspan(r["lo"], r["hi"], color=c, alpha=0.7, zorder=0)

    # ---- Top: fit + uncertainty bands ---------------------------------
    shade(ax_fit)
    ax_fit.plot(x_grid, np.sin(x_grid), color="#1f2937", lw=2.0,
                 label="true y = sin(x)", zorder=2)
    ax_fit.plot(x_grid, mu, color=COL_ACE, lw=2.4, label="model μ(x)",
                 zorder=3)
    ax_fit.fill_between(x_grid, mu - 2*sigma_tot, mu + 2*sigma_tot,
                         color=COL_ACE, alpha=0.18, label="±2σ total",
                         zorder=1)
    ax_fit.fill_between(x_grid, mu - 2*sigma_epi, mu + 2*sigma_epi,
                         color=COL_ACE, alpha=0.30, label="±2σ epistemic",
                         zorder=1)
    ax_fit.scatter(x_train, y_train, s=18, color="#374151", alpha=0.65,
                    zorder=4, label="training data")
    ax_fit.set_xlim(-3.05, 3.05); ax_fit.set_ylim(-2.0, 2.0)
    ax_fit.set_ylabel("y", fontsize=13, color=COL_MUTED)
    ax_fit.set_title("Function + predicted mean + uncertainty bands",
                      color=COL_MUTED, fontsize=15)
    ax_fit.grid(alpha=0.4)
    ax_fit.legend(loc="upper left", fontsize=11, frameon=False, ncol=2)
    for r, c in zip(REGIONS, REGION_COL):
        ax_fit.text((r["lo"] + r["hi"]) / 2, 1.85, r["kind"],
                     ha="center", va="top", fontsize=10,
                     color="#374151", weight="semibold",
                     bbox=dict(boxstyle="round,pad=0.25",
                               facecolor="white", edgecolor="#d1d5db",
                               alpha=0.85))
    probe_line_fit = ax_fit.axvline(x_grid[0], color="#dc2626", lw=2.0, alpha=0.85,
                                     zorder=5)
    probe_pt = ax_fit.plot([x_grid[0]], [mu[0]], "o", color="#dc2626", ms=12,
                            mec="white", mew=1.5, zorder=6)[0]

    # ---- Middle: ε(x) -------------------------------------------------
    shade(ax_eps)
    eps_max = max(float(eps.max()), 1e-6) * 1.15
    ax_eps.plot(x_grid, eps, color=COL_ACE, lw=2.4)
    ax_eps.set_ylabel("ε (epistemic)", fontsize=13, color=COL_ACE)
    ax_eps.tick_params(axis="y", labelcolor=COL_ACE)
    ax_eps.set_title("ε(x) — model says 'I have not seen this input'",
                      color=COL_MUTED, fontsize=14)
    ax_eps.grid(alpha=0.4)
    ax_eps.set_ylim(0, eps_max)
    probe_line_eps = ax_eps.axvline(x_grid[0], color="#dc2626", lw=2.0, alpha=0.85)
    probe_eps_pt = ax_eps.plot([x_grid[0]], [eps[0]], "o", color="#dc2626", ms=10,
                                mec="white", mew=1.2)[0]

    # ---- Bottom: a(x) ------------------------------------------------
    shade(ax_ale)
    ale_max = max(float(ale.max()), 1e-6) * 1.15
    ax_ale.plot(x_grid, ale, color="#b45309", lw=2.4)
    ax_ale.set_ylabel("a (aleatoric)", fontsize=13, color="#b45309")
    ax_ale.tick_params(axis="y", labelcolor="#b45309")
    ax_ale.set_title("a(x) — model says 'this target was inherently noisy'",
                      color=COL_MUTED, fontsize=14)
    ax_ale.set_xlabel("x", fontsize=13, color=COL_MUTED)
    ax_ale.grid(alpha=0.4)
    ax_ale.set_ylim(0, ale_max)
    probe_line_ale = ax_ale.axvline(x_grid[0], color="#dc2626", lw=2.0, alpha=0.85)
    probe_ale_pt = ax_ale.plot([x_grid[0]], [ale[0]], "o", color="#dc2626", ms=10,
                                mec="white", mew=1.2)[0]

    # callout text
    callout = fig.text(0.5, 0.88, "", ha="center", va="top",
                        fontsize=14, weight="semibold", color="#9a3412")

    def which_region(xv: float) -> int:
        for i, r in enumerate(REGIONS):
            if r["lo"] <= xv < r["hi"]:
                return i
        return len(REGIONS) - 1

    def callout_text(xv: float) -> str:
        idx = which_region(xv)
        labels = [
            "Clean region — both ε and a near zero  (model is confident)",
            "Noisy region — a rises (irreducible target noise), ε stays low",
            "OOD gap — ε rises sharply (no training data), a is moderate",
            "Clean region — both ε and a near zero  (model is confident)",
        ]
        return labels[idx]

    n_frames_per_sweep = int(round(RENDER_FPS * PROBE_PERIOD_SEC))
    n_frames = N_SWEEPS * n_frames_per_sweep

    def update(i: int):
        # Sweep x left → right, then snap back, repeat
        phase = (i % n_frames_per_sweep) / n_frames_per_sweep
        xv = x_grid[0] + phase * (x_grid[-1] - x_grid[0])
        kx = int(np.argmin(np.abs(x_grid - xv)))
        probe_line_fit.set_xdata([xv, xv])
        probe_line_eps.set_xdata([xv, xv])
        probe_line_ale.set_xdata([xv, xv])
        probe_pt.set_data([xv], [mu[kx]])
        probe_eps_pt.set_data([xv], [eps[kx]])
        probe_ale_pt.set_data([xv], [ale[kx]])
        callout.set_text(callout_text(xv))
        return [probe_line_fit, probe_line_eps, probe_line_ale,
                probe_pt, probe_eps_pt, probe_ale_pt, callout]

    anim = FuncAnimation(fig, update, frames=n_frames,
                          interval=1000.0 / RENDER_FPS, blit=False)
    return fig, anim


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",
                   default=str(ROOT.parent / "assets" / "animations"))
    p.add_argument("--bitrate", type=int, default=1400)
    args = p.parse_args(argv)
    _check_ffmpeg()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evidential_primer.mp4"

    print("Generating training data ...")
    x_train, y_train = make_data()
    print(f"  N = {len(x_train)}; regions = {[r['kind'] for r in REGIONS]}")
    print("Training NIG model ...")
    t0 = time.perf_counter()
    model, losses = train_model(x_train, y_train)
    print(f"  trained in {time.perf_counter()-t0:.1f}s  "
          f"final loss = {losses[-1]:.4f}")
    print("Evaluating on grid ...")
    x_grid = np.linspace(-3.0, 3.0, 600)
    mu, eps, ale = evaluate(model, x_grid)

    # Print region-mean summary for the record
    def stats_in(lo, hi):
        mask = (x_grid >= lo) & (x_grid < hi)
        return eps[mask].mean(), ale[mask].mean()
    print("\nRegion-mean ε and a:")
    for r in REGIONS:
        e, a = stats_in(r["lo"], r["hi"])
        print(f"  {r['kind']:<22}  ε̄ = {e:.4f}   ā = {a:.4f}")

    print("Building animation ...")
    t0 = time.perf_counter()
    fig, anim = build_animation(x_grid, mu, eps, ale, x_train, y_train)
    writer = FFMpegWriter(fps=RENDER_FPS, codec="libx264",
                           bitrate=args.bitrate,
                           extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"Rendered in {time.perf_counter() - t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
