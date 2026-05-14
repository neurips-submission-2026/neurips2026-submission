# ACE: Active Continual Learning with Evidential Uncertainty

Anonymous code release for the NeurIPS 2026 submission *"ACE: Active
Continual Learning with Evidential Uncertainty for Inverse-Dynamics
Adaptation."*

ACE adapts a pre-trained inverse-dynamics network online to runtime
shifts (payload, drag, wind, sensor and actuator noise) without
forgetting the offline solution. The same hyperparameters are used
across all three mobile-robot platforms it is evaluated on (ground,
water, air).

Pretrained checkpoints are shipped under `pretrained/`, so the
headline benchmark reproduces in a few minutes without retraining
anything. A full retrain takes roughly an hour and a half on CPU.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then check the install with the tests, which finish in a couple of
seconds:

```bash
python3 -m pytest tests/ -q
```

## Layout

```
envs/                 unicycle, AUV-3D, drone-3D simulators
controllers/          LQR (2D / 3D), NN controller, state-feedback
models/               EvidentialNet (NIG head) + InverseModelMLP
training/             offline + online (ACE, ER, EWC) adapters
utils/                trajectories, replay buffers
scripts/              entry points (see below)
tests/                pytest suite for ACE and EWC
pretrained/           NIG checkpoints, one per platform
config.py             centralised hyperparameter dataclasses
```

The checkpoints in `pretrained/` are the same ones whose numbers
appear in the paper's headline table.

## Reproducing the paper results

The shipped checkpoints already match the paper, so the headline
table reproduces directly:

```bash
python3 scripts/run_benchmark.py --out-dir results/benchmark
```

This sweeps 3 platforms × 5 scenarios × 6 methods × 3 random seeds
(270 rollouts) in about ten minutes on CPU and writes:

```
results/benchmark/multi_platform_runs.csv       one row per rollout
results/benchmark/multi_platform_summary.csv    mean ± std per cell
results/benchmark/multi_platform_traces.json    per-cell trajectories
```

The summary CSV is the table reported in the paper: rows are
(platform, method) pairs, columns are distribution-shift scenarios.
For a one-seed sanity check, add `--quick`.

To retrain the inverse-dynamics networks before running the
benchmark, train each platform separately (they are independent):

```bash
python3 scripts/train_unicycle.py --data-mode pe --out-dir pretrained
python3 scripts/train_3d.py --platform auv  --data-mode pe --out-dir pretrained
python3 scripts/train_3d.py --platform drone --data-mode pe --out-dir pretrained
```

Each command overwrites `pretrained/<platform>.pth`. On a 20-thread
CPU, unicycle takes about 30 minutes; AUV and drone take about 30
to 40 minutes each. On a 4-core laptop expect roughly two hours
total.

If you run several platforms in parallel, cap the per-process
thread count so they do not contend for the MKL/OMP pool:

```bash
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 python3 scripts/train_3d.py --platform auv   &
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 python3 scripts/train_3d.py --platform drone &
wait
```

Both training scripts also accept `--fast` (smaller dataset and
model, about three minutes per platform) and `--demo` (smallest,
one minute). These give weaker checkpoints intended only for
smoke-testing the pipeline.

## Animations

Two interactive demos let you watch four controllers run in parallel
under each scenario.

The 2D pygame view (also used as a top-down rendering of the 3D
platforms):

```bash
python3 scripts/animate.py
```

Use the left and right arrow keys to switch platforms (unicycle to
AUV to drone), digits 1 to 4 to switch the unicycle trajectory, 5 to
8 to switch scenario (nominal, wind, heavy, noisy), space to pause,
R to restart, H to toggle the help overlay, and Q or Esc to quit.
The window accepts `--platform {unicycle,auv,drone}` on the command
line if you want to start on a specific platform.

For a true 3D rendering of the AUV or drone with a rotatable camera:

```bash
python3 scripts/animate_3d.py --platform drone
python3 scripts/animate_3d.py --platform auv
```

Both animations load the same checkpoints that the benchmark uses,
so the controllers behave identically to the ones reported in the
paper. To save a clip (ffmpeg required on your system):

```bash
python3 scripts/animate_3d.py --platform drone --save demo.mp4
```

## Regenerate the website demo clips

The supplementary site's "See ACE adapt" section is driven by five
MP4 clips under `../assets/animations/`. To rebuild all of them from
the pretrained checkpoint:

```bash
python3 scripts/render_demo_clips.py --out-dir ../assets/animations
```

To rebuild just one (e.g. after tweaking palette or layout):

```bash
python3 scripts/render_demo_clips.py --scenario wind --out-dir ../assets/animations
```

Rendering all five clips takes roughly ten minutes on CPU. Requires
`ffmpeg` on PATH.

## Method in a paragraph

We train an evidential inverse-dynamics network offline (Normal-
Inverse-Gamma head) that exposes both *epistemic* and *aleatoric*
uncertainty per output dimension in a single forward pass. At
deployment the network predicts an action and a fixed-gain
state-feedback stabiliser is added on top. A noise-discounted
priority replay buffer keyed by `ε / (1 + a)` decides which
transitions to learn from; an exponentially-smoothed schedule
adapts the evidence regulariser; an aleatoric-aware floor blocks
adaptation when the residual is mostly noise; and a constant L²
anchor to the offline weights bounds parameter drift.

The six baselines compared in the benchmark are LQR (fixed gain
from the nominal linearisation), Frozen (pre-trained NN, no online
updates), ER (NN with uniform experience replay), EWC (NN with
elastic weight consolidation), fixed-λ Evidential, and ACE (full
adaptive schedule with anchor and priority replay). All NN methods
share the same offline checkpoint and the same fixed state-feedback
gain.

## Environment

Reference numbers come from a 10-core / 20-thread Xeon workstation
running Linux with Python 3.10 and PyTorch 2.x. The code is
CPU-only and also runs on macOS and Windows. No GPU is needed
because the networks are small and the rollouts are serial.

## Tests

```bash
python3 -m pytest tests/ -q
```

Covers the ACE and EWC adapters: priority calculation, schedule
update, aleatoric-aware floor, noise gates, anchor, end-to-end
smoke tests, Fisher non-negativity, and determinism under a fixed
seed.
