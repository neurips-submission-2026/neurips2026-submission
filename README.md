# ACE: anonymous supplementary (NeurIPS 2026)

This repository accompanies our NeurIPS 2026 submission. At the root
it serves the supplementary website; the code that produced every
number in the paper lives in `submission_code/`, together with
pretrained checkpoints for all three platforms.

## Reproducing the numbers in the paper

The headline table (Section 4 of the paper) drops out of:

```bash
cd submission_code
pip install -r requirements.txt
python scripts/run_benchmark.py
```

This runs three platforms × five scenarios × six methods × three
seeds on CPU in roughly ten minutes and writes
`results/benchmark/multi_platform_summary.csv`. The mean ± std cells
in that file are what we report in the paper. The per-rollout
long-form CSV and the JSON traces used to generate the figures sit
alongside it.

If you would rather retrain the inverse-dynamics networks from
scratch before running the benchmark (about an hour and a half
end-to-end on a 20-thread CPU), train the three platforms in any
order and then re-run the benchmark:

```bash
python scripts/train_unicycle.py --data-mode pe
python scripts/train_3d.py --platform auv  --data-mode pe
python scripts/train_3d.py --platform drone --data-mode pe
python scripts/run_benchmark.py
```

Each training script overwrites the corresponding checkpoint in
`pretrained/`. The unit tests (`pytest tests/`) cover the ACE and
EWC adapters and finish in a couple of seconds.

## Layout

```
.
├── index.html, styles.css, script.js, assets/   supplementary site
└── submission_code/                             source + pretrained models
    ├── envs/ controllers/ models/ training/ utils/
    ├── scripts/    train_unicycle, train_3d, run_benchmark, animate(_3d)
    ├── tests/      pytest suite (ACE, EWC)
    └── pretrained/ unicycle.pth, auv.pth, drone.pth
```

## Anonymity

Nothing in this repository, site or code, names authors,
affiliations, or institutions.
