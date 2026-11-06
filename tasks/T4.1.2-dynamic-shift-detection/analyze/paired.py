"""Paired comparison of D between the stock and the shifted one-step transition of the same (s, a, z).

Reads data/inputs/scored_<detector>_s<seed>.npz of zero_shot_rl.eval.paired_score
(detector: dataset or rollout) and writes data/paired.csv, one row per (detector, seed,
config) with

    state_gap      median |s'_config - s'_stock|  (L2 over the observation)
    D_stock, D_config   mean D of the stock and the shifted transition
    diff_mean      mean of D_config - D_stock over the paired samples
    frac_pos       fraction of samples with D_config > D_stock (ties half)
    auroc          P(D_config > D_stock) over all pairs of samples, stock as negatives

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.2-dynamic-shift-detection uv run python -m analyze.paired [--check]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

TASK = Path(__file__).resolve().parents[1]
CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
DETECTORS = ["dataset", "rollout"]
FIELDS = ["detector", "seed", "config", "state_gap", "D_stock", "D_config", "diff_mean", "frac_pos", "auroc"]


def auroc(negative, positive):
    ranks = rankdata(np.concatenate([negative, positive]))
    n, m = len(negative), len(positive)
    return (ranks[n:].sum() - m * (m + 1) / 2) / (n * m)


def rows():
    out = []
    for detector in DETECTORS:
        for seed in range(3):
            data = np.load(TASK / f"data/inputs/scored_{detector}_s{seed}.npz")
            base = data["D_rollout"].ravel()
            if not np.allclose(data["D_baseline"].ravel(), base, rtol=1e-5):
                raise ValueError("replayed baseline does not reproduce the rollout")
            for c in CONFIGS:
                d = data[f"D_{c}"].ravel()
                out.append({
                    "detector": detector, "seed": seed, "config": c, "state_gap": np.median(data[f"gap_{c}"]),
                    "D_stock": base.mean(), "D_config": d.mean(), "diff_mean": (d - base).mean(),
                    "frac_pos": ((d > base) + 0.5 * (d == base)).mean(), "auroc": auroc(base, d),
                })
    return out


def render(records):
    lines = [",".join(FIELDS)]
    for r in records:
        lines.append(",".join(str(r[k]) if k in ("detector", "seed", "config") else f"{r[k]:.6g}" for k in FIELDS))
    return "\n".join(lines) + "\n"


def main():
    text = render(rows())
    path = TASK / "data/paired.csv"
    if "--check" in sys.argv:
        if path.read_text() != text:
            sys.exit(f"{path} differs from the regenerated table")
        print("ok")
    else:
        path.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
