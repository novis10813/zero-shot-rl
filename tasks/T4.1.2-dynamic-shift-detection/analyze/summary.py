"""Table of dynamic-shift scores D per dynamics config, from the traces of
zero_shot_rl.eval.shift_trace.

For each config and training seed it reads data/inputs/shift_s<seed>_<config>.npz and writes
data/summary.csv with

    D_mean, D_median, D_p95   over all steps and episodes
    auroc_step     P(D_step of the config > D_step of baseline) over steps, baseline vs config
    auroc_episode  the same for the mean D of each episode
    ret            mean return

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.2-dynamic-shift-detection uv run python -m analyze.summary [--check]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

TASK = Path(__file__).resolve().parents[1]
CONFIGS = ["baseline", "friction_mild", "friction_strong", "actuator_mild", "actuator_strong"]
FIELDS = ["seed", "config", "D_mean", "D_median", "D_p95", "auroc_step", "auroc_episode", "ret"]


def auroc(negative, positive):
    """P(positive > negative) with ties counted half, by rank sum."""
    ranks = rankdata(np.concatenate([negative, positive]))
    n, m = len(negative), len(positive)
    return (ranks[n:].sum() - m * (m + 1) / 2) / (n * m)


def rows():
    out = []
    for seed in range(3):
        trace = {c: np.load(TASK / f"data/inputs/shift_s{seed}_{c}.npz") for c in CONFIGS}
        base = trace["baseline"]["D"]
        for c in CONFIGS:
            d = trace[c]["D"]
            out.append({
                "seed": seed, "config": c, "D_mean": d.mean(), "D_median": np.median(d),
                "D_p95": np.quantile(d, 0.95),
                "auroc_step": auroc(base.ravel(), d.ravel()),
                "auroc_episode": auroc(base.mean(1), d.mean(1)),
                "ret": trace[c]["return"].mean(),
            })
    return out


def render(records):
    lines = [",".join(FIELDS)]
    for r in records:
        lines.append(",".join(str(r[k]) if k in ("seed", "config") else f"{r[k]:.6g}" for k in FIELDS))
    return "\n".join(lines) + "\n"


def main():
    text = render(rows())
    path = TASK / "data/summary.csv"
    if "--check" in sys.argv:
        if path.read_text() != text:
            sys.exit(f"{path} differs from the regenerated table")
        print("ok")
    else:
        path.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
