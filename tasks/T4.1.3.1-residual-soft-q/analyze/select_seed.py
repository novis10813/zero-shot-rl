"""Per-seed scores of the vanilla FB runs at the step analysis.table6_aggregate selects.

Reads data/fb_train/eval_fb_s<seed>.csv (analysis/extract_eval.py of each training log), picks the
step with the highest all-task IQM of the seed means, and prints each seed's four task scores,
their IQM and their mean at that step. The FB used downstream is the seed of the highest IQM.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.select_seed
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.table6_aggregate import TASKS, aggregate, iqm

DATA = Path(__file__).resolve().parents[1] / "data" / "fb_train"
SEEDS = range(5)


def main():
    runs = [pd.read_csv(DATA / f"eval_fb_s{s}.csv").set_index("step") for s in SEEDS]
    scores = np.stack([run[TASKS].to_numpy() for run in runs])
    all_task = aggregate(scores)
    step = int(runs[0].index[int(np.argmax(all_task))])
    print(f"selected step {step}, all-task IQM {all_task.max():.1f}")
    best = max(SEEDS, key=lambda s: iqm(runs[s].loc[step, TASKS].to_numpy()))
    for s in SEEDS:
        row = runs[s].loc[step, TASKS].to_numpy()
        tasks = " ".join(f"{t} {v:.1f}" for t, v in zip(TASKS, row))
        print(f"seed {s}: {tasks}, IQM {iqm(row):.1f}, mean {row.mean():.1f}")
    print(f"selected seed {best}")


if __name__ == "__main__":
    main()
