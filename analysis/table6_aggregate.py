"""
Aggregates per-seed eval CSVs the way appendix A.5 and Table 6 of Jeen et al. (2024) describe.

1. Each CSV row is one checkpoint of one seed; each task value is already the IQM of that
   checkpoint's 10 rollouts (logged by zero_shot_rl/train/main_exorl.py).
2. At each checkpoint, a task's score is the mean of that value over seeds.
3. The all-task IQM of a checkpoint is the IQM (25% trimmed mean) of the task scores.
4. The reported checkpoint is the one with the highest all-task IQM. Its per-task scores are
   reported, with a 95% bootstrap interval over seeds (selected checkpoint held fixed).

Run from the repo root:
    PYTHONPATH=. uv run python -m analysis.table6_aggregate <seed0.csv> <seed1.csv> ...
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy import stats

TASKS = ["stand", "walk", "run", "flip"]


def iqm(x: np.ndarray) -> float:
    return float(stats.trim_mean(x, 0.25))


def aggregate(scores: np.ndarray) -> np.ndarray:
    """scores: [seeds, steps, tasks] -> all-task IQM per step, shape [steps]."""
    task_scores = scores.mean(axis=0)  # [steps, tasks]
    return np.array([iqm(row) for row in task_scores])


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = [pd.read_csv(path).set_index("step") for path in args.csvs]
    steps = runs[0].index
    assert all(run.index.equals(steps) for run in runs), "runs must share eval steps"
    scores = np.stack([run[TASKS].to_numpy() for run in runs])  # [seeds, steps, tasks]

    all_task_iqm = aggregate(scores)
    best = int(np.argmax(all_task_iqm))
    task_scores = scores.mean(axis=0)

    rng = np.random.default_rng(args.seed)
    n = len(runs)
    boot = np.stack(
        [scores[rng.integers(0, n, n), best].mean(axis=0) for _ in range(args.reps)]
    )  # [reps, tasks]
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)

    print(f"seeds: {n}, checkpoints: {len(steps)}")
    print(f"selected step {int(steps[best])}, all-task IQM {all_task_iqm[best]:.1f}")
    for i, task in enumerate(TASKS):
        print(f"  {task:6s} {task_scores[best, i]:6.1f}  ({low[i]:.0f}-{high[i]:.0f})")
    print(f"  mean of the {len(TASKS)} task scores {task_scores[best].mean():.1f}")
    print(
        f"final step {int(steps[-1])}: all-task IQM {all_task_iqm[-1]:.1f}, "
        f"mean {task_scores[-1].mean():.1f}"
    )


if __name__ == "__main__":
    main()
