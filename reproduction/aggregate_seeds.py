"""
Aggregates per-seed eval CSVs the way the paper's Table 6 does: pick the step at which the
all-task IQM over seeds is highest, then report each task's IQM over seeds at that step
with a 95% stratified-bootstrap interval (rliable).
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import library as rly
from scipy import stats

TASKS = ["stand", "walk", "run", "flip"]

parser = ArgumentParser()
parser.add_argument("csvs", nargs="+", type=str)
parser.add_argument("-o", "--output", type=str, default="curves_seeds.png")
parser.add_argument("--reps", type=int, default=10000)
args = parser.parse_args()

runs = [pd.read_csv(path).set_index("step") for path in args.csvs]
steps = runs[0].index
assert all(run.index.equals(steps) for run in runs), "runs must share eval steps"
scores = np.stack([run.loc[steps, TASKS].to_numpy() for run in runs])  # [seeds, steps, tasks]


def iqm(x: np.ndarray) -> float:
    return stats.trim_mean(x.reshape(-1), 0.25)


all_task_iqm = np.array([iqm(scores[:, i, :]) for i in range(len(steps))])
best = int(np.argmax(all_task_iqm))
best_step = int(steps[best])
final = len(steps) - 1


def report(index: int, label: str) -> None:
    point, interval = rly.get_interval_estimates(
        {label: scores[:, index, :]},
        lambda x: np.array([iqm(x[:, t]) for t in range(len(TASKS))] + [iqm(x)]),
        reps=args.reps,
    )
    print(f"{label} (step {int(steps[index])})")
    for t, name in enumerate([*TASKS, "all tasks"]):
        low, high = interval[label][0][t], interval[label][1][t]
        print(f"  {name:9s} IQM {point[label][t]:6.1f}  ({low:.0f}-{high:.0f})")


np.random.seed(0)
report(best, "best step")
report(final, "final step")

fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
for ax, t in zip(axes, range(5)):
    if t < 4:
        per_seed = scores[:, :, t]
        title = TASKS[t]
    else:
        per_seed = scores.mean(axis=2)
        title = "mean of 4 tasks"
    for seed_curve in per_seed:
        ax.plot(steps / 1e3, seed_curve, color="#c3c2b7", linewidth=1)
    ax.plot(steps / 1e3, per_seed.mean(axis=0), color="#2a78d6", linewidth=2)
    ax.axvline(best_step / 1e3, color="#52514e", linewidth=1, linestyle="--")
    ax.set_title(title, color="#0b0b0b")
    ax.set_xlabel("training step (k)", color="#52514e")
    ax.set_ylim(0, 1000)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e")
axes[0].set_ylabel("episode return (IQM of 10)", color="#52514e")
axes[-1].annotate(
    f"selected step\n({best_step // 1000}k)",
    xy=(best_step / 1e3, 920),
    xytext=(8, 0),
    textcoords="offset points",
    fontsize=8,
    color="#52514e",
    va="top",
)
fig.tight_layout()
fig.savefig(args.output, dpi=150, facecolor="#fcfcfb")
