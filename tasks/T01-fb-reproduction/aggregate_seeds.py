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
parser.add_argument("--comparison_output", type=str, default=None)
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


def report(index: int, label: str) -> tuple:
    point, interval = rly.get_interval_estimates(
        {label: scores[:, index, :]},
        lambda x: np.array([iqm(x[:, t]) for t in range(len(TASKS))] + [iqm(x)]),
        reps=args.reps,
    )
    print(f"{label} (step {int(steps[index])})")
    for t, name in enumerate([*TASKS, "all tasks"]):
        low, high = interval[label][0][t], interval[label][1][t]
        print(f"  {name:9s} IQM {point[label][t]:6.1f}  ({low:.0f}-{high:.0f})")
    return point[label][:4], interval[label][0][:4], interval[label][1][:4]


np.random.seed(0)
best_estimates = report(best, "best step")
final_estimates = report(final, "final step")

if args.comparison_output is not None:
    # Paper Table 6 (Rnd-100k, Walker, FB): IQM and 95% interval per task.
    paper = (
        np.array([558, 184, 101, 163]),
        np.array([498, 123, 90, 90]),
        np.array([637, 278, 135, 212]),
    )
    groups = [
        (f"this reproduction, selected step ({best_step // 1000}k)", best_estimates, "#2a78d6"),
        ("this reproduction, final step (1M)", final_estimates, "#eb6834"),
        ("paper FB (Table 6)", paper, "#1baf7a"),
    ]
    x = np.arange(len(TASKS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for g, (label, (point, low, high), color) in enumerate(groups):
        offset = (g - 1) * (width + 0.01)
        ax.bar(x + offset, point, width, color=color, label=label, zorder=2)
        ax.errorbar(
            x + offset,
            point,
            yerr=[point - low, high - point],
            fmt="none",
            ecolor="#0b0b0b",
            elinewidth=1,
            capsize=3,
            zorder=3,
        )
        for xi, value, top in zip(x + offset, point, high):
            ax.text(xi, top + 12, f"{value:.0f}", ha="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(x, TASKS)
    ax.set_ylabel("episode return (IQM over 5 seeds)", color="#52514e")
    ax.set_ylim(0, 760)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.comparison_output, dpi=150, facecolor="#fcfcfb")

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
