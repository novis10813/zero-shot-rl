"""Plots per-task eval IQM over training steps from a CSV written from the job log."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

parser = ArgumentParser()
parser.add_argument("csv", type=str)
parser.add_argument("--best_step", type=int, default=None)
parser.add_argument("-o", "--output", type=str, default="curves.png")
args = parser.parse_args()

df = pd.read_csv(args.csv)
panels = [
    ("stand", "stand"),
    ("walk", "walk"),
    ("run", "run"),
    ("flip", "flip"),
    ("mean", "mean of 4 tasks"),
]

fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
for ax, (column, title) in zip(axes, panels):
    ax.plot(df["step"] / 1e3, df[column], color="#2a78d6", linewidth=2)
    if args.best_step is not None:
        ax.axvline(args.best_step / 1e3, color="#52514e", linewidth=1, linestyle="--")
    ax.set_title(title, color="#0b0b0b")
    ax.set_xlabel("training step (k)", color="#52514e")
    ax.set_ylim(0, 1000)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e")
axes[0].set_ylabel("episode return (IQM of 10)", color="#52514e")
if args.best_step is not None:
    axes[-1].annotate(
        f"best checkpoint\n({args.best_step // 1000}k)",
        xy=(args.best_step / 1e3, 920),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=8,
        color="#52514e",
        va="top",
    )
fig.tight_layout()
fig.savefig(args.output, dpi=150, facecolor="#fcfcfb")
