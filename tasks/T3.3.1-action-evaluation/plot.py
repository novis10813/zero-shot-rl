"""
Return difference against distance difference, one dot per (task, env seed, seed pair k).

x = D(VC-FB seed k) - D(FB seed k) at the initial state of env seed e (behavior_distance.csv),
y = return(VC-FB seed k) - return(FB seed k) of the episode from the same state
(episode_returns.csv). Spearman correlation over all dots of a task is printed in the panel.

Usage (repo root):
    uv run python tasks/T3.3.1-action-evaluation/plot_return_vs_distance.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
TASKS = ["stand", "walk", "run", "flip"]


def pairs(distance, returns, metric):
    d = distance[distance["algorithm"].isin(["FB", "VC-FB"])]
    d = d.pivot_table(index=["task", "env_seed", "checkpoint"], columns="algorithm", values=metric)
    r = returns.pivot_table(index=["task", "env_seed", "checkpoint"], columns="algorithm", values="return")
    out = pd.DataFrame({"dx": d["VC-FB"] - d["FB"], "dy": r["VC-FB"] - r["FB"]}).dropna()
    return out.reset_index()


def main():
    distance = pd.read_csv(HERE / "data" / "behavior_distance.csv")
    returns = pd.read_csv(HERE / "data" / "episode_returns.csv")
    colours = plt.cm.tab10(np.arange(3))
    for metric in ["joint", "conditional"]:
        df = pairs(distance, returns, metric)
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax, task in zip(axes.ravel(), TASKS):
            sub = df[df["task"] == task]
            for k, colour in enumerate(colours):
                s = sub[sub["checkpoint"] == k]
                ax.scatter(s["dx"], s["dy"], s=14, color=colour, label=f"seed {k}")
            rho, p = stats.spearmanr(sub["dx"], sub["dy"])
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_title(task)
            ax.set_xlabel(f"{metric} distance: VC-FB minus FB")
            ax.set_ylabel("return: VC-FB minus FB")
            ax.text(0.03, 0.97, f"Spearman {rho:+.2f} (p={p:.2f}), n={len(sub)}",
                    transform=ax.transAxes, va="top", fontsize=8)
            print(f"{metric:11s} {task:5s} rho {rho:+.2f} p {p:.3f} n {len(sub)}")
        axes[0, 0].legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(HERE / "figures" / f"return_vs_distance_{metric}.png", dpi=150)


if __name__ == "__main__":
    main()
