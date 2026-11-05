"""
Every figure of this task, from the CSVs in data/.

    first_action.png                 action at the first state, one bar per checkpoint at
                                     each joint, one panel per task (first_action.csv)
    behavior_distance_<metric>.png   distance of first-step actions to the data: FB seed k
                                     on x, VC-FB seed k on y, one dot per env seed; below the
                                     diagonal the VC-FB action is closer (behavior_distance.csv)
    rollout_<pool>_states_<metric>.png  the same at the states visited every 10 steps by the
                                     <pool> seed k episode (rollout_distances.csv)
    return_vs_distance_<metric>.png  return difference against distance difference (VC-FB
                                     minus FB), one dot per (env seed, seed pair)
                                     (behavior_distance.csv, episode_returns.csv)
    reward_h1_<pool>_states_<metric>.png  one-step reward difference against distance
                                     difference, one dot per state (reward_h1.csv)
    action_difference.png            VC-FB action minus FB action at dataset states, one panel
                                     per (task, seed pair) (action_difference.csv)

Spearman correlations of the difference plots are printed.

Usage (repo root):
    uv run python tasks/T3.3.1-action-evaluation/plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
TASKS = ["stand", "walk", "run", "flip"]
METRICS = ["joint", "conditional"]
POOLS = ["FB", "VC-FB"]


def first_action(df, path):
    """One bar per (algorithm, checkpoint) at each joint; one colour family per algorithm."""
    joints = list(dict.fromkeys(df["joint"]))
    algorithms = list(dict.fromkeys(df["algorithm"]))
    groups = [(a, c) for a in algorithms for c in sorted(df[df["algorithm"] == a]["checkpoint"].unique())]
    cmaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples]
    colours = {}
    for k, algorithm in enumerate(algorithms):
        members = [g for g in groups if g[0] == algorithm]
        shades = np.linspace(0.45, 0.9, len(members))
        colours.update({g: cmaps[k % len(cmaps)](v) for g, v in zip(members, shades)})
    width = 0.9 / len(groups)
    x = np.arange(len(joints))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.ravel()
    for ax, task in zip(axes, TASKS):
        for k, (algorithm, checkpoint) in enumerate(groups):
            sub = df[(df["task"] == task) & (df["algorithm"] == algorithm) & (df["checkpoint"] == checkpoint)]
            values = [sub[sub["joint"] == j]["action"].item() for j in joints]
            offset = x + (k - (len(groups) - 1) / 2) * width
            ax.bar(offset, values, width, color=colours[(algorithm, checkpoint)],
                   label=f"{algorithm} seed {checkpoint}")
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels(joints, rotation=45, ha="right")
        ax.axhline(0, color="gray", linewidth=0.5)
    for ax in axes[::2]:
        ax.set_ylabel("action")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def paired_scatter(pairs, metric, path, note=None, title=None):
    """pairs: task -> [(seed k, FB distances, VC-FB distances at the same states)]."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, task in zip(axes.ravel(), TASKS):
        colours = plt.cm.tab10(np.arange(len(pairs[task])))
        deltas = []
        top = 0.0
        for colour, (k, fb, vc) in zip(colours, pairs[task]):
            ax.scatter(fb, vc, s=14, color=colour, label=f"seed {k}")
            deltas.append((vc - fb).to_numpy())
            top = max(top, fb.max(), vc.max())
        deltas = np.concatenate(deltas)
        ax.plot([0, top], [0, top], color="gray", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel(f"FB {metric} distance")
        ax.set_ylabel(f"VC-FB {metric} distance")
        text = f"VC-FB closer: {(deltas < 0).mean():.0%}\nmedian VC-FB minus FB: {np.median(deltas):+.3f}"
        if note:
            text += "\n" + note(task)
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", fontsize=8)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def difference_scatter(points, xlabel, ylabel, path, title=None, prefix=""):
    """points: task -> [(seed k, x, y)]; prints and shows the Spearman correlation per task."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, task in zip(axes.ravel(), TASKS):
        colours = plt.cm.tab10(np.arange(len(points[task])))
        for colour, (k, x, y) in zip(colours, points[task]):
            ax.scatter(x, y, s=14, color=colour, label=f"seed {k}")
        xs = np.concatenate([x for _, x, _ in points[task]])
        ys = np.concatenate([y for _, _, y in points[task]])
        rho, p = stats.spearmanr(xs, ys)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(0.03, 0.97, f"Spearman {rho:+.2f} (p={p:.2f}), n={len(xs)}",
                transform=ax.transAxes, va="top", fontsize=8)
        print(f"{prefix}{task:5s} rho {rho:+.2f} p {p:.3f} n {len(xs)}")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def pool_seeds(df, pool):
    """Seed indices k of the episodes of `pool` in a CSV with "pool" = "<algorithm> seed <k>"."""
    names = df["pool"].unique()
    return sorted(int(n.rsplit(" ", 1)[1]) for n in names if n.rsplit(" seed ", 1)[0] == pool)


def behavior_distance(df):
    seeds = sorted(df[df["algorithm"] == "FB"]["checkpoint"].unique())
    for metric in METRICS:
        def note(task, metric=metric):
            sub = df[df["task"] == task]
            uniform = sub[sub["algorithm"] == "uniform"][metric].median()
            nn_action = sub[sub["algorithm"] == "nn_action"][metric].median()
            return f"median uniform: {uniform:.2f}, nn_action: {nn_action:.2f}"

        pairs = {}
        for task in TASKS:
            sub = df[df["task"] == task]
            pairs[task] = []
            for k in seeds:
                fb = sub[(sub["algorithm"] == "FB") & (sub["checkpoint"] == k)].set_index("env_seed")[metric]
                vc = sub[(sub["algorithm"] == "VC-FB") & (sub["checkpoint"] == k)].set_index("env_seed")[metric]
                pairs[task].append((k, fb, vc.loc[fb.index]))
        paired_scatter(pairs, metric, FIGURES / f"behavior_distance_{metric}.png", note=note)


def rollout(df):
    for metric in METRICS:
        for pool in POOLS:
            pairs = {}
            for task in TASKS:
                pairs[task] = []
                for k in pool_seeds(df, pool):
                    sub = df[(df["task"] == task) & (df["pool"] == f"{pool} seed {k}")]
                    fb = sub[sub["actor"] == f"FB seed {k}"].set_index("step")[metric]
                    vc = sub[sub["actor"] == f"VC-FB seed {k}"].set_index("step")[metric].loc[fb.index]
                    pairs[task].append((k, fb, vc))
            paired_scatter(pairs, metric, FIGURES / f"rollout_{pool}_states_{metric}.png",
                           title=f"states visited by {pool}")


def return_vs_distance(distance, returns):
    """Per (task, env seed, seed pair k): x = D(VC-FB seed k) - D(FB seed k) at the initial
    state, y = return(VC-FB seed k) - return(FB seed k) of the episode from that state."""
    r = returns.pivot_table(index=["task", "env_seed", "checkpoint"], columns="algorithm", values="return")
    for metric in METRICS:
        d = distance[distance["algorithm"].isin(["FB", "VC-FB"])]
        d = d.pivot_table(index=["task", "env_seed", "checkpoint"], columns="algorithm", values=metric)
        df = pd.DataFrame({"dx": d["VC-FB"] - d["FB"], "dy": r["VC-FB"] - r["FB"]}).dropna().reset_index()
        points = {}
        for task in TASKS:
            sub = df[df["task"] == task]
            points[task] = [
                (k, s["dx"].to_numpy(), s["dy"].to_numpy())
                for k, s in ((k, sub[sub["checkpoint"] == k]) for k in sorted(sub["checkpoint"].unique()))
            ]
        difference_scatter(points, f"{metric} distance: VC-FB minus FB", "return: VC-FB minus FB",
                           FIGURES / f"return_vs_distance_{metric}.png", prefix=f"{metric:11s} ")


def reward(df, name, horizon):
    for metric in METRICS:
        for pool in POOLS:
            points = {}
            for task in TASKS:
                points[task] = []
                for k in pool_seeds(df, pool):
                    sub = df[(df["task"] == task) & (df["pool"] == f"{pool} seed {k}")]
                    fb = sub[sub["actor"] == f"FB seed {k}"].set_index("step")
                    vc = sub[sub["actor"] == f"VC-FB seed {k}"].set_index("step").loc[fb.index]
                    points[task].append((k, (vc[metric] - fb[metric]).to_numpy(),
                                         (vc["reward"] - fb["reward"]).to_numpy()))
            difference_scatter(points, f"{metric} distance: VC-FB minus FB",
                               f"reward over {horizon} step(s): VC-FB minus FB",
                               FIGURES / f"{name}_{pool}_states_{metric}.png",
                               title=f"states visited by {pool}",
                               prefix=f"{pool} states {metric:11s} ")


def action_difference(df, path):
    joints = list(dict.fromkeys(df["actuator"]))
    pairs = sorted(df["pair"].unique())
    fig, axes = plt.subplots(len(TASKS), len(pairs), figsize=(3.4 * len(pairs), 3.6 * len(TASKS)),
                             squeeze=False)
    image = None
    for r, task in enumerate(TASKS):
        for c, pair in enumerate(pairs):
            sub = df[(df["task"] == task) & (df["pair"] == pair)]
            delta = sub.pivot(index="state", columns="actuator", values="delta")[joints]
            ax = axes[r, c]
            image = ax.imshow(delta.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2,
                              interpolation="nearest")
            ax.set_title(f"{task}, seed pair {pair}", fontsize=9)
            ax.set_xticks(range(len(joints)))
            ax.set_xticklabels(joints if r == len(TASKS) - 1 else [], rotation=60, ha="right",
                               fontsize=8)
            ax.set_ylabel("dataset state" if c == 0 else "")
    fig.colorbar(image, ax=axes, shrink=0.6, label="VC-FB action minus FB action")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    distance = pd.read_csv(DATA / "behavior_distance.csv")
    first_action(pd.read_csv(DATA / "first_action.csv"), FIGURES / "first_action.png")
    behavior_distance(distance)
    rollout(pd.read_csv(DATA / "rollout_distances.csv"))
    return_vs_distance(distance, pd.read_csv(DATA / "episode_returns.csv"))
    reward(pd.read_csv(DATA / "reward_h1.csv"), "reward_h1", horizon=1)
    action_difference(pd.read_csv(DATA / "action_difference.csv"), FIGURES / "action_difference.png")


if __name__ == "__main__":
    main()
