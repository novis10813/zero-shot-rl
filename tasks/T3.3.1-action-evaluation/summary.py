"""
The tables of REPORT.md, from the CSVs in data/ (the Spearman correlations come from plot.py).

Usage (repo root):
    uv run python tasks/T3.3.1-action-evaluation/summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
TASKS = ["stand", "walk", "run", "flip"]
METRICS = ["joint", "conditional"]
POOLS = ["FB", "VC-FB"]


def first_step_distance():
    df = pd.read_csv(DATA / "behavior_distance.csv")
    medians = df.groupby(["task", "algorithm"])[METRICS].median().unstack()
    print("Median distance at the 50 reset states")
    print(medians.reindex(TASKS).round(3))
    for metric in METRICS:
        pivot = df[df["algorithm"].isin(POOLS)].pivot_table(
            index=["task", "env_seed", "checkpoint"], columns="algorithm", values=metric
        )
        delta = (pivot["VC-FB"] - pivot["FB"]).groupby("task")
        print(f"{metric}: VC-FB closer", delta.apply(lambda x: (x < 0).mean()).reindex(TASKS).round(2).to_dict())
    print()


def rollout_distance():
    df = pd.read_csv(DATA / "rollout_distances.csv")
    own = df[df["pool"] == df["actor"]]
    own = own.assign(algorithm=own["actor"].str.rsplit(" seed", n=1).str[0])
    print("Mean distance at each checkpoint's own visited states")
    print(own.groupby(["task", "algorithm"])[METRICS].mean().unstack().reindex(TASKS).round(3))
    for pool in POOLS:
        for metric in METRICS:
            closer = {}
            for task in TASKS:
                deltas = []
                for k in range(3):
                    sub = df[(df["task"] == task) & (df["pool"] == f"{pool} seed {k}")]
                    fb = sub[sub["actor"] == f"FB seed {k}"].set_index("step")[metric]
                    vc = sub[sub["actor"] == f"VC-FB seed {k}"].set_index("step")[metric].loc[fb.index]
                    deltas.append(vc - fb)
                closer[task] = round(float((pd.concat(deltas) < 0).mean()), 2)
            print(f"states of {pool}, {metric}: VC-FB closer {closer}")
    print()


def returns():
    df = pd.read_csv(DATA / "episode_returns.csv")
    pivot = df.pivot_table(index=["task", "checkpoint"], columns="algorithm", values="return")
    pivot["VC-FB minus FB"] = pivot["VC-FB"] - pivot["FB"]
    print("Mean return over the 50 reset states, per seed pair")
    print(pivot.reindex(TASKS, level=0).round(1))
    print("mean over seed pairs")
    print(pivot.groupby("task").mean().reindex(TASKS).round(1))
    print()


def actions():
    df = pd.read_csv(DATA / "action_difference.csv")
    print("Actions at 200 dataset states")
    print("mean |VC-FB minus FB| per task", df["delta"].abs().groupby(df["task"]).mean().reindex(TASKS).round(2).to_dict())
    print("opposite sign", round(float((np.sign(df["fb"]) != np.sign(df["vc"])).mean()), 2))
    majority = df.groupby(["task", "pair", "actuator"])["delta"].agg(
        lambda x: max((x > 0).mean(), (x < 0).mean())
    )
    print("largest share of one sign of the difference in a (task, pair, actuator)",
          round(float(majority.max()), 2))
    for column, name in [("fb", "FB"), ("vc", "VC-FB")]:
        saturated = df[column].abs() > 0.99
        print(f"{name}: |a| > 0.99 {saturated.mean():.2f}, per seed pair",
              saturated.groupby(df["pair"]).mean().round(2).to_dict())
    first = pd.read_csv(DATA / "first_action.csv")
    print("first state: |a| > 0.99",
          (first["action"].abs() > 0.99).groupby(first["algorithm"]).mean().round(2).to_dict())


def main():
    first_step_distance()
    rollout_distance()
    returns()
    actions()


if __name__ == "__main__":
    main()
