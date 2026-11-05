"""Tables and figures of REPORT.md from the runs in data/inputs/.

    data/inputs/main/       zero_shot_rl.eval.online_inference, stock Walker, 10 episodes
    data/inputs/episodic/   the same with the -EP rules, 50 episodes
    data/inputs/dynamics/   the same in the four shifted Walkers of T3.2, <shift>.<part>_s<seed>.csv
    data/inputs/fixed_z/    zero_shot_rl.eval.fixed_z_returns

File names end in _s<seed>, the T04 FB seed. A run's score in an episode is its return; per
seed, scores are averaged over the 10 runs. Relative performance divides by the Oracle return
of the same seed and task (mean over all its episodes and runs, in the same environment).
Across seeds, per-task scores are means and all-task scores the IQM of the four task means
(analysis.table6_aggregate.aggregate); figures show the mean over tasks of the relative
performance with the min-max over seeds, as the paper does.

Writes data/*.csv and figures/*.png and prints the numbers of REPORT.md.

    PYTHONPATH=. uv run python tasks/T4.1-optimistic-task-inference/analyze.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.table6_aggregate import TASKS, aggregate  # noqa: E402

ROOT = Path(__file__).parent
INPUTS, DATA, FIGURES = ROOT / "data" / "inputs", ROOT / "data", ROOT / "figures"
SHIFTS = ["baseline", "friction_mild", "friction_strong", "actuator_mild", "actuator_strong"]
NAMES = {"oracle": "Oracle", "random": "Random", "ucb": "OpTI-BFM", "ts": "OpTI-BFM-TS"}
COLORS = {"Oracle": "k", "Random": "tab:gray", "OpTI-BFM": "tab:blue", "OpTI-BFM-TS": "tab:orange",
          "OpTI-BFM-EP": "tab:blue", "OpTI-BFM-TS-EP": "tab:orange", "RND": "tab:green"}


def load(pattern):
    frames = []
    for path in sorted(INPUTS.glob(pattern)):
        seed = int(re.search(r"_s(\d+)\.csv$", path.name).group(1))
        frames.append(pd.read_csv(path).assign(seed=seed, file=path.stem))
    return pd.concat(frames, ignore_index=True)


def label(row):
    name = NAMES[row["method"]]
    return name + ("-EP" if row["period"] > 1 else "")


def per_seed(runs):
    """Mean return over runs per (rule, task, seed, episode)."""
    runs = runs.assign(param=runs["param"].fillna(-1))
    return (runs.groupby(["method", "param", "period", "task", "seed", "episode"])["return"]
            .mean().reset_index())


def oracle_reference(scores):
    """(task, seed) -> mean Oracle return over episodes."""
    return scores[scores["method"] == "oracle"].groupby(["task", "seed"])["return"].mean()


def relative(scores, reference):
    ref = reference.reindex(pd.MultiIndex.from_frame(scores[["task", "seed"]])).to_numpy()
    return scores.assign(relative=scores["return"].to_numpy() / ref)


def select(scores):
    """Per method, the param with the highest return summed over episodes (mean over
    tasks and seeds), as in appendix F of the paper."""
    rows = []
    for (method, param), group in scores[scores["method"].isin(["ucb", "ts"])].groupby(
            ["method", "param"]):
        total = group.groupby(["task", "seed"])["return"].sum().mean()
        rows.append({"method": method, "param": param, "cumulative_return": total})
    table = pd.DataFrame(rows)
    best = table.loc[table.groupby("method")["cumulative_return"].idxmax()]
    return table, dict(zip(best["method"], best["param"]))


def keep(scores, chosen):
    is_chosen = scores.apply(
        lambda r: r["method"] not in chosen or np.isclose(r["param"], chosen[r["method"]]), axis=1)
    return scores[is_chosen].assign(name=lambda d: d.apply(label, axis=1))


def all_task_iqm(scores, value="return"):
    """name, episode -> IQM over tasks of the seed-mean (table6_aggregate.aggregate)."""
    out = {}
    for (name, episode), group in scores.groupby(["name", "episode"]):
        table = group.pivot(index="seed", columns="task", values=value)[TASKS].to_numpy()
        out[(name, episode)] = aggregate(table[:, None, :])[0]
    return pd.Series(out)


def curve(ax, scores, names, title):
    for name in names:
        g = scores[scores["name"] == name].groupby(["seed", "episode"])["relative"].mean()
        g = g.unstack("seed")
        x = g.index + 1
        ax.plot(x, g.mean(axis=1), color=COLORS[name], label=name,
                linestyle="--" if name.endswith("-EP") else "-")
        ax.fill_between(x, g.min(axis=1), g.max(axis=1), color=COLORS[name], alpha=0.2)
    ax.axhline(1.0, color="k", linewidth=1, label="Oracle")
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel("relative performance")


def main_experiment():
    scores = per_seed(load("main/*.csv"))
    reference = oracle_reference(scores)
    table, chosen = select(scores)
    table.to_csv(DATA / "hyperparameters.csv", index=False, float_format="%.6g")
    print("Cumulative return over 10 episodes (mean over tasks and seeds)")
    print(table.to_string(index=False, float_format="%.1f"))
    print("chosen:", chosen)

    main = relative(keep(scores, chosen), reference)
    main.to_csv(DATA / "main.csv", index=False, float_format="%.6g")
    names = ["Random", "OpTI-BFM", "OpTI-BFM-TS"]

    pivot = main.groupby(["name", "task", "episode"])["return"].mean().unstack("task")[TASKS]
    print("\nReturn per task, mean over seeds")
    for episode in (0, 1, 2, 4, 9):
        print(f"episode {episode + 1}")
        print(pivot.xs(episode, level="episode").round(0).to_string())
    iqm = all_task_iqm(main)
    print("\nAll-task IQM of the return")
    print(iqm.unstack(1)[[0, 1, 2, 4, 9]].round(0).to_string())

    fig, ax = plt.subplots(figsize=(5, 3.5))
    curve(ax, main, names, "Walker, stock dynamics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "main_relative.png", dpi=150)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, task in zip(axes, TASKS):
        for name in ["Oracle"] + names:
            g = main[(main["name"] == name) & (main["task"] == task)]
            g = g.groupby(["seed", "episode"])["return"].mean().unstack("seed")
            ax.plot(g.index + 1, g.mean(axis=1), color=COLORS[name], label=name)
            ax.fill_between(g.index + 1, g.min(axis=1), g.max(axis=1), color=COLORS[name],
                            alpha=0.2)
        ax.set_title(task)
        ax.set_xlabel("episode")
    axes[0].set_ylabel("episode return")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "main_per_task.png", dpi=150)
    return main, reference, chosen


def episodic(main, reference):
    path = INPUTS / "episodic"
    if not path.exists():
        return
    scores = relative(per_seed(load("episodic/*.csv")), reference)
    scores = scores.assign(name=scores.apply(label, axis=1))
    both = pd.concat([main[main["name"].isin(["OpTI-BFM", "OpTI-BFM-TS"])], scores])
    both.to_csv(DATA / "episodic.csv", index=False, float_format="%.6g")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, names in zip(axes, [["OpTI-BFM", "OpTI-BFM-EP"], ["OpTI-BFM-TS", "OpTI-BFM-TS-EP"]]):
        curve(ax, both, names, " vs ".join(names))
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "episodic.png", dpi=150)
    iqm = all_task_iqm(both)
    print("\nAll-task IQM of the return, per-step vs episodic")
    print(iqm.unstack(1)[[0, 1, 2, 4, 9, 19, 49]].round(0).to_string())


def data_efficiency():
    path = INPUTS / "fixed_z"
    if not path.exists():
        return
    runs = load("fixed_z/*.csv")
    oracle = runs[runs["source"] == "oracle"].groupby(["task", "seed"])["return"].mean()
    scores = (runs[runs["source"] != "oracle"]
              .groupby(["source", "n", "task", "seed"])["return"].mean().reset_index())
    ref = oracle.reindex(pd.MultiIndex.from_frame(scores[["task", "seed"]])).to_numpy()
    scores = scores.assign(relative=scores["return"].to_numpy() / ref)
    scores.to_csv(DATA / "data_efficiency.csv", index=False, float_format="%.6g")
    print("\nRelative performance of the least-squares z from the first n transitions"
          " (mean over tasks and seeds)")
    print(scores.groupby(["source", "n"])["relative"].mean().unstack("n").round(2).to_string())
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for source, g in scores.groupby("source"):
        g = g.groupby(["seed", "n"])["relative"].mean().unstack("seed")
        ax.plot(g.index, g.mean(axis=1), marker="o", color=COLORS[source], label=source)
        ax.fill_between(g.index, g.min(axis=1), g.max(axis=1), color=COLORS[source], alpha=0.2)
    ax.axhline(1.0, color="k", linewidth=1, label="Oracle")
    ax.set_xscale("log")
    ax.set_xlabel("transitions n")
    ax.set_ylabel("relative performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "data_efficiency.png", dpi=150)


def dynamics(main):
    path = INPUTS / "dynamics"
    if not path.exists():
        return
    runs = load("dynamics/*.csv")
    runs["shift"] = runs["file"].str.split(".").str[0]
    frames = [main.assign(shift="baseline")]
    for shift, group in runs.groupby("shift"):
        scores = per_seed(group)
        scores = relative(scores, oracle_reference(scores))
        frames.append(scores.assign(shift=shift, name=scores.apply(label, axis=1)))
    scores = pd.concat(frames, ignore_index=True)
    scores = scores[scores["name"].isin(["Oracle", "Random", "OpTI-BFM", "OpTI-BFM-TS"])]
    scores.to_csv(DATA / "dynamics.csv", index=False, float_format="%.6g")

    print("\nShifted dynamics: all-task IQM of the return in episodes 1 and 10, and mean of"
          " episodes 6-10")
    rows = []
    for shift in SHIFTS:
        s = scores[scores["shift"] == shift]
        iqm = all_task_iqm(s)
        late = all_task_iqm(s[s["episode"] >= 5].groupby(
            ["name", "task", "seed"])["return"].mean().reset_index().assign(episode=0))
        for name in ["Oracle", "Random", "OpTI-BFM", "OpTI-BFM-TS"]:
            rows.append({"shift": shift, "name": name, "episode_1": iqm[(name, 0)],
                         "episode_10": iqm[(name, 9)], "episodes_6_10": late[(name, 0)]})
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "dynamics_summary.csv", index=False, float_format="%.6g")
    print(table.to_string(index=False, float_format="%.0f"))

    fig, axes = plt.subplots(1, len(SHIFTS), figsize=(4 * len(SHIFTS), 3.5), sharey=True)
    for ax, shift in zip(axes, SHIFTS):
        curve(ax, scores[scores["shift"] == shift], ["Random", "OpTI-BFM", "OpTI-BFM-TS"],
              shift)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "dynamics.png", dpi=150)


def main():
    FIGURES.mkdir(exist_ok=True)
    scores, reference, _ = main_experiment()
    episodic(scores, reference)
    data_efficiency()
    dynamics(scores)


if __name__ == "__main__":
    main()
