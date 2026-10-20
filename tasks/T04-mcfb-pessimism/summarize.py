"""Tables and figure for T04 from the probe outputs in data/probe_<algo>_s<seed>.json.

Per model and task, on the same 1000 training states (paired):
    level      mean(Q_hat - G) for data actions (offset is not identified by FB, see README)
    d_ood      mean[(Q_hat(ood) - Q_hat(data)) - (G(ood) - G(data))]; < 0 means uniform
               actions are valued below data actions by more than their true deficit
    d_policy   mean[(Q_hat(pi) - Q_hat(data)) - (G(pi) - G(data))]; > 0 means the
               policy's own action is overestimated relative to the data action
    spearman   rank correlation of Q_hat and G over data actions
    d_visited  mean(Q_hat - G) on states the eval policy visits minus the same on
               training states (both with a = pi(s)); < 0 means more pessimistic off-data
Each is averaged over the four tasks per seed, then mean and std over seeds.
Table 0 reads the training eval curves in data/eval_<algo>_s<seed>.csv (extract_eval.py).

Usage: python tasks/T04-mcfb-pessimism/summarize.py [--fig <png>]
"""

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DATA = Path(__file__).parent / "data"
TASKS = ["stand", "walk", "run", "flip"]
ALGOS = {"fb": "FB", "vcfb": "VC-FB", "mcfb": "MC-FB", "dvcfb": "DVC-FB"}
METRICS = ["level", "d_ood", "d_policy", "spearman", "d_visited"]
GAP_SETS = ["prior", "backward", "task"]

parser = ArgumentParser()
parser.add_argument("--fig", type=str, default=None)
args = parser.parse_args()


def task_metrics(cal):
    q = {k: np.array(v["q_hat"]) for k, v in cal.items()}
    g = {k: np.array(v["mc_return"]) for k, v in cal.items()}
    return {
        "level": np.mean(q["data"] - g["data"]),
        "d_ood": np.mean((q["ood"] - q["data"]) - (g["ood"] - g["data"])),
        "d_policy": np.mean((q["policy"] - q["data"]) - (g["policy"] - g["data"])),
        "spearman": spearmanr(q["data"], g["data"])[0],
        "d_visited": np.mean(q["visited"] - g["visited"]) - np.mean(q["policy"] - g["policy"]),
        "true_ood_deficit": np.mean(g["ood"] - g["data"]),
        "pred_ood_deficit": np.mean(q["ood"] - q["data"]),
    }


per_seed = defaultdict(list)  # algo -> list of {metric: value averaged over tasks}
per_task = defaultdict(lambda: defaultdict(list))  # (algo, task) -> metric -> seeds
gaps = defaultdict(lambda: defaultdict(list))  # algo -> z set -> seeds of (gap, frac)
nn = defaultdict(list)
returns = defaultdict(list)  # algo -> seeds of (mean G, std G) over data actions, tasks averaged
for path in sorted(DATA.glob("probe_*.json")):
    algo = path.stem.split("_")[1]
    probe = json.loads(path.read_text())
    (model,) = probe["models"].values()
    rows = {t: task_metrics(model["calibration"][t]) for t in TASKS}
    for t in TASKS:
        for m, v in rows[t].items():
            per_task[(algo, t)][m].append(v)
    per_seed[algo].append({m: np.mean([rows[t][m] for t in TASKS]) for m in rows["stand"]})
    for name in GAP_SETS:
        keys = [f"task_{t}" for t in TASKS] if name == "task" else [name]
        gap = np.mean([model["action_gaps"][k]["q_data"] - model["action_gaps"][k]["q_uniform"] for k in keys])
        scale = np.mean([model["action_gaps"][k]["q_data_std"] for k in keys])
        frac = np.mean([model["action_gaps"][k]["frac_data_above_uniform"] for k in keys])
        gaps[algo][name].append((gap, gap / scale, frac))
    returns[algo].append(
        (np.mean([np.mean(model["calibration"][t]["data"]["mc_return"]) for t in TASKS]),
         np.mean([np.std(model["calibration"][t]["data"]["mc_return"]) for t in TASKS])))
    nn[algo].append(
        (np.median(probe["data_nn_distance"]),
         np.median(np.concatenate([model["calibration"][t]["visited"]["nn_distance"] for t in TASKS])))
    )


def fmt(values):
    return f"{np.mean(values):+.2f} ± {np.std(values):.2f}"


evals = defaultdict(list)  # algo -> seeds of (final mean, best mean, best step, final per task)
for path in sorted(DATA.glob("eval_*.csv")):
    curve = np.genfromtxt(path, delimiter=",", names=True)
    best = np.argmax(curve["mean"])
    evals[path.stem.split("_")[1]].append(
        (curve["mean"][-1], curve["mean"][best], curve["step"][best], [curve[t][-1] for t in TASKS]))

print("Table 0: eval return (IQM over episodes, mean of the four tasks), mean ± std over seeds\n")
print("| Algorithm | n | final (1M) | best during training | best step per seed | " + " | ".join(f"final {t}" for t in TASKS) + " |")
print("|---" * (5 + len(TASKS)) + "|")
for a in [a for a in ALGOS if a in evals]:
    v = evals[a]
    per_task_final = np.array([e[3] for e in v])
    print(f"| {ALGOS[a]} | {len(v)} | {np.mean([e[0] for e in v]):.0f} ± {np.std([e[0] for e in v]):.0f} "
          f"| {np.mean([e[1] for e in v]):.0f} ± {np.std([e[1] for e in v]):.0f} "
          f"| {', '.join(f'{int(e[2]) // 1000}k' for e in v)} | "
          + " | ".join(f"{per_task_final[:, i].mean():.0f}" for i in range(len(TASKS))) + " |")
print()

algos = [a for a in ALGOS if a in per_seed]
print("Table 1: calibration metrics, reward units, mean ± std over seeds (tasks averaged)\n")
print("| Metric | " + " | ".join(f"{ALGOS[a]} (n={len(per_seed[a])})" for a in algos) + " |")
print("|---" * (len(algos) + 1) + "|")
for m in METRICS + ["true_ood_deficit", "pred_ood_deficit"]:
    print(f"| {m} | " + " | ".join(fmt([s[m] for s in per_seed[a]]) for a in algos) + " |")

print("\nTable 2: d_ood per task, mean ± std over seeds\n")
print("| Task | " + " | ".join(ALGOS[a] for a in algos) + " |")
print("|---" * (len(algos) + 1) + "|")
for t in TASKS:
    print(f"| {t} | " + " | ".join(fmt(per_task[(a, t)]["d_ood"]) for a in algos) + " |")

print("\nTable 3: data-action minus uniform-action value, raw F^T z units / in std of Q(s, a_data) / P(Q_data > Q_uniform)\n")
print("| z | " + " | ".join(ALGOS[a] for a in algos) + " |")
print("|---" * (len(algos) + 1) + "|")
for name in GAP_SETS:
    cells = []
    for a in algos:
        v = np.array(gaps[a][name])
        cells.append(f"{v[:, 0].mean():+.1f} / {v[:, 1].mean():+.2f} / {v[:, 2].mean():.2f}")
    print(f"| {name} | " + " | ".join(cells) + " |")

print("\nMedian distance to the nearest training state (standardised obs), training probe states vs eval-visited states:")
for a in algos:
    v = np.array(nn[a])
    print(f"  {ALGOS[a]}: training {v[:, 0].mean():.2f}, visited {v[:, 1].mean():.2f}")

print("\nScale of G on data actions (reward units, tasks averaged): mean, std over states")
for a in algos:
    v = np.array(returns[a])
    print(f"  {ALGOS[a]}: mean {v[:, 0].mean():.1f}, std {v[:, 1].mean():.1f}")

if args.fig:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [("d_ood", "OOD-action bias\n(Q̂ − G)(ood) − (Q̂ − G)(data)"),
             ("d_policy", "policy-action bias\n(Q̂ − G)(π) − (Q̂ − G)(data)"),
             ("d_visited", "visited-state bias\n(Q̂ − G)(visited) − (Q̂ − G)(train)")]
    fig, axes = plt.subplots(1, len(shown), figsize=(4 * len(shown), 3.2))
    for ax, (m, title) in zip(axes, shown):
        for i, a in enumerate(algos):
            v = [s[m] for s in per_seed[a]]
            ax.bar(i, np.mean(v), color=f"C{i}", alpha=0.6)
            ax.scatter([i] * len(v), v, color="k", s=12, zorder=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(algos)), [ALGOS[a] for a in algos])
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("reward units (Σ γ^t r)")
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=150)
    print(f"\nwrote {args.fig}")
