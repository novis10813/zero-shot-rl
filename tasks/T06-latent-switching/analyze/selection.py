"""Post-training latent selection with a budget of online episodes, simulated from rollouts.

The candidates of one FB seed are the latents of a CQL angle sweep CSV (cql_sweep.py and
zero_shot_rl.eval.latents), one per (family, target_angle), each with logged episode
returns. Episodes 0-49 of a candidate form its selection pool, episodes 50-99 its
evaluation set, so no episode both picks and scores a candidate. A method spends a budget
of B episodes, each a return drawn with replacement from the pool of the candidate it
plays, and selects one candidate. Its value is the evaluation mean of that candidate.

    uniform   play the candidates in a random order, round robin, select the best
              sample mean among the played ones
    ucb       UCB1 on returns / 1000: play each candidate once (in a random order while
              B < K), then argmax mean + sqrt(2 ln t / n), select the best sample mean

Reference values per seed: z_r (the FB policy without selection, B = 0), random (mean
over candidates), oracle (best evaluation mean). It writes

    selection_candidates.csv   seed, candidate, pool_mean, eval_mean
    selection.csv              seed (or "mean" over seeds), method, budget, value (mean
                               over repeats), regret (oracle - value), p_best (fraction
                               of repeats that select the oracle candidate)
    selection_budget.png       value against budget, one panel per seed and the mean

Run from the repository root:
    PYTHONPATH=.:tasks/T06-latent-switching uv run python -m analyze.selection \
        --rollouts <seed0.csv> <seed1.csv> <seed2.csv> --out <dir> [--check]
"""

import argparse
import csv
import filecmp
from pathlib import Path
import tempfile

import numpy as np

POOL_EPISODES = 50
BUDGETS = (1, 5, 11, 22, 33, 55, 110, 220)
REPEATS = 2000
SCALE = 1000.0  # DMC returns lie in [0, 1000]


def load_candidates(path):
    """{candidate: (pool returns, eval returns)} of one rollout CSV."""
    returns = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            angle = row["target_angle"]
            name = row["family"] if angle == "nan" else f"{row['family']} {angle}"
            returns.setdefault(name, {})[int(row["episode"])] = float(row["return"])
    candidates = {}
    for name, episodes in returns.items():
        values = np.array([episodes[e] for e in sorted(episodes)])
        candidates[name] = (values[:POOL_EPISODES], values[POOL_EPISODES:])
    return candidates


def select(method, pools, budget, rng):
    """Index of the candidate a method selects after budget episodes."""
    k = len(pools)
    order = rng.permutation(k)
    sums, counts = np.zeros(k), np.zeros(k)

    def play(i):
        sums[i] += rng.choice(pools[i]) / SCALE
        counts[i] += 1

    for t in range(budget):
        if method == "uniform" or t < k:
            play(order[t % k])
        elif method == "ucb":
            play(np.argmax(sums / counts + np.sqrt(2 * np.log(t) / counts)))
        else:
            raise ValueError(method)
    played = counts > 0
    means = np.where(played, sums / np.maximum(counts, 1), -np.inf)
    return int(np.argmax(means))


def seed_rows(seed, candidates, rng):
    names = list(candidates)
    pools = [candidates[n][0] for n in names]
    values = np.array([candidates[n][1].mean() for n in names])
    oracle, best = values.max(), int(values.argmax())
    rows = [
        {"seed": seed, "method": "z_r", "budget": 0, "value": values[names.index("z_r 0")]},
        {"seed": seed, "method": "random", "budget": 0, "value": values.mean()},
        {"seed": seed, "method": "oracle", "budget": 0, "value": oracle},
    ]
    for row in rows:
        row["p_best"] = float(row["method"] == "oracle")
    for method in ("uniform", "ucb"):
        for budget in BUDGETS:
            picks = np.array([select(method, pools, budget, rng) for _ in range(REPEATS)])
            rows.append({"seed": seed, "method": method, "budget": budget,
                         "value": values[picks].mean(), "p_best": (picks == best).mean()})
    for row in rows:
        row["regret"] = oracle - row["value"]
    return rows


def mean_rows(rows):
    keys = dict.fromkeys((r["method"], r["budget"]) for r in rows)
    out = []
    for method, budget in keys:
        group = [r for r in rows if (r["method"], r["budget"]) == (method, budget)]
        out.append({"seed": "mean", "method": method, "budget": budget,
                    **{f: np.mean([r[f] for r in group]) for f in ("value", "p_best", "regret")}})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v
                             for k, v in row.items()})


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = list(dict.fromkeys(r["seed"] for r in rows))
    fig, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 3.4), sharey=True)
    for ax, seed in zip(axes, seeds):
        mine = [r for r in rows if r["seed"] == seed]
        ref = {r["method"]: r["value"] for r in mine if r["budget"] == 0}
        for method, style in (("uniform", "o-"), ("ucb", "s-")):
            pts = [(r["budget"], r["value"]) for r in mine if r["method"] == method]
            ax.plot(*zip(*pts), style, label=method, markersize=4)
        for method, color in (("oracle", "k"), ("z_r", "tab:red"), ("random", "tab:gray")):
            ax.axhline(ref[method], color=color, ls="--", lw=1, label=method)
        ax.set_xscale("log")
        ax.set_xlabel("online episodes")
        ax.set_title(f"FB seed {seed}" if seed != "mean" else "mean over seeds")
    axes[0].set_ylabel("walk return of selected latent")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(rollouts, out):
    rng = np.random.default_rng(0)
    candidate_rows, rows = [], []
    for seed, path in enumerate(rollouts):
        candidates = load_candidates(path)
        for name, (pool, evaluation) in candidates.items():
            candidate_rows.append({"seed": seed, "candidate": name,
                                   "pool_mean": pool.mean(), "eval_mean": evaluation.mean()})
        rows += seed_rows(seed, candidates, rng)
    rows += mean_rows(rows)
    write_csv(out / "selection_candidates.csv", candidate_rows)
    write_csv(out / "selection.csv", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", nargs="+", required=True, help="one CSV per seed, in order")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--figure", type=str, default=None)
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare with the CSVs in --out, write nothing")
    args = parser.parse_args()
    out = Path(args.out)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            run(args.rollouts, Path(tmp))
            for name in ("selection_candidates.csv", "selection.csv"):
                same = filecmp.cmp(Path(tmp) / name, out / name, shallow=False)
                print(f"{name}: {'identical' if same else 'DIFFERS'}")
                if not same:
                    raise SystemExit(1)
        return
    out.mkdir(parents=True, exist_ok=True)
    rows = run(args.rollouts, out)
    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        plot(rows, args.figure)


if __name__ == "__main__":
    main()
