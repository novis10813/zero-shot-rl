"""Stage 0 analysis of the probe NPZs written by probe.py, without new evaluation.

FB misestimates the value of a constant reward: c_k = F_k^T u_1 is near -1/(1-gamma)
instead of 1/(1-gamma). Since z_hat_r = r_mean u_1 + E_D[(r - r_mean) B], the value of r
with the constant part valued exactly is

    q_k^corr = q_k + r_mean (1/(1-gamma) - c_k),    r_mean = mean reward of the task.

With q_hat = min(q1, q2) the agent's value of r, over = q_hat - G, and q_hat^corr and
over^corr likewise from the corrected heads, it writes

    stage0_signals.csv   Spearman correlation of each trust signal with over^corr
                         (rho_<signal>) and with over (rho_raw_<signal>), per seed, task
                         and source, and over all sources pooled ("all")
    stage0_interval.csv  fraction of rows with q_hat^corr + p <= G (lower),
                         G <= q_hat^corr (upper) and both, p = (p1 + p2) / 2 the
                         uncorrected implicit CQL reward value
    stage0_switch.csv    on paired states (policy/pess, visited/visited_pess): mean G when
                         the fraction f of states with the largest signal at the
                         optimistic action takes the pessimistic action instead

Signals at an (s, a):

    head_gap   |q1 - q2|
    penalty    -p, the implicit CQL penalty
    const_dev  |(c1 + c2) / 2 - 1 / (1 - gamma)|
    nn         nearest training (s, a)

Run from the repository root:
    PYTHONPATH=.:tasks/T06-latent-switching uv run python -m analyze.stage0 \
        --probes <probe_seed0.npz> <probe_seed1.npz> <probe_seed2.npz> --out <dir>
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DISCOUNT = 0.98
SIGNALS = ("head_gap", "penalty", "const_dev", "nn")
PAIRS = (("policy", "pess"), ("visited", "visited_pess"))
FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def signals(probe):
    """Trust signals of every probe row."""
    return {
        "head_gap": np.abs(probe["q1"] - probe["q2"]),
        "penalty": -(probe["p1"] + probe["p2"]) / 2,
        "const_dev": np.abs((probe["c1"] + probe["c2"]) / 2 - 1 / (1 - DISCOUNT)),
        "nn": probe["nn"],
    }


def q_hat(probe, corrected=True):
    """min(q1, q2), with the constant part of r valued exactly if corrected."""
    if not corrected:
        return np.minimum(probe["q1"], probe["q2"])
    r_mean = dict(zip(probe["latent_task"], probe["latent_r_mean"]))
    r_mean = np.array([r_mean[task] for task in probe["task"]])
    return np.minimum(*(probe[f"q{k}"] + r_mean * (1 / (1 - DISCOUNT) - probe[f"c{k}"])
                        for k in (1, 2)))


def overestimation(probe, corrected=True):
    return q_hat(probe, corrected) - probe["G"]


def signal_rows(seed, probe):
    over, over_raw, u = overestimation(probe), overestimation(probe, False), signals(probe)
    rows = []
    for task in dict.fromkeys(probe["task"]):
        in_task = probe["task"] == task
        for source in [*dict.fromkeys(probe["source"]), "all"]:
            mask = in_task if source == "all" else in_task & (probe["source"] == source)
            row = {"seed": seed, "task": task, "source": source, "n": int(mask.sum()),
                   "over_mean": over[mask].mean(), "over_raw_mean": over_raw[mask].mean(),
                   "G_mean": probe["G"][mask].mean()}
            for name in SIGNALS:
                row[f"rho_{name}"] = spearmanr(u[name][mask], over[mask]).statistic
            for name in SIGNALS:
                row[f"rho_raw_{name}"] = spearmanr(u[name][mask], over_raw[mask]).statistic
            rows.append(row)
    return rows


def interval_rows(seed, probe):
    q_corr = q_hat(probe)
    lower = q_corr + (probe["p1"] + probe["p2"]) / 2
    g = probe["G"]
    rows = []
    for task in dict.fromkeys(probe["task"]):
        for source in dict.fromkeys(probe["source"]):
            mask = (probe["task"] == task) & (probe["source"] == source)
            below, above = lower[mask] <= g[mask], g[mask] <= q_corr[mask]
            rows.append({"seed": seed, "task": task, "source": source, "n": int(mask.sum()),
                         "frac_lower": below.mean(), "frac_upper": above.mean(),
                         "frac_inside": (below & above).mean()})
    return rows


def paired(probe, task, opt, pess):
    """Rows of the optimistic and pessimistic source at the same states, in state order."""
    rows = []
    for source in (opt, pess):
        idx = np.flatnonzero((probe["task"] == task) & (probe["source"] == source))
        rows.append(idx[np.argsort(probe["state"][idx], kind="stable")])
    if not np.array_equal(probe["state"][rows[0]], probe["state"][rows[1]]):
        raise ValueError(f"{task}: {opt} and {pess} cover different states")
    return rows


def switched_return(g_opt, g_pess, u, fraction):
    """Mean G when the ceil(fraction * n) states with the largest u switch to pess."""
    k = int(np.ceil(fraction * len(u) - 1e-9))
    switch = np.zeros(len(u), dtype=bool)
    switch[np.argsort(-u, kind="stable")[:k]] = True
    return np.where(switch, g_pess, g_opt).mean()


def switch_rows(seed, probe):
    u = signals(probe)
    rows = []
    for task in dict.fromkeys(probe["task"]):
        for opt, pess in PAIRS:
            i_opt, i_pess = paired(probe, task, opt, pess)
            g_opt, g_pess = probe["G"][i_opt], probe["G"][i_pess]
            for fraction in FRACTIONS:
                row = {"seed": seed, "task": task, "states": opt, "n": len(i_opt),
                       "fraction": fraction,
                       "random": (1 - fraction) * g_opt.mean() + fraction * g_pess.mean(),
                       "oracle": switched_return(g_opt, g_pess, g_pess - g_opt, fraction)}
                for name in SIGNALS:
                    row[name] = switched_return(g_opt, g_pess, u[name][i_opt], fraction)
                rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", nargs="+", required=True, help="one NPZ per seed, in order")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"stage0_signals.csv": [], "stage0_interval.csv": [], "stage0_switch.csv": []}
    for seed, path in enumerate(args.probes):
        probe = dict(np.load(path))
        tables["stage0_signals.csv"] += signal_rows(seed, probe)
        tables["stage0_interval.csv"] += interval_rows(seed, probe)
        tables["stage0_switch.csv"] += switch_rows(seed, probe)
    for name, rows in tables.items():
        write_csv(out / name, rows)
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
