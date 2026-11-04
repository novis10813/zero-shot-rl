"""Four policies under the five dynamics configs: always opt, always pess, random, Q-guided.

Reads the per-episode CSVs of zero_shot_rl.eval.switch (run_switch.sh), one per training seed
and config, and writes data/switch.csv with one row per config: the mean and standard
deviation of the return over all (training seed, episode) pairs for each policy, the share of
steps on which Q-guided switching takes the pessimistic action (mean, standard deviation, and
the paired difference to the clean config with its standard error), and the paired difference of
Q-guided switching to random switching and to the better fixed policy (by mean return in that
config) with standard errors. Episodes with the same seed share the start state.

Usage (repo root):
    PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.switch
    PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.switch --check

--check recomputes the table and fails if data/switch.csv differs byte for byte.
"""

import csv
import io
import math
import statistics as st
from argparse import ArgumentParser
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
CONFIGS = ["baseline", "friction_mild", "friction_strong", "actuator_mild", "actuator_strong"]
SEEDS = [0, 1, 2]
POLICIES = ["opt", "pess", "random", "switch"]
FIELDS = (
    ["config", "n"]
    + [f"{p}_{s}" for p in POLICIES for s in ("mean", "std")]
    + ["switch_pess_frac", "switch_pess_frac_std", "switch_pess_frac_minus_clean",
       "switch_pess_frac_minus_clean_se", "best_fixed", "q_minus_random", "q_minus_random_se",
       "q_minus_best", "q_minus_best_se"]
)


def read(path):
    out = {p: {} for p in POLICIES}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["condition"]][int(r["episode"])] = (float(r["return"]), float(r["pess_frac"]))
    return out


def paired(a, b):
    diff = [x - y for x, y in zip(a, b)]
    return st.mean(diff), st.stdev(diff) / math.sqrt(len(diff))


def table(input_dir):
    rows = []
    clean_frac = None
    for config in CONFIGS:
        ret = {p: [] for p in POLICIES}
        frac = []
        for seed in SEEDS:
            data = read(input_dir / f"switch_s{seed}_{config}.csv")
            episodes = sorted(data["opt"])
            for p in POLICIES:
                if sorted(data[p]) != episodes:
                    raise ValueError(f"episodes of {p} differ for seed {seed}, {config}")
                ret[p] += [data[p][e][0] for e in episodes]
            frac += [data["switch"][e][1] for e in episodes]
        best = max(("opt", "pess"), key=lambda p: st.mean(ret[p]))
        q_rand, q_rand_se = paired(ret["switch"], ret["random"])
        q_best, q_best_se = paired(ret["switch"], ret[best])
        if clean_frac is None:
            clean_frac = frac
        paired_frac = [a - b for a, b in zip(frac, clean_frac)]
        row = {"config": config, "n": len(frac), "switch_pess_frac": st.mean(frac),
               "switch_pess_frac_std": st.stdev(frac),
               "switch_pess_frac_minus_clean": st.mean(paired_frac),
               "switch_pess_frac_minus_clean_se": (st.stdev(paired_frac) / math.sqrt(len(frac))
                                                   if config != "baseline" else 0.0),
               "best_fixed": best, "q_minus_random": q_rand, "q_minus_random_se": q_rand_se,
               "q_minus_best": q_best, "q_minus_best_se": q_best_se}
        for p in POLICIES:
            row[f"{p}_mean"], row[f"{p}_std"] = st.mean(ret[p]), st.stdev(ret[p])
        rows.append(row)
    return rows


def render(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    return buf.getvalue()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=TASK / "data" / "inputs")
    parser.add_argument("--out", type=Path, default=TASK / "data" / "switch.csv")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render(table(args.input_dir))
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            raise SystemExit(f"Stale or missing output: {args.out}")
        print("ok")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
