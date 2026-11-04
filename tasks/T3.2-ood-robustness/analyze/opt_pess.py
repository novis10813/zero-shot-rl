"""Always-optimistic and always-pessimistic FB under the five dynamics configs.

Reads the per-episode CSVs of zero_shot_rl.eval.switch (run_opt_pess.sh), one per training
seed and config, and writes data/opt_pess.csv with one row per config: the mean and standard
deviation of the return over all (training seed, episode) pairs for each policy, and the
paired difference pess - opt with its standard error. Episodes with the same seed share the
start state, so the difference is taken within the pair.

Usage (repo root):
    PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.opt_pess
    PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.opt_pess --check

--check recomputes the table and fails if data/opt_pess.csv differs byte for byte.
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
FIELDS = ["config", "n", "opt_mean", "opt_std", "pess_mean", "pess_std", "diff_mean", "diff_se"]


def read_returns(path):
    out = {"opt": {}, "pess": {}}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["condition"]][int(r["episode"])] = float(r["return"])
    return out


def table(input_dir):
    rows = []
    for config in CONFIGS:
        opt, pess, diff = [], [], []
        for seed in SEEDS:
            ret = read_returns(input_dir / f"opt_pess_s{seed}_{config}.csv")
            if ret["opt"].keys() != ret["pess"].keys():
                raise ValueError(f"opt and pess episodes differ for seed {seed}, {config}")
            for e in sorted(ret["opt"]):
                opt.append(ret["opt"][e])
                pess.append(ret["pess"][e])
                diff.append(ret["pess"][e] - ret["opt"][e])
        rows.append({
            "config": config, "n": len(diff),
            "opt_mean": st.mean(opt), "opt_std": st.stdev(opt),
            "pess_mean": st.mean(pess), "pess_std": st.stdev(pess),
            "diff_mean": st.mean(diff), "diff_se": st.stdev(diff) / math.sqrt(len(diff)),
        })
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
    parser.add_argument("--out", type=Path, default=TASK / "data" / "opt_pess.csv")
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
