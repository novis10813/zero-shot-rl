"""Walk return against action deviation of the policies of a fixed FB, per seed.

It joins the deviations of deviation.py with the logged walk returns of the same
policies:
    candidate   every latent of the CQL angle sweep (rollouts CSV of
                zero_shot_rl.eval.latents, 100 episodes)
    switch      rule C at each kappa (switch_eval.py kappa sweep CSV, 100 episodes)
    random      a_pess with probability 0.5 at every step (switch_eval.py --random_ps
                CSV, 100 episodes)
and writes

    deviation.csv              seed, kind, name, kappa, p, pess_frac, dev_data (mean
                               ||a - a_D||^2 on dataset states), dev_opt (mean
                               ||a - pi(s, z_r)||^2), return (mean), return_se, episodes
    return_vs_deviation.png    return against dev_data (top) and dev_opt (bottom), one
                               column per seed

Run from the repository root:
    PYTHONPATH=.:tasks/T06-latent-switching uv run python -m analyze.deviation \
        --deviation <dev0.csv> <dev1.csv> <dev2.csv> --rollouts <r0.csv> <r1.csv> <r2.csv> \
        --switch <s0.csv> <s1.csv> <s2.csv> --random <p0.csv> <p1.csv> <p2.csv> \
        --out <deviation.csv> [--figure <png>] [--check]
"""

import argparse
import csv
import filecmp
from pathlib import Path
import tempfile

import numpy as np

FIELDS = ["seed", "kind", "name", "kappa", "p", "pess_frac", "dev_data", "dev_opt",
          "return", "return_se", "episodes"]
RANDOM_P = "0.5"


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def candidate_returns(path):
    """{candidate name: returns}, names as in analyze.selection."""
    out = {}
    for row in read(path):
        angle = row["target_angle"]
        name = row["family"] if angle == "nan" else f"{row['family']} {angle}"
        out.setdefault(name, []).append(float(row["return"]))
    return out


def switch_returns(switch_path, random_path):
    """{name in deviation.py: returns} of rule C kappas and random p = 0.5."""
    out = {}
    for row in read(switch_path):
        if row["condition"] == "switch":
            out.setdefault(f"switch {float(row['kappa']):g}", []).append(float(row["return"]))
    for row in read(random_path):
        if row["condition"] == "random" and row["p"] == RANDOM_P:
            out.setdefault(f"random {RANDOM_P}", []).append(float(row["return"]))
    return out


def seed_rows(seed, deviation, rollouts, switch, random):
    returns = {**candidate_returns(rollouts), **switch_returns(switch, random)}
    rows = []
    for row in read(deviation):
        r = np.array(returns[row["name"]])
        rows.append({"seed": seed, **{k: row[k] for k in FIELDS[1:8]},
                     "return": r.mean(), "return_se": r.std(ddof=1) / np.sqrt(len(r)),
                     "episodes": len(r)})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v
                             for k, v in row.items()})


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = list(dict.fromkeys(r["seed"] for r in rows))
    fig, axes = plt.subplots(2, len(seeds), figsize=(4.2 * len(seeds), 6.4), sharey="col")
    for col, seed in enumerate(seeds):
        mine = [r for r in rows if r["seed"] == seed]
        sweep = [r for r in mine if r["name"] == "z_r 0" or r["name"].startswith("CQL ")]
        groups = (
            ("CQL angle sweep", sweep, "o-", "tab:blue"),
            ("T05 latents", [r for r in mine if r["name"].startswith("T05")], "D", "tab:green"),
            ("rule C, kappa 0 to 0.1", [r for r in mine if r["kind"] == "switch"], "s-",
             "tab:orange"),
            ("random p=0.5", [r for r in mine if r["kind"] == "random"], "^", "tab:gray"),
        )
        for row, x in enumerate(("dev_data", "dev_opt")):
            ax = axes[row, col]
            for label, group, style, color in groups:
                xs = [float(r[x]) for r in group]
                ys = [r["return"] for r in group]
                es = [r["return_se"] for r in group]
                ax.errorbar(xs, ys, yerr=es, fmt=style, color=color, markersize=4, lw=1,
                            capsize=2, label=label)
            for r in sweep:
                ax.annotate(r["name"].split()[-1], (float(r[x]), r["return"]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel({"dev_data": "mean ||a - a_D||^2 on dataset states",
                           "dev_opt": "mean ||a - pi(s, z_r)||^2 on dataset states"}[x])
            if row == 0:
                ax.set_title(f"FB seed {seed}")
        axes[0, 0].set_ylabel("walk return")
        axes[1, 0].set_ylabel("walk return")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args, out):
    rows = []
    for seed, paths in enumerate(zip(args.deviation, args.rollouts, args.switch, args.random)):
        rows += seed_rows(seed, *paths)
    write_csv(out, rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ("deviation", "rollouts", "switch", "random"):
        parser.add_argument(f"--{name}", nargs="+", required=True, help="one CSV per seed")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--figure", type=str, default=None)
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare with --out, write nothing")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            run(args, Path(tmp) / "deviation.csv")
            same = filecmp.cmp(Path(tmp) / "deviation.csv", args.out, shallow=False)
            print(f"{Path(args.out).name}: {'identical' if same else 'DIFFERS'}")
            if not same:
                raise SystemExit(1)
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = run(args, args.out)
    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        plot(rows, args.figure)


if __name__ == "__main__":
    main()
