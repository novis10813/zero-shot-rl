"""Walk return against action deviation of the policies of a fixed FB, per seed.

It joins the deviations of deviation.py with the logged walk returns of the same
policies:
    candidate   every latent of the CQL angle sweep (rollouts CSV of
                zero_shot_rl.eval.latents, 100 episodes)
    switch      rule C at each kappa (switch_eval.py kappa sweep CSV, 100 episodes)
    random      a_pess with probability 0.5 at every step (switch_eval.py --random_ps
                CSV, 100 episodes)
    final       the 300-episode comparison (switch_eval.py --episodes 300 --kappas 0
                --random_ps 0.5): pure FB, always pess, random and Algo 1 (rule C at
                kappa 0), with the deviations of z_r, the CQL latent, random p 0.5 and
                rule C at kappa 0
and writes

    deviation.csv              seed, kind, name, kappa, p, pess_frac, dev_data (mean
                               ||a - a_D||^2 on dataset states), dev_opt (mean
                               ||a - pi(s, z_r)||^2), return (mean), return_se, episodes
    return_vs_deviation.png    return against dev_opt, one panel per seed: the angle
                               sweep and the four policies of the final comparison

Run from the repository root:
    PYTHONPATH=.:tasks/T06-latent-switching uv run python -m analyze.deviation \
        --deviation <dev0.csv> <dev1.csv> <dev2.csv> --rollouts <r0.csv> <r1.csv> <r2.csv> \
        --switch <s0.csv> <s1.csv> <s2.csv> --random <p0.csv> <p1.csv> <p2.csv> \
        --final <f0.csv> <f1.csv> <f2.csv> --out <deviation.csv> [--figure <png>] [--check]
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
ANGLE = "7.5"  # the CQL latent of rule C
# the 300-episode comparison: condition of switch_eval.py, name, deviation row
FINAL = (("opt", "pure FB", "z_r 0"), ("pess", "always pess", f"CQL {ANGLE}"),
         ("random", "random", f"random {RANDOM_P}"), ("switch", "Algo 1", "switch 0"))
FINAL_COLORS = {"pure FB": "tab:red", "always pess": "tab:blue", "random": "tab:purple",
                "Algo 1": "tab:orange"}


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


def final_returns(path):
    """{name in FINAL: returns} of the 300-episode comparison (kappa 0, p 0.5)."""
    names = {condition: name for condition, name, _ in FINAL}
    out = {}
    for row in read(path):
        out.setdefault(names[row["condition"]], []).append(float(row["return"]))
    return out


def stats(seed, deviation_row, returns, **fields):
    r = np.array(returns)
    return {"seed": seed, **{k: deviation_row[k] for k in FIELDS[1:8]}, **fields,
            "return": r.mean(), "return_se": r.std(ddof=1) / np.sqrt(len(r)),
            "episodes": len(r)}


def seed_rows(seed, deviation, rollouts, switch, random, final=None):
    returns = {**candidate_returns(rollouts), **switch_returns(switch, random)}
    devs = {row["name"]: row for row in read(deviation)}
    rows = [stats(seed, row, returns[name]) for name, row in devs.items()]
    if final is not None:
        final_ret = final_returns(final)
        for _, name, dev_name in FINAL:
            rows.append(stats(seed, devs[dev_name], final_ret[name], kind="final", name=name))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v
                             for k, v in row.items()})


def plot(rows, path):
    """One panel per seed: return against change from the pure FB actions. Gray: the
    angle sweep, 100 episodes per point except 0 and 7.5 deg, which are the pure FB and
    always pess points of the 300-episode comparison. Colored: that comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = list(dict.fromkeys(r["seed"] for r in rows))
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.6 * len(seeds), 4.2))
    for ax, seed in zip(axes, seeds):
        mine = [r for r in rows if r["seed"] == seed]
        final = {r["name"]: r for r in mine if r["kind"] == "final"}
        own = {dev_name: final[name] for _, name, dev_name in FINAL}
        sweep = [own.get(r["name"], r) for r in mine if r["kind"] == "candidate"
                 and (r["name"] == "z_r 0" or r["name"].startswith("CQL "))]
        ax.errorbar([float(r["dev_opt"]) for r in sweep], [r["return"] for r in sweep],
                    yerr=[r["return_se"] for r in sweep], fmt="o-", color="0.65",
                    markersize=3, lw=1, capsize=2, zorder=1,
                    label="z_r moved 0 to 20 deg toward CQL")
        for r in sweep:
            if r["kind"] == "candidate":
                ax.annotate(f"{r['name'].split()[-1]} deg", (float(r["dev_opt"]), r["return"]),
                            fontsize=6, color="0.45", xytext=(3, -9),
                            textcoords="offset points")
        for r in (r for r in mine if r["kind"] == "final"):
            color = FINAL_COLORS[r["name"]]
            ax.errorbar(float(r["dev_opt"]), r["return"], yerr=r["return_se"], fmt="o",
                        color=color, markersize=7, capsize=3, zorder=2)
            ax.annotate(r["name"], (float(r["dev_opt"]), r["return"]), fontsize=9,
                        color=color, weight="bold", xytext=(6, 4), textcoords="offset points")
        ax.set_title(f"FB seed {seed}")
        ax.set_xlabel("action change from pure FB\nmean ||a - pi(s, z_r)||^2 on dataset states")
    axes[0].set_ylabel("walk return (mean +- SE)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args, out):
    rows = []
    for seed, paths in enumerate(zip(args.deviation, args.rollouts, args.switch, args.random,
                                     args.final)):
        rows += seed_rows(seed, *paths)
    write_csv(out, rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ("deviation", "rollouts", "switch", "random", "final"):
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
