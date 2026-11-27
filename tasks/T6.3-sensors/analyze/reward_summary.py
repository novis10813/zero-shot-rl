"""Summary tables of the implied-reward analyses for REPORT.md section 5.

Reads `data/z_reward_corr.csv` (from `analyze.z_reward`) and `data/reward_template.csv` (from
`analyze.reward_template`) and writes `data/reward_summary.md` with three tables:

    1. mean Pearson correlation of r_z with the true reward and with r_{z_r}, per task
    2. R^2 of the Walker reward-template fit, per task (seed-average and per-seed fits)
    3. walk speed weight b_v of the template fit, per OOD environment

Means run over the 4 OOD environments and the 10 seeds.

Usage (repo root):
    PYTHONPATH=.:tasks/T6.3-sensors uv run python -m analyze.reward_summary
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

TASKS = ("stand", "walk", "run", "flip")
NAMES = {"cmaes_anfis": "ANFIS 4 sensors", "cmaes_z": "direct z"}


def correlation_table(corr: pd.DataFrame) -> str:
    zr = corr[corr.condition == "z_r"].set_index("task").corr_true
    found = corr[corr.condition != "z_r"].groupby(["task", "condition"])[["corr_true", "corr_zr"]].mean()
    lines = [
        "| task | z_r vs true | ANFIS 4 sensors vs true | direct z vs true | ANFIS 4 sensors vs z_r | direct z vs z_r |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t in TASKS:
        a, z = found.loc[(t, "cmaes_anfis")], found.loc[(t, "cmaes_z")]
        lines.append(f"| {t} | {zr[t]:.2f} | {a.corr_true:.2f} | {z.corr_true:.2f} | {a.corr_zr:.2f} | {z.corr_zr:.2f} |")
    return "\n".join(lines)


def r2_table(fits: pd.DataFrame) -> str:
    fits = fits.assign(seed=fits.seed.astype(str))
    main = fits[fits.seed.isin(["-", "mean"])]
    per_seed = fits[~fits.seed.isin(["-", "mean"])]
    lines = [
        "| task | true | z_r | ANFIS 4 sensors (seed mean) | ANFIS 4 sensors (per seed) "
        "| direct z (seed mean) | direct z (per seed) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in TASKS:
        m, s = main[main.task == t], per_seed[per_seed.task == t]

        def one(target):
            return m[m.target == target].r2.iloc[0]

        def rng(target):
            v = m[m.target == target].r2
            return f"{v.mean():.2f} ({v.min():.2f}–{v.max():.2f})"

        lines.append(
            f"| {t} | {one('true'):.3f} | {one('z_r'):.2f} | {rng('cmaes_anfis')} "
            f"| {s[s.target == 'cmaes_anfis'].r2.mean():.2f} | {rng('cmaes_z')} | {s[s.target == 'cmaes_z'].r2.mean():.2f} |"
        )
    return "\n".join(lines)


def walk_speed_table(fits: pd.DataFrame) -> str:
    fits = fits.assign(seed=fits.seed.astype(str))
    w = fits[(fits.task == "walk") & fits.seed.isin(["-", "mean"])]
    lines = ["| target | env | b_v | R² |", "|---|---|---:|---:|"]
    for _, r in w.iterrows():
        lines.append(f"| {NAMES.get(r.target, r.target)} | {r.env} | {r.b_v:.2f} | {r.r2:.2f} |")
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", type=Path, default=Path("tasks/T6.3-sensors/data"))
    args = parser.parse_args()
    corr = pd.read_csv(args.data_dir / "z_reward_corr.csv")
    fits = pd.read_csv(args.data_dir / "reward_template.csv")
    text = "\n\n".join([correlation_table(corr), r2_table(fits), walk_speed_table(fits)]) + "\n"
    (args.data_dir / "reward_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
