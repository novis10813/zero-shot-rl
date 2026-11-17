"""Summarize 3-seed Table-6 IQM of Action Transformation vs. T4.1.3.2 FB-AA.

Usage (repo root):
    PYTHONPATH=.:tasks/T5.1-fb-aa-improvement uv run python -m analyze.summarize [--check]
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.table6_aggregate import TASKS, iqm

TASK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parent.parent
NEW_INPUTS = [
    TASK_DIR / "data/inputs/eval_action_transform_stock_s0.npz",
    TASK_DIR / "data/inputs/eval_action_transform_stock_s1.npz",
    TASK_DIR / "data/inputs/eval_action_transform_stock_s2.npz",
]
OLD_INPUT = REPO_ROOT / "tasks/T4.1.3.2-fb-aa/data/inputs/adapt_episode.npz"
OLD_FRICTION_INPUTS = [
    TASK_DIR / "data/inputs/old_friction_s0.npz",
    TASK_DIR / "data/inputs/old_friction_s1.npz",
    TASK_DIR / "data/inputs/old_friction_s2.npz",
]
OUT_CSV = TASK_DIR / "data/table6_iqm.csv"


def table6_iqm(ret_3_20: np.ndarray, task_arr: np.ndarray) -> float:
    t_iqm = np.zeros((3, len(TASKS)))
    for s in range(3):
        for ti, task in enumerate(TASKS):
            t_iqm[s, ti] = iqm(ret_3_20[s, task_arr == task])
    return iqm(t_iqm.mean(axis=0))


def old_table6_iqm(ret_60: np.ndarray, old_seed: np.ndarray, old_task: np.ndarray) -> float:
    seeds = np.unique(old_seed)
    t_iqm = np.zeros((len(seeds), len(TASKS)))
    for si, s in enumerate(seeds):
        for ti, task in enumerate(TASKS):
            t_iqm[si, ti] = iqm(ret_60[(old_seed == s) & (old_task == task)])
    return iqm(t_iqm.mean(axis=0))


def build_table() -> pd.DataFrame:
    new_archs = [np.load(p) for p in NEW_INPUTS]
    new_rets = np.stack([a["ret"] for a in new_archs], axis=0)  # [3, C, 3, 20]
    configs = list(new_archs[0]["configs"])
    tasks = new_archs[0]["task"]

    old = np.load(OLD_INPUT)
    old_ret = old["ret"]
    old_configs = list(old["configs"])
    old_seed = old["seed"]
    old_task = old["task"]

    old_f_archs = [np.load(p) for p in OLD_FRICTION_INPUTS]
    old_f_rets = np.stack([a["ret"] for a in old_f_archs], axis=0)  # [3, C_f, 3, 20]
    old_f_configs = list(old_f_archs[0]["configs"])
    old_f_tasks = old_f_archs[0]["task"]

    rows = []
    for ci, cfg in enumerate(configs):
        none_iqm = table6_iqm(new_rets[:, ci, 0, :], tasks)
        star_iqm = table6_iqm(new_rets[:, ci, 1, :], tasks)
        new_iqm = table6_iqm(new_rets[:, ci, 2, :], tasks)
        if cfg in old_configs:
            oi = old_configs.index(cfg)
            old_iqm = old_table6_iqm(old_ret[oi, 2, :], old_seed, old_task)
            old_diff = old_iqm - none_iqm
        elif cfg in old_f_configs:
            oi = old_f_configs.index(cfg)
            old_iqm = table6_iqm(old_f_rets[:, oi, 2, :], old_f_tasks)
            old_diff = old_iqm - none_iqm
        else:
            old_iqm = np.nan
            old_diff = np.nan
        rows.append({
            "config": cfg,
            "none": round(none_iqm, 1),
            "star": round(star_iqm, 1),
            "old_fb_aa": round(old_iqm, 1) if not np.isnan(old_iqm) else np.nan,
            "new_transform": round(new_iqm, 1),
            "old_minus_none": round(old_diff, 1) if not np.isnan(old_diff) else np.nan,
            "new_minus_none": round(new_iqm - none_iqm, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify stored CSV matches byte-for-byte")
    args = parser.parse_args()

    df = build_table()
    csv_text = df.to_csv(index=False)
    if args.check:
        stored = OUT_CSV.read_text()
        if stored != csv_text:
            raise AssertionError(f"{OUT_CSV} does not match regenerated output")
        print(f"OK: {OUT_CSV} matches")
        return

    OUT_CSV.write_text(csv_text)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
