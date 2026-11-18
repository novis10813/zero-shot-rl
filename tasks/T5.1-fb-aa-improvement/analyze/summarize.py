"""Summarize Table-6 IQM of Action Transformation (T5.1) for both 660000.pt (10 finetune seeds,
20 shared eval envs 5..24) and T05 3-seed comparison vs. T4.1.3.2 FB-AA.

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

SWEEP_DIR = TASK_DIR / "data/inputs/sweep_660k"
SWEEP_FT_SEEDS = list(range(1000, 1100, 10))
SWEEP_GROUPS = {
    "baseline": ["baseline"],
    "actuator_mild": ["actuator_mild", "actuator_strong"],
    "actuator_high_mild": ["actuator_high_mild", "actuator_high_strong"],
    "friction_mild": ["friction_mild", "friction_strong", "friction_high_mild", "friction_high_strong"],
}
ALL_CONFIGS = [
    "baseline",
    "actuator_mild",
    "actuator_strong",
    "actuator_high_mild",
    "actuator_high_strong",
    "friction_mild",
    "friction_strong",
    "friction_high_mild",
    "friction_high_strong",
]
OUT_660K_CSV = TASK_DIR / "data/table6_iqm_660k.csv"

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


def table6_iqm(ret_n_ep: np.ndarray, task_arr: np.ndarray) -> float:
    n_seeds = ret_n_ep.shape[0]
    t_iqm = np.zeros((n_seeds, len(TASKS)))
    for s in range(n_seeds):
        for ti, task in enumerate(TASKS):
            t_iqm[s, ti] = iqm(ret_n_ep[s, task_arr == task])
    return iqm(t_iqm.mean(axis=0))


def old_table6_iqm(ret_60: np.ndarray, old_seed: np.ndarray, old_task: np.ndarray) -> float:
    seeds = np.unique(old_seed)
    t_iqm = np.zeros((len(seeds), len(TASKS)))
    for si, s in enumerate(seeds):
        for ti, task in enumerate(TASKS):
            t_iqm[si, ti] = iqm(ret_60[(old_seed == s) & (old_task == task)])
    return iqm(t_iqm.mean(axis=0))


def build_660k_table() -> pd.DataFrame:
    by_cfg = {c: [] for c in ALL_CONFIGS}
    da_by_cfg = {c: [] for c in ALL_CONFIGS}
    tasks = None
    for fs in SWEEP_FT_SEEDS:
        for tag in SWEEP_GROUPS:
            with np.load(SWEEP_DIR / f"{tag}_ft{fs}.npz") as d:
                if tasks is None:
                    tasks = d["task"]
                for ci, cfg in enumerate(d["configs"]):
                    by_cfg[cfg].append(d["ret"][ci])
                    da_by_cfg[cfg].append(float(d["da"][ci, 2].mean()))

    clean_iqm = table6_iqm(np.stack(by_cfg["baseline"])[:, 0, :], tasks)
    rows = []
    for cfg in ALL_CONFIGS:
        arr = np.stack(by_cfg[cfg])  # [10, 3, 80]
        none_iqm = table6_iqm(arr[:, 0, :], tasks)
        star_iqm = np.nan if cfg.startswith("friction_") else table6_iqm(arr[:, 1, :], tasks)
        trans_iqm = table6_iqm(arr[:, 2, :], tasks)
        rows.append({
            "config": cfg,
            "none": round(none_iqm, 1),
            "star": round(star_iqm, 1) if not np.isnan(star_iqm) else np.nan,
            "new_transform": round(trans_iqm, 1),
            "none_minus_clean": round(none_iqm - clean_iqm, 1),
            "transform_minus_clean": round(trans_iqm - clean_iqm, 1),
            "mean_da": round(float(np.mean(da_by_cfg[cfg])), 3),
        })
    return pd.DataFrame(rows)


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

    baseline_idx = configs.index("baseline")
    clean_iqm = table6_iqm(new_rets[:, baseline_idx, 0, :], tasks)

    rows = []
    for ci, cfg in enumerate(configs):
        none_iqm = table6_iqm(new_rets[:, ci, 0, :], tasks)
        star_iqm = np.nan if cfg.startswith("friction_") else table6_iqm(new_rets[:, ci, 1, :], tasks)
        new_iqm = table6_iqm(new_rets[:, ci, 2, :], tasks)
        if cfg in old_configs:
            oi = old_configs.index(cfg)
            old_iqm = old_table6_iqm(old_ret[oi, 2, :], old_seed, old_task)
            old_diff = old_iqm - clean_iqm
        elif cfg in old_f_configs:
            oi = old_f_configs.index(cfg)
            old_iqm = table6_iqm(old_f_rets[:, oi, 2, :], old_f_tasks)
            old_diff = old_iqm - clean_iqm
        else:
            old_iqm = np.nan
            old_diff = np.nan
        rows.append({
            "config": cfg,
            "none": round(none_iqm, 1),
            "star": round(star_iqm, 1) if not np.isnan(star_iqm) else np.nan,
            "old_fb_aa": round(old_iqm, 1) if not np.isnan(old_iqm) else np.nan,
            "new_transform": round(new_iqm, 1),
            "old_minus_clean": round(old_diff, 1) if not np.isnan(old_diff) else np.nan,
            "new_minus_clean": round(new_iqm - clean_iqm, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify stored CSVs match byte-for-byte")
    args = parser.parse_args()

    outputs = [
        (OUT_660K_CSV, build_660k_table().to_csv(index=False)),
        (OUT_CSV, build_table().to_csv(index=False)),
    ]
    if args.check:
        for path, text in outputs:
            if path.read_text() != text:
                raise AssertionError(f"{path} does not match regenerated output")
            print(f"OK: {path} matches")
        return

    for path, text in outputs:
        path.write_text(text)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
