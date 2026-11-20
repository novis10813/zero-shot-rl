"""Summarize T5.1-ood-fuzzy Table-6 IQM across OOD dynamics, conditions, and finetune seeds.

Expected input layout under `--inputs_dir` (default `tasks/T5.1-ood-fuzzy/data/inputs`):
    <condition>/<dynamics>/ft<seed>/table6.csv
    <condition>/<dynamics>/ft<seed>/baseline_table6.csv   (used for clean_fb_zr)

Conditions:
    - clean_anfis_transfer : clean-trained ANFIS-z evaluated zero-shot in OOD env
    - ood_4sensor          : 4-sensor ANFIS-z finetuned in OOD env
    - ood_custom           : shift-specific expanded-sensor ANFIS-z finetuned in OOD env

Usage (repo root):
    PYTHONPATH=. uv run python -m tasks.T5.1-ood-fuzzy.analyze.summarize
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.table6_aggregate import TASKS as TABLE6_TASKS
from analysis.table6_aggregate import aggregate as table6_aggregate

DYNAMICS_ORDER = ("friction_mild", "friction_high_mild", "actuator_mild", "actuator_high_mild")
CONDITIONS = ("clean_anfis_transfer", "ood_4sensor", "ood_custom")


def _load_step_row(csv_path: Path, step: int) -> np.ndarray:
    df = pd.read_csv(csv_path).set_index("step")
    if step not in df.index:
        step = int(df.index.max())
    return df.loc[step, TABLE6_TASKS].to_numpy(dtype=np.float64)


def summarize(inputs_dir: Path, step: int = 50) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dyn in DYNAMICS_ORDER:
        # Clean FB (z_r) baseline from any available condition's baseline_table6.csv
        zr_scores: List[np.ndarray] = []
        for cond in CONDITIONS:
            cand_dirs = sorted((inputs_dir / cond / dyn).glob("ft*"))
            if cand_dirs:
                for d in cand_dirs:
                    base_csv = d / "baseline_table6.csv"
                    if base_csv.exists():
                        zr_scores.append(_load_step_row(base_csv, 0))
                if zr_scores:
                    break
        if zr_scores:
            arr = np.stack(zr_scores)
            t6 = float(table6_aggregate(arr)[0])
            row: Dict[str, object] = {"dynamics": dyn, "condition": "clean_fb_zr", "n_seeds": len(zr_scores), "table6_iqm": t6}
            for i, t in enumerate(TABLE6_TASKS):
                row[t] = float(arr[:, i].mean())
            rows.append(row)

        for cond in CONDITIONS:
            seed_dirs = sorted((inputs_dir / cond / dyn).glob("ft*"))
            scores: List[np.ndarray] = []
            for d in seed_dirs:
                t6_csv = d / "table6.csv"
                if t6_csv.exists():
                    scores.append(_load_step_row(t6_csv, 0 if cond == "clean_anfis_transfer" else step))
            if not scores:
                continue
            arr = np.stack(scores)
            t6 = float(table6_aggregate(arr)[0])
            row = {"dynamics": dyn, "condition": cond, "n_seeds": len(scores), "table6_iqm": t6}
            for i, t in enumerate(TABLE6_TASKS):
                row[t] = float(arr[:, i].mean())
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--inputs_dir", type=Path, default=Path("tasks/T5.1-ood-fuzzy/data/inputs"))
    parser.add_argument("--out_csv", type=Path, default=Path("tasks/T5.1-ood-fuzzy/data/summary.csv"))
    parser.add_argument("--step", type=int, default=50)
    args = parser.parse_args()

    df = summarize(args.inputs_dir, step=args.step)
    if df.empty:
        print(f"No results found under {args.inputs_dir}")
        return
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, float_format="%.2f")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
