"""Summarize T6.1-ood-fuzzy Table-6 IQM across OOD dynamics, conditions, and finetune seeds.

Expected input layout under `--inputs_dir` (default `tasks/T6.1-ood-fuzzy/data/inputs`):
    <condition>/<dynamics>/ft<seed>/table6.csv
    <condition>/<dynamics>/ft<seed>/baseline_table6.csv   (used for clean_fb_zr)

Conditions:
    - clean_anfis_transfer : clean-trained ANFIS-z evaluated zero-shot in OOD env
    - ood_4sensor          : 4-sensor ANFIS-z finetuned in OOD env
    - ood_custom           : shift-specific expanded-sensor ANFIS-z finetuned in OOD env

Usage (repo root):
    PYTHONPATH=.:tasks/T6.1-ood-fuzzy uv run python -m analyze.summarize
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


def _load_steps(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path).set_index("step")[TABLE6_TASKS]


def _load_finetune_run(inputs_dir: Path, cond: str, dyn: str, seed_dir: Path) -> pd.DataFrame:
    """Per-step scores of one finetune seed: steps 0..50 plus the continued run (steps 60..) if present.

    The continued run re-evaluates step 50 from the restored model, which must reproduce the original.
    """
    base = _load_steps(seed_dir / "table6.csv")
    cont_csv = inputs_dir / f"{cond}_cont" / dyn / seed_dir.name / "table6.csv"
    if not cont_csv.exists():
        return base
    cont = _load_steps(cont_csv)
    resume_step = int(cont.index.min())
    np.testing.assert_allclose(cont.loc[resume_step].to_numpy(), base.loc[resume_step].to_numpy(), atol=1e-3)
    return pd.concat([base.loc[:resume_step], cont.loc[cont.index > resume_step]])


def summarize(inputs_dir: Path) -> pd.DataFrame:
    """Finetune conditions report the step with the highest all-task IQM (as analysis.table6_aggregate).

    Finetune runs continued beyond step 50 (`<condition>_cont/`) are joined to the original steps.
    `iqm_step50` is the all-task IQM at step 50 and `final_iqm` at the last step, both without selection.
    """
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
                        zr_scores.append(_load_steps(base_csv).loc[0].to_numpy(dtype=np.float64))
                if zr_scores:
                    break
        if zr_scores:
            arr = np.stack(zr_scores)
            t6 = float(table6_aggregate(arr[:, None, :])[0])
            row: Dict[str, object] = {"dynamics": dyn, "condition": "clean_fb_zr", "n_seeds": len(zr_scores), "step": 0, "table6_iqm": t6, "iqm_step50": t6, "final_iqm": t6}
            for i, t in enumerate(TABLE6_TASKS):
                row[t] = float(arr[:, i].mean())
            rows.append(row)

        for cond in CONDITIONS:
            seed_dirs = sorted((inputs_dir / cond / dyn).glob("ft*"))
            runs = [_load_finetune_run(inputs_dir, cond, dyn, d) for d in seed_dirs if (d / "table6.csv").exists()]
            if not runs:
                continue
            steps = runs[0].index
            assert all(r.index.equals(steps) for r in runs), "runs must share eval steps"
            arr = np.stack([r.to_numpy(dtype=np.float64) for r in runs])  # [seeds, steps, tasks]
            per_step = table6_aggregate(arr)
            best = int(np.argmax(per_step))
            row = {
                "dynamics": dyn,
                "condition": cond,
                "n_seeds": len(runs),
                "step": int(steps[best]),
                "table6_iqm": float(per_step[best]),
                "iqm_step50": float(per_step[list(steps).index(50)]) if 50 in steps else float(per_step[best]),
                "final_iqm": float(per_step[-1]),
            }
            for i, t in enumerate(TABLE6_TASKS):
                row[t] = float(arr[:, best, i].mean())
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    zr = df[df["condition"] == "clean_fb_zr"].set_index("dynamics")["table6_iqm"]
    df["delta_vs_zr"] = df["table6_iqm"] - df["dynamics"].map(zr)
    mean_rows = (
        df.groupby("condition", sort=False)[["table6_iqm", "iqm_step50", "final_iqm", *TABLE6_TASKS, "delta_vs_zr"]]
        .mean()
        .reset_index()
    )
    mean_rows.insert(0, "dynamics", "mean_of_4_dynamics")
    mean_rows.insert(2, "n_seeds", int(df["n_seeds"].min()))
    mean_rows.insert(3, "step", np.nan)
    return pd.concat([df, mean_rows], ignore_index=True)


def to_markdown(df: pd.DataFrame) -> str:
    names = {
        "clean_fb_zr": "Frozen FB $z_r$",
        "clean_anfis_transfer": "Clean ANFIS-z 直接遷移",
        "ood_4sensor": "OOD finetune，4 sensors",
        "ood_custom": "OOD finetune，客製 sensors",
    }
    lines = [
        "| OOD 環境 | 方法 | 選定 step | stand | walk | run | flip | All-task IQM | 相對 $z_r$ | 第 50 步 | 最後一步 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        dyn = "四環境平均" if r["dynamics"] == "mean_of_4_dynamics" else r["dynamics"]
        step = "-" if pd.isna(r["step"]) or r["condition"] in ("clean_fb_zr", "clean_anfis_transfer") else f"{int(r['step'])}"
        lines.append(
            f"| {dyn} | {names[r['condition']]} | {step} | {r['stand']:.1f} | {r['walk']:.1f} | {r['run']:.1f} | "
            f"{r['flip']:.1f} | {r['table6_iqm']:.1f} | {r['delta_vs_zr']:+.1f} | "
            f"{r['iqm_step50']:.1f} | {r['final_iqm']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--inputs_dir", type=Path, default=Path("tasks/T6.1-ood-fuzzy/data/inputs"))
    parser.add_argument("--out_csv", type=Path, default=Path("tasks/T6.1-ood-fuzzy/data/summary.csv"))
    args = parser.parse_args()

    df = summarize(args.inputs_dir)
    if df.empty:
        print(f"No results found under {args.inputs_dir}")
        return
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, float_format="%.2f")
    args.out_csv.with_suffix(".md").write_text(to_markdown(df))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
