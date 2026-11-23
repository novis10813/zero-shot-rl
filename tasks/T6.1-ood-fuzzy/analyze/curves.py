"""Training curves of the finetune conditions over steps 0..200.

Reads `data/inputs/<condition>/<dynamics>/ft*/` (steps 0..50) and `<condition>_cont/` (steps 50..200).
Writes `data/curves.csv` (per step: mean over seeds of the all-task IQM and of cos(z, z_r)) and
`figures/curves.png`.

Usage (repo root):
    PYTHONPATH=.:tasks/T6.1-ood-fuzzy uv run python -m analyze.curves
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze.summarize import (  # noqa: E402
    DYNAMICS_ORDER,
    TABLE6_TASKS,
    _load_finetune_run,
    table6_aggregate,
)

FINETUNE = {"ood_4sensor": "4 sensors", "ood_custom": "customised sensors"}
COLORS = {"ood_4sensor": "#4C72B0", "ood_custom": "#DD8452"}


def _cos_curve(inputs_dir: Path, cond: str, dyn: str) -> pd.Series:
    """Mean over seeds and tasks of cos(z, z_r) per training step (1..200)."""
    frames = []
    for seed_dir in sorted((inputs_dir / cond / dyn).glob("ft*")):
        parts = [pd.read_csv(seed_dir / "train.csv")]
        cont = inputs_dir / f"{cond}_cont" / dyn / seed_dir.name / "train.csv"
        if cont.exists():
            parts.append(pd.read_csv(cont))
        frames.append(pd.concat(parts)[["task", "step", "cos_to_zr"]])
    return pd.concat(frames).groupby("step").cos_to_zr.mean()


def build_curves(inputs_dir: Path) -> pd.DataFrame:
    rows = []
    for dyn in DYNAMICS_ORDER:
        for cond in FINETUNE:
            seed_dirs = sorted((inputs_dir / cond / dyn).glob("ft*"))
            runs = [_load_finetune_run(inputs_dir, cond, dyn, d) for d in seed_dirs]
            steps = runs[0].index
            arr = np.stack([r.to_numpy(dtype=np.float64) for r in runs])
            iqm = table6_aggregate(arr)
            cos = _cos_curve(inputs_dir, cond, dyn)
            for i, step in enumerate(steps):
                rows.append({
                    "dynamics": dyn,
                    "condition": cond,
                    "step": int(step),
                    "table6_iqm": float(iqm[i]),
                    "cos_to_zr": float(cos.get(int(step), np.nan)),
                    **{t: float(arr[:, i, j].mean()) for j, t in enumerate(TABLE6_TASKS)},
                })
    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, len(DYNAMICS_ORDER), figsize=(4.2 * len(DYNAMICS_ORDER), 7), sharex=True)
    for c, dyn in enumerate(DYNAMICS_ORDER):
        ax_iqm, ax_cos = axes[0, c], axes[1, c]
        ref = summary[summary.dynamics == dyn].set_index("condition")
        ax_iqm.axhline(ref.loc["clean_fb_zr", "table6_iqm"], color="0.4", ls=":", label="Frozen FB $z_r$")
        ax_iqm.axhline(ref.loc["clean_anfis_transfer", "table6_iqm"], color="0.4", ls="--", label="Clean ANFIS-z transfer")
        for cond, label in FINETUNE.items():
            d = curves[(curves.dynamics == dyn) & (curves.condition == cond)]
            ax_iqm.plot(d.step, d.table6_iqm, color=COLORS[cond], lw=2, label=label)
            ax_cos.plot(d.step, d.cos_to_zr, color=COLORS[cond], lw=2)
        ax_iqm.axvline(50, color="0.8", lw=1)
        ax_cos.axvline(50, color="0.8", lw=1)
        ax_iqm.set_title(dyn, fontweight="bold")
        ax_iqm.grid(alpha=0.25)
        ax_cos.grid(alpha=0.25)
        ax_cos.set_xlabel("REINFORCE step")
        if c == 0:
            ax_iqm.set_ylabel("All-task IQM (mean over 10 seeds)")
            ax_cos.set_ylabel("cos(z, $z_r$) (mean over tasks, seeds)")
            ax_iqm.legend(fontsize=8)
    fig.suptitle("Finetune curves (gray vertical line: step 50, end of the first run)", y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    from analyze.summarize import summarize

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--inputs_dir", type=Path, default=Path("tasks/T6.1-ood-fuzzy/data/inputs"))
    parser.add_argument("--out_csv", type=Path, default=Path("tasks/T6.1-ood-fuzzy/data/curves.csv"))
    parser.add_argument("--out_fig", type=Path, default=Path("tasks/T6.1-ood-fuzzy/figures/curves.png"))
    args = parser.parse_args()

    curves = build_curves(args.inputs_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.out_csv, index=False, float_format="%.4f")
    plot_curves(curves, summarize(args.inputs_dir), args.out_fig)
    print(f"Wrote {args.out_csv} and {args.out_fig}")


if __name__ == "__main__":
    main()
