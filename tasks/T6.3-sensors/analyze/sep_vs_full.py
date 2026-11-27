"""Compare sep-CMA-ES on direct z (T6.3) with T6.2 CMA-ES on z and sep-CMA-ES on ANFIS, on the same seeds.

Every condition is restricted to the finetune seeds present in T6.3 (`ft1000`, `ft1010`, `ft1020`),
so all rows pair the same seeds and OOD environments. Scores are aggregated per environment with
`analysis.table6_aggregate`: `select_iqm` is the all-task IQM at the best eval step, `final_iqm` at
the last step (800k env steps). Writes `data/sep_vs_full.csv` and `data/sep_vs_full.md`.

Usage (repo root):
    PYTHONPATH=.:tasks/T6.3-sensors uv run python -m analyze.sep_vs_full
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from analysis.table6_aggregate import TASKS
from analysis.table6_aggregate import aggregate as table6_aggregate

ENVS = ("friction_mild", "friction_high_mild", "actuator_mild", "actuator_high_mild")
SEEDS = ("ft1000", "ft1010", "ft1020")
T63_INPUTS = Path("tasks/T6.3-sensors/data/inputs")
T62_INPUTS = Path("tasks/T6.2-faster-update-algo/data/inputs")

# (label, inputs dir, condition dir)
CONDITIONS = (
    ("sep-CMA-ES / z", T63_INPUTS, "sepcmaes_z"),
    ("CMA-ES (full-cov) / z", T62_INPUTS, "cmaes_z"),
    ("sep-CMA-ES / anfis (4 sensors)", T62_INPUTS, "cmaes_anfis"),
    ("sep-CMA-ES / anfis (custom sensors)", T62_INPUTS, "cmaes_anfis_custom"),
)


def condition_rows(label: str, root: Path, cond: str, seeds: Sequence[str] = SEEDS) -> List[Dict[str, object]]:
    rows = []
    for env in ENVS:
        runs = [pd.read_csv(root / cond / env / s / "table6.csv").set_index("step")[TASKS] for s in seeds]
        steps = runs[0].index
        if not all(r.index.equals(steps) for r in runs):
            raise ValueError(f"{cond}/{env}: runs must share eval steps")
        arr = np.stack([r.to_numpy(dtype=np.float64) for r in runs])  # [seeds, steps, tasks]
        per_step = table6_aggregate(arr)
        best = int(np.argmax(per_step))
        row: Dict[str, object] = {
            "env": env,
            "condition": label,
            "selected_step": int(steps[best]),
            "select_iqm": float(per_step[best]),
            "final_iqm": float(per_step[-1]),
        }
        row.update({t: float(arr[:, -1, i].mean()) for i, t in enumerate(TASKS)})
        rows.append(row)
    return rows


def summarize(conditions=CONDITIONS) -> pd.DataFrame:
    rows = [r for label, root, cond in conditions for r in condition_rows(label, root, cond)]
    df = pd.DataFrame(rows)
    means = df.groupby("condition", sort=False)[["select_iqm", "final_iqm", *TASKS]].mean().reset_index()
    means.insert(0, "env", "4-env mean")
    return pd.concat([df, means], ignore_index=True)


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["env", "condition", "select_iqm", "final_iqm", *TASKS]
    header = "| OOD env | condition | select IQM | final IQM | " + " | ".join(f"{t} (final)" for t in TASKS) + " |"
    lines = [header, "|---|---|" + "---:|" * (len(cols) - 2)]
    for _, r in df[cols].iterrows():
        vals = " | ".join(f"{r[c]:.1f}" for c in cols[2:])
        lines.append(f"| {r['env']} | {r['condition']} | {vals} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_dir", default="tasks/T6.3-sensors/data")
    args = parser.parse_args()
    df = summarize()
    out = Path(args.out_dir)
    df.to_csv(out / "sep_vs_full.csv", index=False, float_format="%.4f")
    (out / "sep_vs_full.md").write_text(to_markdown(df))
    print(to_markdown(df))


if __name__ == "__main__":
    main()
