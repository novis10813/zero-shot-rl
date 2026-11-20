"""Plot the learned ANFIS weight r~(s) against the original reward r(s) for step-50 models.

Reuses the extraction and plotting functions of `tasks/T6-anfis/analyze/compare.py`, applied to the
`<task>_final.pt` (step 50) models of the 4-sensor ANFIS of finetune seed 1000, one set per
environment (clean and the four OOD dynamics).

Modes (repo root):
    1. Extract reward grids from saved models (writes data/inputs/reward_grid/<env>.npz):
       PYTHONPATH=.:tasks/T5.1-ood-fuzzy uv run python -m analyze.reward_figs --extract \\
           clean=<models_dir> friction_mild=<models_dir> ...
    2. Regenerate figures/step50/<env>/reward_{slices,heatmap}.png from data/inputs/reward_grid/:
       PYTHONPATH=.:tasks/T5.1-ood-fuzzy uv run python -m analyze.reward_figs
"""

import importlib.util
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from analysis.table6_aggregate import TASKS

REPO = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]
GRID_DIR = TASK_DIR / "data" / "inputs" / "reward_grid"
FIG_DIR = TASK_DIR / "figures" / "step50"

_spec = importlib.util.spec_from_file_location("t6_compare", REPO / "tasks/T6-anfis/analyze/compare.py")
t6_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t6_compare)


def extract(name: str, models_dir: Path) -> Path:
    """compare.extract_reward_grid reads `<task>_best.pt`, so link the step-50 `_final.pt` under that name."""
    out = GRID_DIR / f"{name}.npz"
    with tempfile.TemporaryDirectory() as tmp:
        for task in TASKS:
            (Path(tmp) / f"{task}_best.pt").symlink_to((models_dir / f"{task}_final.pt").resolve())
        t6_compare.extract_reward_grid(Path(tmp), out)
    return out


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--extract", nargs="+", metavar="ENV=MODELS_DIR", default=None)
    args = parser.parse_args()

    if args.extract:
        for item in args.extract:
            name, models_dir = item.split("=", 1)
            print(f"Wrote {extract(name, Path(models_dir))}")
        return

    for npz in sorted(GRID_DIR.glob("*.npz")):
        grid = np.load(npz)
        out_dir = FIG_DIR / npz.stem
        t6_compare.plot_slices(grid, out_dir / "reward_slices.png")
        t6_compare.plot_heatmaps(grid, out_dir / "reward_heatmap.png")
        print(f"Wrote figures in {out_dir}")


if __name__ == "__main__":
    main()
