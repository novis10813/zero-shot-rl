"""Plot the learned ANFIS weight r~(s) against the original reward r(s) for step-50 models.

Reuses the extraction and plotting functions of `tasks/T6-anfis/analyze/compare.py`, applied to the
`<task>_final.pt` (step 50) models of the 4-sensor ANFIS of finetune seed 1000, one set per
environment (clean and the four OOD dynamics).

Modes (repo root):
    1. Extract reward grids from saved models (writes data/inputs/reward_grid/<env>.npz):
       PYTHONPATH=.:tasks/T5.1-ood-fuzzy uv run python -m analyze.reward_figs --extract \\
           clean=<models_dir> friction_mild=<models_dir> ...
       and, for the custom-sensor ANFIS (8 or 7 sensors), `--extract_extended friction_mild_custom=<models_dir>`
    2. Regenerate figures/step50/<env>/reward_{slices,heatmap}.png from data/inputs/reward_grid/:
       PYTHONPATH=.:tasks/T5.1-ood-fuzzy uv run python -m analyze.reward_figs
"""

import importlib.util
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.table6_aggregate import TASKS  # noqa: E402
from zero_shot_rl.latent.anfis import load_anfis  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]
GRID_DIR = TASK_DIR / "data" / "inputs" / "reward_grid"  # step 50; `reward_grid_step<N>` otherwise
FIG_DIR = TASK_DIR / "figures" / "step50"  # `figures/step<N>` otherwise

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


def extract_extended(name: str, models_dir: Path, n_slice: int = 200, n_grid: int = 80) -> Path:
    """Like compare.extract_reward_grid for ANFIS models with extra sensors beyond the 4 reward features.

    The extra sensors are held at the reference-data mean stored in the model (`x_mean`) when sweeping
    the 4 reward features. Each extra sensor also gets its own 1D slice over mean +- 3 std, with the
    4 reward features at the task's reference posture. The original reward does not depend on the extras.
    """
    first, _ = load_anfis(models_dir / f"{TASKS[0]}_final.pt")
    names = list(first.sensor_names)
    assert names[:4] == t6_compare.FEATURES, names
    arrays = {
        "tasks": np.array(TASKS),
        "features": np.array(names),
        "probe_ids": np.array([p[0] for p in t6_compare.PROBES]),
        "probe_labels": np.array([p[1] for p in t6_compare.PROBES]),
    }
    for task in TASKS:
        model, _ = load_anfis(models_dir / f"{task}_final.pt")
        assert list(model.sensor_names) == names
        mean = model.x_mean.numpy().astype(np.float32)
        std = model.x_std.numpy().astype(np.float32)
        extra = mean[4:]

        def full(x4):
            x4 = np.asarray(x4, dtype=np.float32)
            return np.concatenate([x4, np.tile(extra, (len(x4), 1))], axis=1)

        def anfis(x):
            with torch.no_grad():
                return model(torch.from_numpy(x)).squeeze(-1).numpy().astype(np.float32)

        probe_x = np.array([p[2] for p in t6_compare.PROBES], dtype=np.float32)
        arrays["probe_x"] = probe_x
        arrays[f"probe_true_{task}"] = t6_compare.true_reward(task, probe_x).astype(np.float32)
        arrays[f"probe_anfis_{task}"] = anfis(full(probe_x))

        base = np.concatenate([np.array(t6_compare.SLICE_BASE[task], dtype=np.float32), extra])
        for j, feat in enumerate(names):
            if j < 4:
                lo, hi = t6_compare.SLICE_RANGES[feat]
            else:
                lo, hi = float(mean[j] - 3 * std[j]), float(mean[j] + 3 * std[j])
            xs = np.linspace(lo, hi, n_slice, dtype=np.float32)
            pts = np.tile(base, (n_slice, 1))
            pts[:, j] = xs
            arrays[f"slice_x_{task}_{feat}"] = xs
            arrays[f"slice_true_{task}_{feat}"] = t6_compare.true_reward(task, pts).astype(np.float32)
            arrays[f"slice_anfis_{task}_{feat}"] = anfis(pts)

        ix, iy = t6_compare.HEATMAP_AXES[task]
        gx = np.linspace(*t6_compare.SLICE_RANGES[names[ix]], n_grid, dtype=np.float32)
        gy = np.linspace(*t6_compare.SLICE_RANGES[names[iy]], n_grid, dtype=np.float32)
        xx, yy = np.meshgrid(gx, gy)
        pts = np.tile(base, (n_grid * n_grid, 1))
        pts[:, ix] = xx.ravel()
        pts[:, iy] = yy.ravel()
        arrays[f"grid_x_{task}"] = gx
        arrays[f"grid_y_{task}"] = gy
        arrays[f"grid_true_{task}"] = t6_compare.true_reward(task, pts).reshape(n_grid, n_grid).astype(np.float32)
        arrays[f"grid_anfis_{task}"] = anfis(pts).reshape(n_grid, n_grid)

    out = GRID_DIR / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


def plot_slices_extended(grid, out_path: Path) -> None:
    """1D slices for every sensor in the model: the 4 reward features and the extra sensors."""
    names = [str(n) for n in grid["features"]]
    labels = t6_compare.FEATURE_LABELS + names[4:]
    fig, axes = plt.subplots(len(TASKS), len(names), figsize=(3.3 * len(names), 9.5), sharey="row")
    for r, task in enumerate(TASKS):
        for c, (feat, label) in enumerate(zip(names, labels)):
            ax = axes[r, c]
            x = grid[f"slice_x_{task}_{feat}"]
            y_anf = grid[f"slice_anfis_{task}_{feat}"]
            ax.axhline(0.0, color="0.65", lw=0.8, ls=":")
            ax.plot(x, grid[f"slice_true_{task}_{feat}"], color="#4C72B0", lw=2.0, ls="--", label="Original r(s)")
            ax.plot(x, y_anf, color="#DD8452", lw=2.2, label="ANFIS r~(s)")
            ax.fill_between(x, y_anf, 0.0, where=(y_anf < 0), color="#C44E52", alpha=0.18)
            if r == 0:
                ax.set_title(label, fontsize=9, fontweight="bold", color="#C44E52" if c >= 4 else "black")
            if c == 0:
                ax.set_ylabel(f"{task}\nreward / weight", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle(
        "Original Reward r(s) vs. Learned ANFIS Weight r~(s) (1D slices; extra sensors in red, "
        "other features held at the reference posture / data mean)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--extract", nargs="+", metavar="ENV=MODELS_DIR", default=None)
    parser.add_argument(
        "--extract_extended",
        nargs="+",
        metavar="NAME=MODELS_DIR",
        default=None,
        help="Models with extra sensors beyond the 4 reward features (the custom-sensor ANFIS)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Step of the saved `<task>_final.pt` models (50 for the first run, 200 after continuing)",
    )
    args = parser.parse_args()

    global GRID_DIR, FIG_DIR
    if args.step != 50:
        GRID_DIR = TASK_DIR / "data" / "inputs" / f"reward_grid_step{args.step}"
        FIG_DIR = TASK_DIR / "figures" / f"step{args.step}"

    if args.extract or args.extract_extended:
        for item in args.extract or []:
            name, models_dir = item.split("=", 1)
            print(f"Wrote {extract(name, Path(models_dir))}")
        for item in args.extract_extended or []:
            name, models_dir = item.split("=", 1)
            print(f"Wrote {extract_extended(name, Path(models_dir))}")
        return

    for npz in sorted(GRID_DIR.glob("*.npz")):
        grid = np.load(npz)
        out_dir = FIG_DIR / npz.stem
        if len(grid["features"]) > 4:
            plot_slices_extended(grid, out_dir / "reward_slices.png")
        else:
            t6_compare.plot_slices(grid, out_dir / "reward_slices.png")
        t6_compare.plot_heatmaps(grid, out_dir / "reward_heatmap.png")
        print(f"Wrote figures in {out_dir}")


if __name__ == "__main__":
    main()
