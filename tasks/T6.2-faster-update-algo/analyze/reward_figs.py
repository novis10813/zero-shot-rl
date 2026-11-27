"""Plot 10-seed mean +- std ANFIS weights r~(s) vs. original reward r(s) for T6.2 sep-CMA-ES (800k).

Reuses feature definitions, true rewards, and 2D heatmap plotting from
``tasks/T6-anfis/analyze/compare.py``, applied to ``<task>_final.pt`` (800k env steps) across all
10 finetune seeds for both 4-sensor ANFIS (``cmaes_anfis``) and shift-specific custom-sensor ANFIS
(``cmaes_anfis_custom``) in each of the four OOD dynamics.

Modes (repo root):
    1. Extract 10-seed reward grids from local artifact directories (writes data/inputs/reward_grid/<name>.npz):
       PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.reward_figs \\
           --extract_multi friction_mild=dir1,dir2,... friction_mild_custom=dirA,dirB,...
    2. Regenerate figures/cmaes_800k/<name>/reward_{slices,heatmap}.png from data/inputs/reward_grid/:
       PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.reward_figs
"""

import importlib.util
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.table6_aggregate import TASKS  # noqa: E402
from zero_shot_rl.finetune.anfis_z import collect_reference_set  # noqa: E402
from zero_shot_rl.latent.anfis import load_anfis  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]
GRID_DIR = TASK_DIR / "data" / "inputs" / "reward_grid"
REF_BIN_DIR = TASK_DIR / "data" / "inputs" / "ref_binned"
FIG_DIR = TASK_DIR / "figures" / "cmaes_800k"

ALL_WALKER_SENSORS: Sequence[str] = (
    "torso_height",
    "torso_upright",
    "horizontal_velocity",
    "angmomentum",
    "com_vz",
    "qvel_right_ankle",
    "qvel_left_ankle",
    "qvel_right_hip",
    "qvel_left_hip",
    "qvel_right_knee",
    "qvel_left_knee",
)

_spec = importlib.util.spec_from_file_location("t6_compare", REPO / "tasks/T6-anfis/analyze/compare.py")
t6_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t6_compare)


def extract_multi(
    name: str,
    models_dirs: Sequence[Path],
    tag: str = "final",
    n_slice: int = 200,
    n_grid: int = 80,
    out_dir: Path = GRID_DIR,
) -> Path:
    """Extract 1D slices and 2D heatmaps aggregated (mean +- std) across multiple finetune seeds."""
    if not models_dirs:
        raise ValueError(f"No model directories provided for {name}")
    first, _ = load_anfis(models_dirs[0] / f"{TASKS[0]}_{tag}.pt")
    names = list(first.sensor_names)
    assert names[:4] == t6_compare.FEATURES, names

    arrays: Dict[str, np.ndarray] = {
        "tasks": np.array(TASKS),
        "features": np.array(names),
        "n_seeds": np.array(len(models_dirs), dtype=np.int32),
        "probe_ids": np.array([p[0] for p in t6_compare.PROBES]),
        "probe_labels": np.array([p[1] for p in t6_compare.PROBES]),
    }

    # Reference mean/std for extra sensors (shared offline reference pool across all seeds).
    ref_mean = first.x_mean.numpy().astype(np.float32)
    ref_std = first.x_std.numpy().astype(np.float32)
    extra = ref_mean[4:]

    probe_x = np.array([p[2] for p in t6_compare.PROBES], dtype=np.float32)
    arrays["probe_x"] = probe_x

    def full(x4: np.ndarray) -> np.ndarray:
        x4 = np.asarray(x4, dtype=np.float32)
        if extra.size == 0:
            return x4
        return np.concatenate([x4, np.tile(extra, (len(x4), 1))], axis=1)

    for task in TASKS:
        models = []
        for md in models_dirs:
            m, _ = load_anfis(md / f"{task}_{tag}.pt")
            assert list(m.sensor_names) == names
            models.append(m)

        def eval_all(pts: np.ndarray) -> np.ndarray:
            x_t = torch.from_numpy(pts)
            vals = []
            with torch.no_grad():
                for m in models:
                    vals.append(m(x_t).squeeze(-1).numpy().astype(np.float32))
            return np.stack(vals, axis=0)  # [n_seeds, N]

        arrays[f"probe_true_{task}"] = t6_compare.true_reward(task, probe_x).astype(np.float32)
        probe_vals = eval_all(full(probe_x))
        arrays[f"probe_anfis_{task}"] = probe_vals.mean(axis=0)
        arrays[f"probe_anfis_std_{task}"] = probe_vals.std(axis=0)

        base = (
            np.concatenate([np.array(t6_compare.SLICE_BASE[task], dtype=np.float32), extra])
            if extra.size > 0
            else np.array(t6_compare.SLICE_BASE[task], dtype=np.float32)
        )
        for j, feat in enumerate(names):
            if j < 4:
                lo, hi = t6_compare.SLICE_RANGES[feat]
            else:
                lo, hi = float(ref_mean[j] - 3 * ref_std[j]), float(ref_mean[j] + 3 * ref_std[j])
            xs = np.linspace(lo, hi, n_slice, dtype=np.float32)
            pts = np.tile(base, (n_slice, 1))
            pts[:, j] = xs
            vals = eval_all(pts)  # [n_seeds, n_slice]
            arrays[f"slice_x_{task}_{feat}"] = xs
            arrays[f"slice_true_{task}_{feat}"] = t6_compare.true_reward(task, pts).astype(np.float32)
            arrays[f"slice_anfis_{task}_{feat}"] = vals.mean(axis=0)
            arrays[f"slice_anfis_std_{task}_{feat}"] = vals.std(axis=0)

        ix, iy = t6_compare.HEATMAP_AXES[task]
        gx = np.linspace(*t6_compare.SLICE_RANGES[names[ix]], n_grid, dtype=np.float32)
        gy = np.linspace(*t6_compare.SLICE_RANGES[names[iy]], n_grid, dtype=np.float32)
        xx, yy = np.meshgrid(gx, gy)
        pts = np.tile(base, (n_grid * n_grid, 1))
        pts[:, ix] = xx.ravel()
        pts[:, iy] = yy.ravel()
        gvals = eval_all(pts).reshape(len(models), n_grid, n_grid)
        arrays[f"grid_x_{task}"] = gx
        arrays[f"grid_y_{task}"] = gy
        arrays[f"grid_true_{task}"] = t6_compare.true_reward(task, pts).reshape(n_grid, n_grid).astype(np.float32)
        arrays[f"grid_anfis_{task}"] = gvals.mean(axis=0)
        arrays[f"grid_anfis_std_{task}"] = gvals.std(axis=0)

    out = out_dir / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


def plot_slices_multi(grid, out_path: Path) -> None:
    """Plot 1D slices for all sensors showing multi-seed mean +- 1 std vs. original r(s)."""
    names = [str(n) for n in grid["features"]]
    n_seeds = int(grid["n_seeds"]) if "n_seeds" in grid else 1
    labels = t6_compare.FEATURE_LABELS + names[4:]
    fig, axes = plt.subplots(len(TASKS), len(names), figsize=(3.3 * len(names), 9.5), sharey="row")
    for r, task in enumerate(TASKS):
        for c, (feat, label) in enumerate(zip(names, labels)):
            ax = axes[r, c]
            x = grid[f"slice_x_{task}_{feat}"]
            y_mean = grid[f"slice_anfis_{task}_{feat}"]
            std_key = f"slice_anfis_std_{task}_{feat}"
            y_std = grid[std_key] if std_key in grid else None
            ax.axhline(0.0, color="0.65", lw=0.8, ls=":")
            ax.plot(x, grid[f"slice_true_{task}_{feat}"], color="#4C72B0", lw=2.0, ls="--", label="Original r(s)")
            ax.plot(
                x,
                y_mean,
                color="#DD8452",
                lw=2.2,
                label=f"ANFIS r~(s) ({n_seeds}-seed mean)" if n_seeds > 1 else "ANFIS r~(s)",
            )
            if y_std is not None and n_seeds > 1:
                ax.fill_between(
                    x,
                    y_mean - y_std,
                    y_mean + y_std,
                    color="#DD8452",
                    alpha=0.22,
                    label="+- 1 std",
                )
            ax.fill_between(x, y_mean, 0.0, where=(y_mean < 0), color="#C44E52", alpha=0.14)
            if r == 0:
                ax.set_title(label, fontsize=9, fontweight="bold", color="#C44E52" if c >= 4 else "black")
            if c == 0:
                ax.set_ylabel(f"{task}\nreward / weight", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=True)
    extra_note = "; extra sensors in red" if len(names) > 4 else ""
    fig.suptitle(
        f"Original Reward r(s) vs. Learned ANFIS Weight r~(s) (sep-CMA-ES 800k, {n_seeds}-seed mean +- 1 std"
        f"{extra_note})",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def extract_ref_binned(
    name: str,
    models_dirs: Sequence[Path],
    ref_feats_by_name: Dict[str, np.ndarray],
    ref_rewards_by_task: Dict[str, np.ndarray],
    tag: str = "final",
    n_bins: int = 25,
    min_bin_count: int = 20,
    out_dir: Path = REF_BIN_DIR,
) -> Path:
    """Compute binned marginal averages of r(s) and 10-seed ANFIS r~(s) on the real reference dataset."""
    if not models_dirs:
        raise ValueError(f"No model directories provided for {name}")
    first, _ = load_anfis(models_dirs[0] / f"{TASKS[0]}_{tag}.pt")
    names = list(first.sensor_names)
    X_all = np.stack([ref_feats_by_name[s] for s in names], axis=1).astype(np.float32)  # [N, D]
    X_t = torch.from_numpy(X_all)

    arrays: Dict[str, np.ndarray] = {
        "tasks": np.array(TASKS),
        "features": np.array(names),
        "n_seeds": np.array(len(models_dirs), dtype=np.int32),
        "n_samples": np.array(len(X_all), dtype=np.int32),
    }

    # Precompute equal-width bins over [p01, p99] per sensor.
    bin_masks: Dict[str, List[np.ndarray]] = {}
    for j, feat in enumerate(names):
        col = X_all[:, j]
        lo, hi = np.percentile(col, [1.0, 99.0])
        edges = np.linspace(float(lo), float(hi), n_bins + 1, dtype=np.float32)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.zeros(n_bins, dtype=np.int32)
        masks: List[np.ndarray] = []
        for b in range(n_bins):
            if b < n_bins - 1:
                m = (col >= edges[b]) & (col < edges[b + 1])
            else:
                m = (col >= edges[b]) & (col <= edges[b + 1])
            counts[b] = int(m.sum())
            masks.append(m)
        bin_masks[feat] = masks
        arrays[f"bin_edges_{feat}"] = edges
        arrays[f"bin_centers_{feat}"] = centers
        arrays[f"bin_counts_{feat}"] = counts

    for task in TASKS:
        r_true = ref_rewards_by_task[task].reshape(-1).astype(np.float32)
        seed_preds = []
        for md in models_dirs:
            model, _ = load_anfis(md / f"{task}_{tag}.pt")
            assert list(model.sensor_names) == names
            with torch.no_grad():
                seed_preds.append(model(X_t).squeeze(-1).numpy().astype(np.float32))
        preds = np.stack(seed_preds, axis=0)  # [n_seeds, N]

        for feat in names:
            masks = bin_masks[feat]
            true_bin = np.full(n_bins, np.nan, dtype=np.float32)
            anf_mean = np.full(n_bins, np.nan, dtype=np.float32)
            anf_std = np.full(n_bins, np.nan, dtype=np.float32)
            for b, m in enumerate(masks):
                if int(m.sum()) < min_bin_count:
                    continue
                true_bin[b] = float(r_true[m].mean())
                per_seed_bin = preds[:, m].mean(axis=1)  # [n_seeds]
                anf_mean[b] = float(per_seed_bin.mean())
                anf_std[b] = float(per_seed_bin.std())
            arrays[f"bin_true_{task}_{feat}"] = true_bin
            arrays[f"bin_anfis_{task}_{feat}"] = anf_mean
            arrays[f"bin_anfis_std_{task}_{feat}"] = anf_std

    out = out_dir / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


def plot_ref_binned(grid, out_path: Path) -> None:
    """Plot real reference-pool binned marginal means (10-seed mean +- 1 std) with sample-density histograms."""
    names = [str(n) for n in grid["features"]]
    n_seeds = int(grid["n_seeds"]) if "n_seeds" in grid else 1
    n_samples = int(grid["n_samples"]) if "n_samples" in grid else 10000
    labels = t6_compare.FEATURE_LABELS + names[4:]

    fig, axes = plt.subplots(len(TASKS), len(names), figsize=(3.3 * len(names), 9.5), sharey="row")
    for r, task in enumerate(TASKS):
        for c, (feat, label) in enumerate(zip(names, labels)):
            ax = axes[r, c]
            centers = grid[f"bin_centers_{feat}"]
            edges = grid[f"bin_edges_{feat}"]
            counts = grid[f"bin_counts_{feat}"].astype(np.float64)
            y_true = grid[f"bin_true_{task}_{feat}"]
            y_mean = grid[f"bin_anfis_{task}_{feat}"]
            y_std = grid[f"bin_anfis_std_{task}_{feat}"]

            # Background histogram of real reference-pool sample density
            ax_hist = ax.twinx()
            widths = np.diff(edges)
            ax_hist.bar(centers, counts / max(counts.max(), 1.0), width=widths * 0.92, color="0.75", alpha=0.25, zorder=0)
            ax_hist.set_ylim(0.0, 2.5)
            ax_hist.set_yticks([])
            ax_hist.spines["right"].set_visible(False)
            ax_hist.spines["top"].set_visible(False)

            ax.set_zorder(ax_hist.get_zorder() + 1)
            ax.patch.set_visible(False)

            valid = np.isfinite(y_mean)
            ax.axhline(0.0, color="0.65", lw=0.8, ls=":")
            ax.plot(centers[valid], y_true[valid], color="#4C72B0", lw=2.0, ls="--", label="Original r(s)")
            ax.plot(
                centers[valid],
                y_mean[valid],
                color="#DD8452",
                lw=2.2,
                label=f"ANFIS r~(s) ({n_seeds}-seed mean)",
            )
            if n_seeds > 1:
                ax.fill_between(
                    centers[valid],
                    (y_mean - y_std)[valid],
                    (y_mean + y_std)[valid],
                    color="#DD8452",
                    alpha=0.22,
                    label="+- 1 std",
                )
            ax.fill_between(
                centers[valid],
                y_mean[valid],
                0.0,
                where=(y_mean[valid] < 0),
                color="#C44E52",
                alpha=0.14,
            )
            if r == 0:
                ax.set_title(label, fontsize=9, fontweight="bold", color="#C44E52" if c >= 4 else "black")
            if c == 0:
                ax.set_ylabel(f"{task}\nreward / weight", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.25)

    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=True)
    extra_note = "; extra sensors in red" if len(names) > 4 else ""
    fig.suptitle(
        f"Real Reference Pool (N={n_samples}) Binned Mean: Original r(s) vs. ANFIS r~(s) "
        f"(sep-CMA-ES 800k, {n_seeds}-seed mean +- 1 std{extra_note}; gray bars = sample density)",
        fontsize=10.5,
        y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def discover_artifact_dirs(artifacts_root: Path, out_prefix: str) -> List[Path]:
    """Find all ``<artifacts_root>/zero-shot-rl-*/models`` dirs whose run.log wrote ``out/<out_prefix>_ft*/table6.csv``."""
    found: List[Path] = []
    for d in sorted(artifacts_root.glob("zero-shot-rl-*")):
        log = d / "run.log"
        models = d / "models"
        if not log.exists() or not (models / "stand_final.pt").exists():
            continue
        txt = log.read_text(errors="ignore")
        if f"Wrote out/{out_prefix}_ft" in txt:
            found.append(models)
    return found


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract_multi",
        nargs="+",
        metavar="NAME=DIR1,DIR2,...",
        default=None,
        help="Explicit comma-separated model directories per named grid",
    )
    parser.add_argument(
        "--extract_from_artifacts",
        type=Path,
        default=None,
        help="Auto-discover all 10 seeds of cmaes_anfis and cmaes_anfis_custom under this artifacts directory",
    )
    parser.add_argument(
        "--extract_ref_binned",
        type=Path,
        default=None,
        help="Artifacts dir for extracting real reference-pool binned means (requires --dataset_path)",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Offline Walker RND dataset (.npz) used with --extract_ref_binned",
    )
    parser.add_argument("--tag", type=str, default="final", choices=["final", "best"])
    args = parser.parse_args()

    if args.extract_ref_binned is not None:
        if args.dataset_path is None:
            raise SystemExit("--dataset_path is required with --extract_ref_binned")
        _, feats_t, rewards_dict, _ = collect_reference_set(
            dataset_path=args.dataset_path,
            domain="walker",
            sensor_names=ALL_WALKER_SENSORS,
            tasks=TASKS,
            dataset_transitions=100000,
            z_inference_steps=10000,
            seed=42,
            discount=0.98,
            device=torch.device("cpu"),
        )
        assert feats_t is not None
        feats_np = feats_t.numpy()
        ref_feats_by_name = {s: feats_np[:, i] for i, s in enumerate(ALL_WALKER_SENSORS)}
        ref_rewards_by_task = {t: rewards_dict[t].numpy() for t in TASKS}

        envs = ("friction_mild", "friction_high_mild", "actuator_mild", "actuator_high_mild")
        for env in envs:
            for suffix, cond in (("", "cmaes_anfis"), ("_custom", "cmaes_anfis_custom")):
                name = f"{env}{suffix}"
                dirs = discover_artifact_dirs(args.extract_ref_binned, f"{cond}_{env}")
                if not dirs:
                    raise SystemExit(f"No artifact dirs found for {cond}_{env} under {args.extract_ref_binned}")
                out = extract_ref_binned(name, dirs, ref_feats_by_name, ref_rewards_by_task, tag=args.tag)
                print(f"Wrote {out} ({len(dirs)} seeds)")
        return

    if args.extract_from_artifacts is not None:
        envs = ("friction_mild", "friction_high_mild", "actuator_mild", "actuator_high_mild")
        for env in envs:
            for suffix, cond in (("", "cmaes_anfis"), ("_custom", "cmaes_anfis_custom")):
                name = f"{env}{suffix}"
                dirs = discover_artifact_dirs(args.extract_from_artifacts, f"{cond}_{env}")
                if not dirs:
                    raise SystemExit(f"No artifact dirs found for {cond}_{env} under {args.extract_from_artifacts}")
                out = extract_multi(name, dirs, tag=args.tag)
                print(f"Wrote {out} ({len(dirs)} seeds)")
        return

    if args.extract_multi:
        for item in args.extract_multi:
            name, raw_dirs = item.split("=", 1)
            dirs = [Path(p) for p in raw_dirs.split(",") if p]
            out = extract_multi(name, dirs, tag=args.tag)
            print(f"Wrote {out} ({len(dirs)} seeds)")
        return

    for npz in sorted(GRID_DIR.glob("*.npz")):
        grid = np.load(npz)
        out_dir = FIG_DIR / npz.stem
        plot_slices_multi(grid, out_dir / "reward_slices.png")
        t6_compare.plot_heatmaps(grid, out_dir / "reward_heatmap.png")
        print(f"Wrote slice/heatmap figures in {out_dir} ({int(grid['n_seeds']) if 'n_seeds' in grid else 1} seeds)")

    if REF_BIN_DIR.exists():
        for npz in sorted(REF_BIN_DIR.glob("*.npz")):
            grid = np.load(npz)
            out_dir = FIG_DIR / npz.stem
            plot_ref_binned(grid, out_dir / "ref_binned_slices.png")
            print(f"Wrote ref-binned figure in {out_dir} ({int(grid['n_seeds']) if 'n_seeds' in grid else 1} seeds)")


if __name__ == "__main__":
    main()
