"""Generate clean score tables and direct reward-function comparisons for T6-anfis.

Modes:
    1. Extract reward grids from saved ANFIS checkpoints (one-time, writes data/inputs/reward_grid.npz):
       PYTHONPATH=.:tasks/T6-anfis uv run python -m analyze.compare --extract <models_dir>

    2. Regenerate tables and figures from data/inputs/ only (no checkpoints needed):
       PYTHONPATH=.:tasks/T6-anfis uv run python -m analyze.compare [--check]
"""

import csv
import io
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dm_control.utils import rewards as dm_rewards

from analysis.table6_aggregate import TASKS, aggregate
from zero_shot_rl.latent.anfis import load_anfis

TASK_DIR = Path(__file__).resolve().parents[1]
INPUTS = TASK_DIR / "data" / "inputs"
DATA_DIR = TASK_DIR / "data"
FIG_DIR = TASK_DIR / "figures"

FEATURES = ["torso_height", "torso_upright", "horizontal_velocity", "angmomentum"]
FEATURE_LABELS = ["torso_height (m)", "torso_upright", "CoM_vx (m/s)", "angmom_y"]

# Reference posture held fixed when sweeping one feature in 1D slices.
SLICE_BASE = {
    "stand": [1.25, 1.0, 0.0, 0.0],
    "walk":  [1.25, 1.0, 1.5, 0.0],
    "run":   [1.25, 1.0, 4.0, 0.0],
    "flip":  [1.20, 1.0, 0.5, 5.0],
}

SLICE_RANGES = {
    "torso_height": (0.1, 1.45),
    "torso_upright": (-1.0, 1.0),
    "horizontal_velocity": (-2.5, 8.5),
    "angmomentum": (-10.0, 15.0),
}

# Primary (x_axis, y_axis) feature indices for 2D heatmaps per task.
HEATMAP_AXES = {
    "stand": (0, 1),  # height vs upright
    "walk":  (0, 2),  # height vs CoM_vx
    "run":   (0, 2),  # height vs CoM_vx
    "flip":  (0, 3),  # height vs angmom_y
}

PROBES = [
    ("fallen",     "跌倒 (h=0.2, up=-0.5, vx=0, ang=0)",      [0.20, -0.5,  0.0,  0.0]),
    ("low_crouch", "低蹲 (h=0.7, up=0.5, vx=0, ang=0)",       [0.70,  0.5,  0.0,  0.0]),
    ("stand",      "站立 (h=1.25, up=1.0, vx=0, ang=0)",      [1.25,  1.0,  0.0,  0.0]),
    ("walk_fwd",   "前走 (h=1.25, up=1.0, vx=1.5, ang=0)",    [1.25,  1.0,  1.5,  0.0]),
    ("fast_run",   "快跑 (h=1.25, up=1.0, vx=4.0, ang=0)",    [1.25,  1.0,  4.0,  0.0]),
    ("walk_bwd",   "後退 (h=1.25, up=1.0, vx=-1.5, ang=0)",   [1.25,  1.0, -1.5,  0.0]),
    ("fwd_flip",   "前空翻 (h=1.0, up=0.0, vx=0.5, ang=5)",   [1.00,  0.0,  0.5,  5.0]),
    ("bwd_spin",   "反向旋轉 (h=1.0, up=0.0, vx=0, ang=-5)",  [1.00,  0.0,  0.0, -5.0]),
]


def true_reward(task: str, x: np.ndarray) -> np.ndarray:
    """Exact DMC Walker reward evaluated on feature array x [..., 4]."""
    h, up, vx, ang = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    standing = dm_rewards.tolerance(h, bounds=(1.2, float("inf")), margin=0.6)
    upright = (1.0 + up) / 2.0
    sr = (3.0 * standing + upright) / 4.0
    if task == "stand":
        return sr
    if task == "walk":
        mv = dm_rewards.tolerance(vx, bounds=(1.0, float("inf")), margin=0.5, value_at_margin=0.5, sigmoid="linear")
        return sr * (5.0 * mv + 1.0) / 6.0
    if task == "run":
        mv = dm_rewards.tolerance(vx, bounds=(8.0, float("inf")), margin=4.0, value_at_margin=0.5, sigmoid="linear")
        return sr * (5.0 * mv + 1.0) / 6.0
    if task == "flip":
        mv = dm_rewards.tolerance(ang, bounds=(5.0, float("inf")), margin=5.0, value_at_margin=0.0, sigmoid="linear")
        return sr * (5.0 * mv + 1.0) / 6.0
    raise ValueError(f"unknown task {task!r}")


def extract_reward_grid(models_dir: Path, out_path: Path, n_slice: int = 200, n_grid: int = 80) -> None:
    """Evaluate original r(s) and saved ANFIS models on probes, 1D slices, and 2D grids."""
    arrays = {
        "tasks": np.array(TASKS),
        "features": np.array(FEATURES),
        "probe_ids": np.array([p[0] for p in PROBES]),
        "probe_labels": np.array([p[1] for p in PROBES]),
        "probe_x": np.array([p[2] for p in PROBES], dtype=np.float32),
    }
    probe_x = arrays["probe_x"]
    probe_t = torch.from_numpy(probe_x)

    for task in TASKS:
        model, _ = load_anfis(models_dir / f"{task}_best.pt")
        arrays[f"probe_true_{task}"] = true_reward(task, probe_x).astype(np.float32)
        with torch.no_grad():
            arrays[f"probe_anfis_{task}"] = model(probe_t).squeeze(-1).numpy().astype(np.float32)

        # 1D slices along each feature
        base = np.array(SLICE_BASE[task], dtype=np.float32)
        for j, feat in enumerate(FEATURES):
            lo, hi = SLICE_RANGES[feat]
            xs = np.linspace(lo, hi, n_slice, dtype=np.float32)
            pts = np.tile(base, (n_slice, 1))
            pts[:, j] = xs
            arrays[f"slice_x_{task}_{feat}"] = xs
            arrays[f"slice_true_{task}_{feat}"] = true_reward(task, pts).astype(np.float32)
            with torch.no_grad():
                arrays[f"slice_anfis_{task}_{feat}"] = model(torch.from_numpy(pts)).squeeze(-1).numpy().astype(np.float32)

        # 2D heatmap over primary axes
        ix, iy = HEATMAP_AXES[task]
        gx = np.linspace(*SLICE_RANGES[FEATURES[ix]], n_grid, dtype=np.float32)
        gy = np.linspace(*SLICE_RANGES[FEATURES[iy]], n_grid, dtype=np.float32)
        xx, yy = np.meshgrid(gx, gy)  # [n_grid, n_grid]
        pts = np.tile(base, (n_grid * n_grid, 1))
        pts[:, ix] = xx.ravel()
        pts[:, iy] = yy.ravel()
        arrays[f"grid_x_{task}"] = gx
        arrays[f"grid_y_{task}"] = gy
        arrays[f"grid_true_{task}"] = true_reward(task, pts).reshape(n_grid, n_grid).astype(np.float32)
        with torch.no_grad():
            arrays[f"grid_anfis_{task}"] = (
                model(torch.from_numpy(pts)).squeeze(-1).numpy().reshape(n_grid, n_grid).astype(np.float32)
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)


def build_scores_csv(baseline_csv: Path, opt_csv: Path) -> str:
    base_df = pd.read_csv(baseline_csv).set_index("step")
    opt_df = pd.read_csv(opt_csv).set_index("step")

    base_scores = base_df[TASKS].to_numpy()[None]  # [1, 1, 4]
    opt_scores = opt_df[TASKS].to_numpy()[None]    # [1, S, 4]
    base_iqm = float(aggregate(base_scores)[0])
    opt_iqms = aggregate(opt_scores)
    best_idx = int(np.argmax(opt_iqms))
    best_step = int(opt_df.index[best_idx])

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["task", "baseline_zr", "anfis_z", "delta", "selected_step"],
                            lineterminator="\n")
    writer.writeheader()
    for i, task in enumerate(TASKS):
        b = float(base_scores[0, 0, i])
        o = float(opt_scores[0, best_idx, i])
        writer.writerow({
            "task": task,
            "baseline_zr": f"{b:.1f}",
            "anfis_z": f"{o:.1f}",
            "delta": f"{o - b:+.1f}",
            "selected_step": best_step,
        })
    b_mean = float(base_scores[0, 0].mean())
    o_mean = float(opt_scores[0, best_idx].mean())
    writer.writerow({
        "task": "mean",
        "baseline_zr": f"{b_mean:.1f}",
        "anfis_z": f"{o_mean:.1f}",
        "delta": f"{o_mean - b_mean:+.1f}",
        "selected_step": best_step,
    })
    writer.writerow({
        "task": "all_task_iqm",
        "baseline_zr": f"{base_iqm:.1f}",
        "anfis_z": f"{float(opt_iqms[best_idx]):.1f}",
        "delta": f"{float(opt_iqms[best_idx]) - base_iqm:+.1f}",
        "selected_step": best_step,
    })
    return buf.getvalue()


def build_probes_csv(grid) -> str:
    probe_ids = list(grid["probe_ids"])
    probe_labels = list(grid["probe_labels"])
    fields = ["posture_id", "posture"]
    for task in TASKS:
        fields += [f"{task}_orig", f"{task}_anfis", f"{task}_diff"]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for i, (pid, label) in enumerate(zip(probe_ids, probe_labels)):
        row = {"posture_id": pid, "posture": label}
        for task in TASKS:
            orig = float(grid[f"probe_true_{task}"][i])
            anf = float(grid[f"probe_anfis_{task}"][i])
            row[f"{task}_orig"] = f"{orig:.2f}"
            row[f"{task}_anfis"] = f"{anf:+.2f}"
            row[f"{task}_diff"] = f"{anf - orig:+.2f}"
        writer.writerow(row)
    return buf.getvalue()


def plot_slices(grid, out_path: Path) -> None:
    fig, axes = plt.subplots(len(TASKS), len(FEATURES), figsize=(13, 9.5), sharey="row")
    for r, task in enumerate(TASKS):
        for c, (feat, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
            ax = axes[r, c]
            x = grid[f"slice_x_{task}_{feat}"]
            y_orig = grid[f"slice_true_{task}_{feat}"]
            y_anf = grid[f"slice_anfis_{task}_{feat}"]

            ax.axhline(0.0, color="0.65", lw=0.8, ls=":")
            ax.plot(x, y_orig, color="#4C72B0", lw=2.0, ls="--", label="Original r(s)")
            ax.plot(x, y_anf, color="#DD8452", lw=2.2, label="ANFIS r~(s)")
            ax.fill_between(x, y_anf, 0.0, where=(y_anf < 0), color="#C44E52", alpha=0.18)

            if r == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{task}\nreward / weight", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8.5, frameon=True)
    fig.suptitle("Original Reward r(s) vs. Learned ANFIS Weight r~(s) (1D Physical Slices)", fontsize=12, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmaps(grid, out_path: Path) -> None:
    fig, axes = plt.subplots(len(TASKS), 3, figsize=(12, 11.5))
    col_titles = ["Original r(s)", "ANFIS r~(s)", "Difference: r~(s) - r(s)"]
    for r, task in enumerate(TASKS):
        ix, iy = HEATMAP_AXES[task]
        gx = grid[f"grid_x_{task}"]
        gy = grid[f"grid_y_{task}"]
        z_orig = grid[f"grid_true_{task}"]
        z_anf = grid[f"grid_anfis_{task}"]
        z_diff = z_anf - z_orig
        extent = [gx[0], gx[-1], gy[0], gy[-1]]

        vmin_r = min(float(z_orig.min()), float(z_anf.min()), -0.4)
        vmax_r = max(float(z_orig.max()), float(z_anf.max()), 1.0)
        vabs_d = max(float(np.abs(z_diff).max()), 0.5)

        im0 = axes[r, 0].imshow(z_orig, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
        im1 = axes[r, 1].imshow(z_anf, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r",
                                vmin=-max(abs(vmin_r), abs(vmax_r)), vmax=max(abs(vmin_r), abs(vmax_r)))
        im2 = axes[r, 2].imshow(z_diff, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r",
                                vmin=-vabs_d, vmax=vabs_d)

        # Draw zero-contour on ANFIS heatmap so negative region is immediately visible
        if z_anf.min() < 0 < z_anf.max():
            axes[r, 1].contour(gx, gy, z_anf, levels=[0.0], colors="black", linewidths=1.2, linestyles="--")

        for c, im in enumerate((im0, im1, im2)):
            ax = axes[r, c]
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10.5, fontweight="bold")
            ax.set_xlabel(FEATURE_LABELS[ix], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{task}\n{FEATURE_LABELS[iy]}", fontsize=9.5, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle("Direct 2D Comparison: Original Reward vs. ANFIS Learned Weight (dashed = 0 contour)", fontsize=12, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_or_check(path: Path, text: str, check: bool) -> None:
    if check:
        existing = path.read_text()
        if existing != text:
            raise AssertionError(f"Mismatch in {path}; regenerate without --check")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--extract", type=str, default=None, help="models/ dir with <task>_best.pt to write reward_grid.npz")
    parser.add_argument("--check", action="store_true", help="verify generated CSVs match stored files byte for byte")
    args = parser.parse_args()

    grid_path = INPUTS / "reward_grid.npz"
    if args.extract is not None:
        extract_reward_grid(Path(args.extract), grid_path)
        print(f"Wrote {grid_path}")
        return

    scores_text = build_scores_csv(INPUTS / "baseline_table6.csv", INPUTS / "table6.csv")
    write_or_check(DATA_DIR / "scores.csv", scores_text, args.check)

    grid = np.load(grid_path)
    probes_text = build_probes_csv(grid)
    write_or_check(DATA_DIR / "reward_probes.csv", probes_text, args.check)

    if not args.check:
        plot_slices(grid, FIG_DIR / "reward_slices.png")
        plot_heatmaps(grid, FIG_DIR / "reward_heatmap.png")
        print(f"Wrote {DATA_DIR / 'scores.csv'}, {DATA_DIR / 'reward_probes.csv'}, and figures in {FIG_DIR}")
    else:
        print("OK: scores.csv and reward_probes.csv match stored outputs byte for byte.")


if __name__ == "__main__":
    main()
