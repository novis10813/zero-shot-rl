"""Plot the binned implied rewards r_z from `data/z_reward_bins.npz` (written by `analyze.z_reward`).

One figure per task: rows are OOD environments, columns are the 4 sensors. Each panel shows the true
task reward, r_{z_r} of the zero-shot latent, and r_z of the latents found by sep-CMA-ES on 4-sensor
ANFIS and by CMA-ES on direct z (mean over seeds, shaded +-1 std), averaged in quantile bins of the
sensor over the reference states. Grey bars show the share of reference states per bin.

Usage (repo root):
    PYTHONPATH=.:tasks/T6.3-sensors uv run python -m analyze.z_reward_figs
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {"cmaes_anfis": "sep-CMA-ES ANFIS 4 sensors", "cmaes_z": "CMA-ES direct z"}
COLORS = {"cmaes_anfis": "tab:blue", "cmaes_z": "tab:orange"}


def plot_task(d, ti: int, out_path: Path) -> None:
    sensors, envs, conditions = d["sensors"], d["envs"], d["conditions"]
    fig, axes = plt.subplots(len(envs), len(sensors), figsize=(4 * len(sensors), 3 * len(envs)), squeeze=False)
    for ei, env in enumerate(envs):
        for xi, sensor in enumerate(sensors):
            ax = axes[ei, xi]
            x = d["centers"][xi]
            share = d["counts"][xi] / d["counts"][xi].sum()
            bar = ax.twinx()
            bar.bar(x, share, width=x[1] - x[0], color="0.85", zorder=0)
            bar.set_yticks([])
            ax.set_zorder(bar.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.plot(x, d["true"][xi, :, ti], color="black", lw=2, label="true reward")
            ax.plot(x, d["z_r"][xi, :, ti], color="0.4", ls="--", label="$r_{z_r}$ (zero-shot)")
            for ci, cond in enumerate(conditions):
                vals = d["found"][xi, :, ci, ei, ti, :]
                mean, std = np.nanmean(vals, axis=1), np.nanstd(vals, axis=1)
                ax.plot(x, mean, color=COLORS[str(cond)], label=f"$r_z$ {LABELS[str(cond)]}")
                ax.fill_between(x, mean - std, mean + std, color=COLORS[str(cond)], alpha=0.2)
            if ei == 0:
                ax.set_title(str(sensor))
            if xi == 0:
                ax.set_ylabel(str(env))
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{d['tasks'][ti]}: implied reward $r_z(s) = B(s)^T \\Sigma_B^{{-1}} z$ on reference states")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bins", type=Path, default=Path("tasks/T6.3-sensors/data/z_reward_bins.npz"))
    parser.add_argument("--out_dir", type=Path, default=Path("tasks/T6.3-sensors/figures/z_reward"))
    args = parser.parse_args()
    d = np.load(args.bins)
    for ti, task in enumerate(d["tasks"]):
        plot_task(d, ti, args.out_dir / f"{task}.png")
    print(f"wrote {len(d['tasks'])} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
