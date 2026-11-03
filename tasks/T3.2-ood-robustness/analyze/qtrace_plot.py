"""Time series of the ensemble Q of the action taken, with its spread, per dynamics config.

Reads the NPZ traces of zero_shot_rl.eval.qtrace (run_qtrace.sh). In those episodes the
actor at z_r drives, so the action taken at every step is a_opt. For each training seed
(column) and shift type (row) it plots, over the 1000 steps and averaged over the episodes,
the mean over ensemble members of Q(s, a_opt) as a solid line and Q +- u as dashed lines
(u is the ensemble std), for clean, mild and strong dynamics. Q and u are in units of
|z_hat_r| (zero_shot_rl.eval.switch).

Usage (repo root):
    PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.qtrace_plot
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASK = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2]
ROWS = {"Friction": ("friction_mild", "friction_strong"), "Actuator": ("actuator_mild", "actuator_strong")}
STYLES = [("baseline", "Clean", "tab:gray"), ("mild", "Mild", "tab:blue"), ("strong", "Strong", "tab:red")]


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=TASK / "data" / "inputs")
    parser.add_argument("--out", type=Path, default=TASK / "figures" / "qtrace_series.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(len(ROWS), len(SEEDS), figsize=(15, 7), sharex=True, sharey="col")
    for r, (shift, (mild, strong)) in enumerate(ROWS.items()):
        configs = {"baseline": "baseline", "mild": mild, "strong": strong}
        for c, seed in enumerate(SEEDS):
            ax = axes[r, c]
            for key, label, color in STYLES:
                d = np.load(args.input_dir / f"qtrace_s{seed}_{configs[key]}.npz")
                q, u = d["q_opt"].mean(0), d["u_opt"].mean(0)
                ax.plot(q, color=color, lw=1.6, label=f"{label} (n={len(d['episodes'])})")
                ax.plot(q + u, color=color, lw=0.9, ls="--")
                ax.plot(q - u, color=color, lw=0.9, ls="--")
            ax.set_title(f"{shift}, training seed {seed}")
            ax.grid(alpha=0.3)
            if r == len(ROWS) - 1:
                ax.set_xlabel("control step")
            if c == 0:
                ax.set_ylabel(r"$Q(s,a_{opt})$ (solid), $\pm u$ (dashed)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(STYLES))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)


if __name__ == "__main__":
    main()
