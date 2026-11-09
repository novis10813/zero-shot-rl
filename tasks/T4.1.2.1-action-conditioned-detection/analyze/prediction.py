"""Prediction quality of the dynamics models on the stock test transitions.

Reads data/inputs/dynamics_scored_<run>.npz (runs of analyze.detect) and writes

    data/prediction.csv       one row per run: R2 pooled over the 24 dimensions
                              (1 - sum of squared errors / sum of squared deviations of Delta s from
                              its mean), MSE per element and per transition, percentiles of |e|_2 of
                              the test transitions and the median |e|_2 of the validation transitions
    data/prediction_dims.csv  one row per dimension: R2_j of every run,
                              1 - sum_t (Delta s_tj - f_j)^2 / sum_t (Delta s_tj - mean_j)^2

The sums run over the stock rollout transitions of all 60 test episodes. The variance of Delta s
is that of each training seed's file, averaged over the files (as stored by dynamics_score).

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.2.1-action-conditioned-detection uv run python -m analyze.prediction [--check]
"""

import sys
from pathlib import Path

import numpy as np

from analyze.detect import RUNS, TASK

PERCENTILES = [50, 90, 95, 99]
FIELDS = (["run", "r2", "mse_per_element", "mse_per_transition"] + [f"norm_p{p}" for p in PERCENTILES]
          + ["norm_mean", "val_norm_median"])


def r2_dims(mse, var):
    """R2_j from the mean squared error and the variance of Delta s of every dimension."""
    return 1 - mse / var


def summary(data):
    mse, var, norm = data["mse_rollout"], data["var_delta"], data["norm_rollout"]
    row = {"r2": 1 - mse.sum() / var.sum(), "mse_per_element": mse.mean(), "mse_per_transition": mse.sum(),
           "norm_mean": norm.mean(), "val_norm_median": np.median(data["val_norm"])}
    row.update({f"norm_p{p}": np.percentile(norm, p) for p in PERCENTILES})
    return row


def render(rows, fields):
    lines = [",".join(fields)]
    for r in rows:
        lines.append(",".join(str(r[k]) if k == "run" else f"{r[k]:.6g}" for k in fields))
    return "\n".join(lines) + "\n"


def tables():
    data = {run: np.load(TASK / f"data/inputs/dynamics_scored_{run}.npz") for run in RUNS}
    overall = render([{"run": run, **summary(d)} for run, d in data.items()], FIELDS)
    fields = ["dim"] + [f"r2_{run}" for run in RUNS]
    dims = render([{"dim": j, **{f"r2_{run}": r2_dims(d["mse_rollout"], d["var_delta"])[j] for run, d in data.items()}}
                   for j in range(24)], fields)
    return {"prediction.csv": overall, "prediction_dims.csv": dims}


def main():
    check = "--check" in sys.argv
    for name, text in tables().items():
        path = Path(TASK / "data" / name)
        if check:
            if path.read_text() != text:
                sys.exit(f"{path} differs from the regenerated table")
        else:
            path.write_text(text)
    print("ok" if check else "written")


if __name__ == "__main__":
    main()
