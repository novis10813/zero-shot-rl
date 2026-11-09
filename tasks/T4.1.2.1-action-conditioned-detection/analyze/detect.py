"""Detection of dynamics shifts by the action-conditioned dynamics model.

Reads data/inputs/dynamics_scored_<run>.npz of zero_shot_rl.latent.dynamics_score (run: `delta`,
the model regressing Delta s, or `next_state`, the one regressing s') and writes
data/detect_<run>.csv, one row per (score, config), where score is `norm` (|e|_2) or `maha` (D_dyn):

    state_gap       median |s'_config - s'_stock|  (L2 over the observation)
    rel_mse         sum of the MSE of the model on the config over the variance of Delta s of
                    the stock transitions; 1 is the error of predicting the mean displacement
    stock, config   median score of the stock and of the shifted transitions
    auroc, lo, hi   P(score_config > score_stock) over all pairs of test samples, with the 95%
                    interval of resampling the 60 test episodes (paired, 300 draws)
    thr_p95, tpr_p95, fpr_p95   threshold = 95th percentile of the baseline validation scores,
                    and the fractions of shifted and of stock test samples above it (p99 likewise)
    thr_j, tpr_j, fpr_j  the threshold maximising TPR - FPR on the test samples themselves (an
                    oracle: stock as negatives, config as positives), for reference only

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.2.1-action-conditioned-detection uv run python -m analyze.detect [--check]
"""

import sys
from pathlib import Path

import numpy as np

from analysis.detection import auroc, best_youden

TASK = Path(__file__).resolve().parents[1]
CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
RUNS = ["delta", "next_state"]
SCORES = ["norm", "maha"]
PERCENTILES = [95, 99]
DRAWS = 300
FIELDS = (["score", "config", "state_gap", "rel_mse", "stock", "config_median", "auroc", "lo", "hi"]
          + [f"{k}_p{p}" for p in PERCENTILES for k in ("thr", "tpr", "fpr")]
          + ["thr_j", "tpr_j", "fpr_j"])


def rows(run):
    data = np.load(TASK / f"data/inputs/dynamics_scored_{run}.npz")
    var_delta = data["var_delta"].sum()
    rng = np.random.default_rng(0)
    n_ep = len(data["episode"])
    draws = [rng.integers(n_ep, size=n_ep) for _ in range(DRAWS)]
    if not np.allclose(data["norm_baseline"], data["norm_rollout"], rtol=1e-4, atol=1e-6):
        raise ValueError("replayed baseline does not reproduce the rollout")
    out = []
    for score in SCORES:
        base = data[f"{score}_rollout"]
        val = data[f"val_{score}"]
        for c in CONFIGS:
            cur = data[f"{score}_{c}"]
            boot = np.quantile([auroc(base[d].ravel(), cur[d].ravel()) for d in draws], [0.025, 0.975])
            row = {"score": score, "config": c, "state_gap": np.median(data[f"gap_{c}"]),
                   "rel_mse": data[f"mse_{c}"].sum() / var_delta, "stock": np.median(base),
                   "config_median": np.median(cur), "auroc": auroc(base.ravel(), cur.ravel()),
                   "lo": boot[0], "hi": boot[1]}
            for p in PERCENTILES:
                thr = np.percentile(val, p)
                row.update({f"thr_p{p}": thr, f"tpr_p{p}": (cur > thr).mean(), f"fpr_p{p}": (base > thr).mean()})
            row["thr_j"], row["tpr_j"], row["fpr_j"] = best_youden(base.ravel(), cur.ravel())
            out.append(row)
    return out


def render(records):
    lines = [",".join(FIELDS)]
    for r in records:
        lines.append(",".join(str(r[k]) if k in ("score", "config") else f"{r[k]:.6g}" for k in FIELDS))
    return "\n".join(lines) + "\n"


def main():
    check = "--check" in sys.argv
    for run in RUNS:
        text = render(rows(run))
        path = TASK / f"data/detect_{run}.csv"
        if check:
            if path.read_text() != text:
                sys.exit(f"{path} differs from the regenerated table")
        else:
            path.write_text(text)
    print("ok" if check else "written")


if __name__ == "__main__":
    main()
