"""Detection of dynamics shifts by the action-conditioned dynamics model.

Reads data/inputs/dynamics_scored.npz of zero_shot_rl.latent.dynamics_score and writes
data/detect.csv, one row per (score, config), where score is `norm` (|e|_2) or `maha` (D_dyn):

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

TASK = Path(__file__).resolve().parents[1]
CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
SCORES = ["norm", "maha"]
PERCENTILES = [95, 99]
DRAWS = 300
FIELDS = (["score", "config", "state_gap", "rel_mse", "stock", "config_median", "auroc", "lo", "hi"]
          + [f"{k}_p{p}" for p in PERCENTILES for k in ("thr", "tpr", "fpr")]
          + ["thr_j", "tpr_j", "fpr_j"])


def auroc(negative, positive):
    """P(positive > negative) with ties counted half."""
    neg = np.sort(negative)
    below = np.searchsorted(neg, positive, side="left")
    above = np.searchsorted(neg, positive, side="right")
    return ((below + above) / 2).mean() / len(neg)


def best_youden(negative, positive):
    """Threshold maximising TPR - FPR of `score > threshold`, with its TPR and FPR."""
    neg, pos = np.sort(negative), np.sort(positive)
    thr = np.unique(np.concatenate([neg, pos]))
    tpr = 1 - np.searchsorted(pos, thr, side="right") / len(pos)
    fpr = 1 - np.searchsorted(neg, thr, side="right") / len(neg)
    i = (tpr - fpr).argmax()
    return thr[i], tpr[i], fpr[i]


def rows():
    data = np.load(TASK / "data/inputs/dynamics_scored.npz")
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
    text = render(rows())
    path = TASK / "data/detect.csv"
    if "--check" in sys.argv:
        if path.read_text() != text:
            sys.exit(f"{path} differs from the regenerated table")
        print("ok")
    else:
        path.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
