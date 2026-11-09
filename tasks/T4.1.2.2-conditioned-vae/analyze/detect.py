"""Detection of dynamics shifts by the CVAE ensembles of T4.1.2.2.

Reads data/inputs/cvae_<run>.npz, shift_gap.npz and fit_info.json of analyze.pack, one run per
setting of the KL weight beta (`beta<value>`) and `learn_sigma` (learned decoder std, beta 1).
For each run, with r_i the reconstruction MSE of member i of zero_shot_rl.latent.cvae_model
(`mean` is its mean over members, `min` its minimum), it writes

    data/detect.csv    run, score (mean, min), config, auroc, lo, hi: P(score_config > score_stock) over
                       all pairs of test samples, with the 95% interval of resampling the 60 test
                       episodes (paired, 300 draws)
    data/icp.csv       run, delta (0.95, 0.99), config, thr, tpr, fpr: the ICP detector of the paper,
                       OOD iff min_i r_i >= the element floor(delta M N) of the pooled calibration set
                       (the baseline validation errors of all members); tpr and fpr are the OOD
                       fractions of the shifted and of the stock test samples
    data/fit.csv       run, and the mean over members of the best epoch, the validation
                       reconstruction (squared error summed over the standardised dimensions) and KL
                       at it, the number of dimensions of z with KL > 0.01, and the decoder std
    data/compare.csv   config, the `norm` AUROC of the MLPs of T4.1.2.1 (regressing Delta s or s'),
                       and the `min` AUROC of every run

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.2.2-conditioned-vae uv run python -m analyze.detect [--check]
"""

import json
import sys
from pathlib import Path

import numpy as np

from analysis.detection import auroc
from zero_shot_rl.latent.cvae_model import icp_threshold

TASK = Path(__file__).resolve().parents[1]
MLP_TASK = TASK.parent / "T4.1.2.1-action-conditioned-detection"
CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
RUNS = ["beta0", "beta0.0001", "beta0.001", "beta0.01", "beta0.03", "beta0.1", "beta1", "beta4", "learn_sigma"]
SCORES = ["mean", "min"]
DELTAS = [0.95, 0.99]
DRAWS = 300
DETECT = ["run", "score", "config", "auroc", "lo", "hi"]
ICP = ["run", "delta", "config", "thr", "tpr", "fpr"]
FIT = ["run", "best_epoch", "val_rec", "val_kl", "active_dims", "sigma"]


def detect_rows(run, data):
    rng = np.random.default_rng(0)
    n_ep = len(data["episode"])
    draws = [rng.integers(n_ep, size=n_ep) for _ in range(DRAWS)]
    out = []
    for score in SCORES:
        base = data[f"{score}_rollout"]
        for c in CONFIGS:
            cur = data[f"{score}_{c}"]
            boot = np.quantile([auroc(base[d].ravel(), cur[d].ravel()) for d in draws], [0.025, 0.975])
            out.append({"run": run, "score": score, "config": c, "auroc": auroc(base.ravel(), cur.ravel()),
                        "lo": boot[0], "hi": boot[1]})
    return out


def icp_rows(run, data):
    out = []
    for delta in DELTAS:
        thr = icp_threshold(data["val_member"], delta)
        fpr = (data["min_rollout"] >= thr).mean()
        for c in CONFIGS:
            out.append({"run": run, "delta": delta, "config": c, "thr": thr,
                        "tpr": (data[f"min_{c}"] >= thr).mean(), "fpr": fpr})
    return out


def fit_rows(fit):
    out = []
    for run in RUNS:
        members = fit[run]
        sigma = [m["sigma"] for m in members]
        out.append({"run": run, "best_epoch": np.mean([m["best_epoch"] for m in members]),
                    "val_rec": np.mean([m["val_rec"] for m in members]),
                    "val_kl": np.mean([m["val_kl"] for m in members]),
                    "active_dims": np.mean([m["active_dims"] for m in members]),
                    "sigma": "" if sigma[0] is None else np.mean(sigma)})
    return out


def compare_rows(detect):
    mlp = {}
    for target in ("delta", "next_state"):
        lines = (MLP_TASK / f"data/detect_{target}.csv").read_text().splitlines()
        head = lines[0].split(",")
        for line in lines[1:]:
            row = dict(zip(head, line.split(",")))
            if row["score"] == "norm":
                mlp[(target, row["config"])] = float(row["auroc"])
    out = []
    for c in CONFIGS:
        row = {"config": c, "mlp_delta": mlp[("delta", c)], "mlp_next_state": mlp[("next_state", c)]}
        for r in detect:
            if r["config"] == c and r["score"] == "min":
                row[r["run"]] = r["auroc"]
        out.append(row)
    return out


def render(fields, records):
    def cell(v):
        return v if isinstance(v, str) else f"{v:.6g}"
    return "\n".join([",".join(fields)] + [",".join(cell(r[k]) for k in fields) for r in records]) + "\n"


def tables():
    inputs = TASK / "data/inputs"
    detect, icp = [], []
    for run in RUNS:
        data = dict(np.load(inputs / f"cvae_{run}.npz"))
        detect += detect_rows(run, data)
        icp += icp_rows(run, data)
    fit = json.loads((inputs / "fit_info.json").read_text())
    compare_fields = ["config", "mlp_delta", "mlp_next_state"] + RUNS
    return {"detect.csv": render(DETECT, detect), "icp.csv": render(ICP, icp),
            "fit.csv": render(FIT, fit_rows(fit)), "compare.csv": render(compare_fields, compare_rows(detect))}


def main():
    check = "--check" in sys.argv
    for name, text in tables().items():
        path = TASK / "data" / name
        if check:
            if path.read_text() != text:
                sys.exit(f"{path} differs from the regenerated table")
        else:
            path.write_text(text)
    print("ok" if check else "written")


if __name__ == "__main__":
    main()
