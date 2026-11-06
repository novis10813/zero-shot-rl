"""Mean and covariance of the centered residual r = e - mu_0 under shifted dynamics, from the
scored paired transitions of zero_shot_rl.eval.paired_score.

For the same (s, a, z) samples, with C_0 the covariance of r in the stock dynamics, P =
(C_0 + 1e-6 I)^{-1}, and m, C the mean difference and covariance of r in a shifted config:

    mean_shift   m^T P m              mean change in units of the stock spread
    cov_scale    tr(P C) / d          mean variance ratio over the whitened directions
    logdet_diff  log det C - log det C_0   change of the log volume of the residual cloud

Writes data/moments.csv (one row per detector, seed, config) and data/moments_summary.csv
(mean, min and max over the three seeds).

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.2-dynamic-shift-detection uv run python -m analyze.moments [--check]
"""

import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
DETECTORS = ["dataset", "rollout"]
METRICS = ["mean_shift", "cov_scale", "logdet_diff"]
EPS = 1e-6


def rows():
    out = []
    for detector in DETECTORS:
        for seed in range(3):
            data = np.load(TASK / f"data/inputs/scored_{detector}_s{seed}.npz")
            c0 = data["cov_rollout"]
            d = len(c0)
            precision = np.linalg.inv(c0 + EPS * np.eye(d))
            logdet0 = np.linalg.slogdet(c0)[1]
            for config in CONFIGS:
                m = data[f"mean_{config}"] - data["mean_rollout"]
                c = data[f"cov_{config}"]
                out.append({
                    "detector": detector, "seed": seed, "config": config,
                    "mean_shift": m @ precision @ m, "cov_scale": np.trace(precision @ c) / d,
                    "logdet_diff": np.linalg.slogdet(c)[1] - logdet0,
                })
    return out


def render(records):
    fields = ["detector", "seed", "config"] + METRICS
    lines = [",".join(fields)]
    for r in records:
        lines.append(",".join(str(r[k]) if k in fields[:3] else f"{r[k]:.6g}" for k in fields))
    summary = ["detector,config," + ",".join(f"{m}_{s}" for m in METRICS for s in ("mean", "min", "max"))]
    for detector in DETECTORS:
        for config in CONFIGS:
            sel = [r for r in records if r["detector"] == detector and r["config"] == config]
            cells = []
            for m in METRICS:
                v = np.array([r[m] for r in sel])
                cells += [f"{v.mean():.6g}", f"{v.min():.6g}", f"{v.max():.6g}"]
            summary.append(f"{detector},{config}," + ",".join(cells))
    return "\n".join(lines) + "\n", "\n".join(summary) + "\n"


def main():
    per_seed, summary = render(rows())
    paths = (TASK / "data/moments.csv", TASK / "data/moments_summary.csv")
    if "--check" in sys.argv:
        if any(p.read_text() != t for p, t in zip(paths, (per_seed, summary))):
            sys.exit("data/moments*.csv differ from the regenerated tables")
        print("ok")
    else:
        for p, t in zip(paths, (per_seed, summary)):
            p.write_text(t)
        print(summary)


if __name__ == "__main__":
    main()
