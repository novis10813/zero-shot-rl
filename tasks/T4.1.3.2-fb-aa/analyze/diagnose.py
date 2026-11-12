"""Tables of the K-step replay and of the backward features (data/inputs/horizon.npz, b0.npz of analyze.pack).

Writes three tables, one row per config:

data/horizon.csv    k and, for each config, the median of the open- and closed-loop gap
                    |s_k^config - s_k^stock|_2 (columns open_<config>, closed_<config>), and the
                    closed-loop floor of floating point (column floor)
data/multistep.csv  config, k, the median k-step prediction error of the dynamics model on the
                    closed-loop trajectories of the config and of the stock simulator (err,
                    err_stock), the AUROC of that error (auc_multi) and of the mean one-step error
                    over the first k steps (auc_onestep), stock vs config
data/b0.csv         config, and over the transitions the shift changes (|s'_config - s'_stock| > 1e-6):
                    ratio_s, ratio_b   median of |x(s'_config) - x(s'_stock)| / |x(s'_stock) - x(s)|,
                                       x the observation or B
                    keep_step          ratio_b / ratio_s
                    keep_scale         (median b_gap / b_scale) / (median s_gap / s_scale), the
                                       gaps relative to the median distance of random stock pairs
                    step_stock, step_config, auc_step   over all transitions, the median of
                                       |B(s') - B(s)| for stock and config, and the AUROC
                                       P(config > stock)

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.diagnose [--check]
"""

import sys
from pathlib import Path

import numpy as np

from analysis.detection import auroc

TASK = Path(__file__).resolve().parents[1]
CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
ACTIVE = 1e-6


def fmt(v):
    return v if isinstance(v, str) else f"{v:.6g}"


def render(fields, records):
    return "\n".join([",".join(fields)] + [",".join(fmt(r[k]) for k in fields) for r in records]) + "\n"


def horizon_table(h):
    fields = ["k"] + [f"{m}_{c}" for c in CONFIGS for m in ("open", "closed")] + ["floor"]
    records = []
    for j, k in enumerate(h["ks"]):
        r = {"k": int(k), "floor": np.median(h["closed_floor"][:, j])}
        for c in CONFIGS:
            r[f"open_{c}"] = np.median(h[f"open_gap_{c}"][:, j])
            r[f"closed_{c}"] = np.median(h[f"closed_gap_{c}"][:, j])
        records.append(r)
    return render(fields, records)


def multistep_table(h):
    fields = ["config", "k", "err", "err_stock", "auc_multi", "auc_onestep"]
    records = []
    for c in CONFIGS:
        for j, k in enumerate(h["ks"]):
            records.append(dict(
                config=c, k=int(k), err=np.median(h[f"multi_{c}"][:, j]), err_stock=np.median(h["multi_stock"][:, j]),
                auc_multi=auroc(h["multi_stock"][:, j], h[f"multi_{c}"][:, j]),
                auc_onestep=auroc(h["onestep_stock"][:, j], h[f"onestep_{c}"][:, j])))
    return render(fields, records)


def b0_table(b):
    fields = ["config", "ratio_s", "ratio_b", "keep_step", "keep_scale", "step_stock", "step_config", "auc_step"]
    s_scale, b_scale = b["s_scale"].mean(), b["b_scale"].mean()
    records = []
    for c in CONFIGS:
        act = b[f"s_gap_{c}"] > ACTIVE
        ratio_s = np.median(b[f"s_gap_{c}"][act] / b["s_step_stock"][act])
        ratio_b = np.median(b[f"b_gap_{c}"][act] / b["b_step_stock"][act])
        keep_scale = (np.median(b[f"b_gap_{c}"][act]) / b_scale) / (np.median(b[f"s_gap_{c}"][act]) / s_scale)
        records.append(dict(config=c, ratio_s=ratio_s, ratio_b=ratio_b, keep_step=ratio_b / ratio_s,
                            keep_scale=keep_scale, step_stock=np.median(b["b_step_stock"]),
                            step_config=np.median(b[f"b_step_{c}"]),
                            auc_step=auroc(b["b_step_stock"], b[f"b_step_{c}"])))
    return render(fields, records)


def main():
    h, b = np.load(TASK / "data/inputs/horizon.npz"), np.load(TASK / "data/inputs/b0.npz")
    tables = {"horizon.csv": horizon_table(h), "multistep.csv": multistep_table(h), "b0.csv": b0_table(b)}
    check = "--check" in sys.argv
    for name, text in tables.items():
        path = TASK / "data" / name
        if check:
            if path.read_text() != text:
                sys.exit(f"{path} differs from the regenerated table")
        else:
            path.write_text(text)
    print("ok" if check else "written")


if __name__ == "__main__":
    main()
