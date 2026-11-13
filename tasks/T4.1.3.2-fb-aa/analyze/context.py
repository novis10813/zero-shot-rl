"""Tables of the context encoder (data/inputs/context.npz of analyze.pack_context).

data/context_val.csv    encoder, seed, best_step (lowest val_loss) and, at the last step, the
                        validation InfoNCE accuracy (acc_h<H>, the right one of 31 validation
                        dynamics) and prediction loss per B_0 dimension (pred_h<H>)
data/context_auc.csv    encoder, tasks (all or walk), config, H: the AUROC P(config > stock) of
                        |c - c0_H| over all test windows, per FB seed (auc_s0..s2), their mean
                        and range
data/context_start.csv  encoder, config, H, start: the same over the windows that start at one
                        step, mean over seeds
data/context_err.csv    encoder, config (stock included), H, per seed the mean over episodes of
                        the error of g with the window's context (ctx_s<i>) and with the stock
                        context (stock_s<i>), and gain = mean over seeds of 1 - ctx / stock

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.context [--check]
"""

import sys
from pathlib import Path

import numpy as np

from analysis.detection import auroc

TASK = Path(__file__).resolve().parents[1]
NAMES = [
    "stock", "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
ENCODERS = ["transformer", "mean"]
SEEDS = [0, 1, 2]
VAL_HS = [1, 4, 16, 32]


def fmt(v):
    return v if isinstance(v, str) else f"{v:.6g}"


def render(fields, records):
    return "\n".join([",".join(fields)] + [",".join(fmt(r[k]) for k in fields) for r in records]) + "\n"


def val_table(d):
    fields = ["encoder", "seed", "best_step"] + [f"acc_h{h}" for h in VAL_HS] + [f"pred_h{h}" for h in VAL_HS]
    records = []
    for enc in ENCODERS:
        for seed in SEEDS:
            p = f"hist_{enc}_s{seed}_"
            r = dict(encoder=enc, seed=str(seed), best_step=str(int(d["hist_step"][np.argmin(d[p + "val_loss"])])))
            for h in VAL_HS:
                r[f"acc_h{h}"] = d[p + f"val_acc_h{h}"][-1]
                r[f"pred_h{h}"] = d[p + f"val_pred_h{h}"][-1]
            records.append(r)
    return render(fields, records)


def window_auc(dist, sets, config, rows=None, start=None):
    """AUROC of |c - c0| for one H: dist [E, W], windows of the stock walker vs of the config."""
    keep = np.ones(len(sets), bool) if rows is None else rows
    cols = slice(None) if start is None else [start]
    return auroc(dist[(sets == 0) & keep][:, cols].ravel(), dist[(sets == config) & keep][:, cols].ravel())


def auc_table(d):
    fields = ["encoder", "tasks", "config", "H"] + [f"auc_s{s}" for s in SEEDS] + ["mean", "min", "max"]
    records = []
    for enc in ENCODERS:
        for tasks, rows in [("all", None), ("walk", d["task"] == "walk")]:
            for c in range(1, len(NAMES)):
                for j, h in enumerate(d["hs"]):
                    v = [window_auc(d[f"dist_{enc}_s{s}"][:, :, j], d["set"], c, rows) for s in SEEDS]
                    r = dict(encoder=enc, tasks=tasks, config=NAMES[c], H=str(int(h)), mean=np.mean(v), min=min(v), max=max(v))
                    r.update({f"auc_s{s}": x for s, x in zip(SEEDS, v)})
                    records.append(r)
    return render(fields, records)


def start_table(d):
    fields = ["encoder", "config", "H", "start", "auc"]
    records = []
    for enc in ENCODERS:
        for c in range(1, len(NAMES)):
            for j, h in enumerate(d["hs"]):
                for w, start in enumerate(d["starts"]):
                    v = [window_auc(d[f"dist_{enc}_s{s}"][:, :, j], d["set"], c, start=w) for s in SEEDS]
                    records.append(dict(encoder=enc, config=NAMES[c], H=str(int(h)), start=str(int(start)), auc=np.mean(v)))
    return render(fields, records)


def err_table(d):
    fields = ["encoder", "config", "H"] + [f"{k}_s{s}" for s in SEEDS for k in ("ctx", "stock")] + ["gain"]
    records = []
    for enc in ENCODERS:
        for c, name in enumerate(NAMES):
            for j, h in enumerate(d["hs"]):
                r = dict(encoder=enc, config=name, H=str(int(h)))
                for s in SEEDS:
                    rows = d["set"] == c
                    r[f"ctx_s{s}"] = d[f"err_ctx_{enc}_s{s}"][rows, j].mean()
                    r[f"stock_s{s}"] = d[f"err_stock_{enc}_s{s}"][rows, j].mean()
                r["gain"] = np.mean([1 - r[f"ctx_s{s}"] / r[f"stock_s{s}"] for s in SEEDS])
                records.append(r)
    return render(fields, records)


def main():
    d = np.load(TASK / "data/inputs/context.npz")
    tables = {"context_val.csv": val_table(d), "context_auc.csv": auc_table(d),
              "context_start.csv": start_table(d), "context_err.csv": err_table(d)}
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
