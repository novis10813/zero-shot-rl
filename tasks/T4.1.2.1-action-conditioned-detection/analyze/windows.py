"""Detection with the mean |e|_2 over windows of K consecutive steps of an episode.

Reads data/inputs/dynamics_scored_delta.npz (the model regressing Delta s). A window is K
consecutive steps of one episode (non-overlapping, the remainder of the 999 steps dropped), its
score the mean of |e|_2. Writes

    data/windows.csv        one row per (K, config): AUROC of the windows of the shifted episodes
                            against the stock windows of the test episodes; threshold = 99th
                            percentile of the stock windows of the 12 validation episodes, with the
                            fractions of shifted (tpr) and of stock (fpr) test windows above it
    data/episode_mean.csv   one row per config: min, median and max over the 60 test episodes of the
                            mean |e|_2 of the whole episode (the stock row is `stock`)

Windows of one episode share the shift and the policy, so they are not independent samples.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.2.1-action-conditioned-detection uv run python -m analyze.windows [--check]
"""

import sys

import numpy as np

from analyze.detect import CONFIGS, TASK, auroc

STEPS = 999
VAL_EPISODES = 12
WINDOWS = [1, 10, 50, 100, 333, 999]
FIELDS = ["K", "config", "auroc", "thr", "tpr", "fpr"]


def window_means(x, k):
    """Means of non-overlapping windows of k steps, [episodes, steps] -> flat array."""
    n = x.shape[1] // k
    return x[:, : n * k].reshape(x.shape[0], n, k).mean(-1).ravel()


def window_rows(data):
    val = data["val_norm"].reshape(VAL_EPISODES, STEPS)
    out = []
    for k in WINDOWS:
        thr = np.percentile(window_means(val, k), 99)
        base = window_means(data["norm_rollout"], k)
        for c in CONFIGS:
            w = window_means(data[f"norm_{c}"], k)
            out.append({"K": k, "config": c, "auroc": auroc(base, w), "thr": thr,
                        "tpr": (w > thr).mean(), "fpr": (base > thr).mean()})
    return out


def episode_rows(data):
    out = []
    for name in ["rollout"] + CONFIGS:
        m = data[f"norm_{name}"].mean(1)
        out.append({"config": "stock" if name == "rollout" else name, "min": m.min(), "median": np.median(m),
                    "max": m.max()})
    return out


def render(rows, fields):
    lines = [",".join(fields)]
    for r in rows:
        lines.append(",".join(str(r[k]) if k == "config" else f"{r[k]:.6g}" for k in fields))
    return "\n".join(lines) + "\n"


def tables():
    data = np.load(TASK / "data/inputs/dynamics_scored_delta.npz")
    return {"windows.csv": render(window_rows(data), FIELDS),
            "episode_mean.csv": render(episode_rows(data), ["config", "min", "median", "max"])}


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
