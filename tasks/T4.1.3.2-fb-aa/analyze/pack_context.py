"""Pack the context encoder's validation history and test scores into data/inputs/context.npz.

From the context_s{0,1,2}_{transformer,mean}_history.json of run_context.sh and the
score_s{0,1,2}_{transformer,mean}.npz of run_context_eval.sh (zero_shot_rl.latent.context_score),
writes, for every model <enc>_s<seed>:

    hist_step, hist_<enc>_s<seed>_<key>   [V]  validation every 5,000 steps; key val_acc_h<H>,
                                               val_pred_h<H> (H 1, 4, 16, 32) and val_loss
    dist_<enc>_s<seed>       [E, W, nH]  |c - c0_H|, the detection score of every test window
    err_ctx_<enc>_s<seed>    [E, nH]     mean over the windows of the episode of the error of g
    err_stock_<enc>_s<seed>  [E, nH]     with the window's context and with the stock context

and once hs, starts, set, task, policy_seed (the same in every score file; set 0 the stock
walker, 1 to 8 the configs of configs/test_dynamics.yaml in order).

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.pack_context \
        --history <dir of run_context.sh's out> --scores <dir of run_context_eval.sh's out>
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
ENCODERS = ["transformer", "mean"]
SEEDS = [0, 1, 2]
SHARED = ["hs", "starts", "set", "task", "policy_seed"]


def pack(history_dir, score_dir):
    out = {}
    for enc in ENCODERS:
        for seed in SEEDS:
            name = f"{enc}_s{seed}"
            history = json.loads((Path(history_dir) / f"context_s{seed}_{enc}_history.json").read_text())
            steps = np.array([r["step"] for r in history])
            if "hist_step" in out and not np.array_equal(out["hist_step"], steps):
                raise ValueError(f"validation steps of {name} differ")
            out["hist_step"] = steps
            for key in history[0]:
                if key.startswith("val_"):
                    out[f"hist_{name}_{key}"] = np.array([r[key] for r in history], np.float32)
            with np.load(Path(score_dir) / f"score_s{seed}_{enc}.npz") as f:
                for key in SHARED:
                    if key in out and not np.array_equal(out[key], f[key]):
                        raise ValueError(f"{key} of {name} differs from the other score files")
                    out[key] = f[key]
                out[f"dist_{name}"] = f["dist"].astype(np.float32)
                out[f"err_ctx_{name}"] = f["err_ctx"].mean(1).astype(np.float32)
                out[f"err_stock_{name}"] = f["err_stock"].mean(1).astype(np.float32)
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--history", type=str, required=True)
    parser.add_argument("--scores", type=str, required=True)
    args = parser.parse_args()
    out = TASK / "data/inputs"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "context.npz", **pack(args.history, args.scores))
    print("written")


if __name__ == "__main__":
    main()
