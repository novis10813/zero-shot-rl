"""Pack the K-step replay and the backward features of the paired data into data/inputs.

horizon: from the horizon_s{0,1,2}.npz of run_horizon.sh (zero_shot_rl.eval.paired_horizon) and
the dynamics model of T4.1.2.1 (zero_shot_rl.latent.dynamics_model, target Delta s), writes
data/inputs/horizon.npz with, for the trajectories of the three files stacked on the first axis
(episodes x starts) and the steps k in `ks` on the second:

    open_gap_<config>    |s_k^config - s_k^stock|_2, open loop (the stock rollout's actions)
    closed_gap_<config>  the same, closed loop (each simulator's own actor actions)
    closed_floor         |s_k^stock, closed loop - s_k of the stock rollout|_2: the actor of the
                         replay runs on one observation, the rollout's on a batch, so the two
                         differ only by floating point
    multi_<name>         |s_k - s_hat_k|_2 of the closed-loop trajectory of <name> (stock or a
                         config), s_hat_k the k-step prediction of the model from s_0 with the
                         trajectory's actions
    onestep_<name>       mean over j = 1..k of |e_j|_2, e_j the one-step error of the model on the
                         same trajectory

b0: from the paired_raw_s{0,1,2}.npz of run_classifier.sh and the three FB checkpoints, each file
scored with the B of its own checkpoint, writes data/inputs/b0.npz with one value per transition
of the three files stacked (seed in `seed`):

    s_gap_<config>   |s'_config - s'_stock|_2
    b_gap_<config>   |B(s'_config) - B(s'_stock)|_2
    b_step_<name>    |B(s'_name) - B(s)|_2, name stock or a config
    s_step_stock     |s'_stock - s|_2
    s_scale, b_scale [3]  per file, median distance between 20,000 random pairs of stock s'
                          (seed 0), in the observation and in B

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.pack \
        --horizon <dir with horizon_s*.npz> --paired <dir with paired_raw_s*.npz> \
        --model <T4.1.2.1 dynamics.pt> --checkpoints <seed 0 .pt> <seed 1 .pt> <seed 2 .pt>
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from zero_shot_rl.latent.dynamics_model import load_dynamics_model

TASK = Path(__file__).resolve().parents[1]
CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]
KS = [1, 2, 3, 5, 10, 20, 30, 50, 100]
PAIRS = 20000


def norm(x):
    return np.linalg.norm(x, axis=-1)


@torch.no_grad()
def model_errors(model, s0, traj, actions):
    """k-step prediction error and running mean of the one-step error, [N, K] each."""
    n, horizon, d = traj.shape
    a = torch.as_tensor(actions, dtype=torch.float32)
    s, pred = torch.as_tensor(s0, dtype=torch.float32), []
    for j in range(horizon):
        s = s + model(s, a[:, j])
        pred.append(s.numpy())
    multi = norm(np.stack(pred, 1) - traj)
    prev = np.concatenate([s0[:, None], traj[:, :-1]], 1)
    one = model.error(prev.reshape(-1, d), a.reshape(n * horizon, -1), traj.reshape(-1, d)).numpy()
    one = norm(one.reshape(n, horizon, d))
    return multi, np.cumsum(one, 1) / np.arange(1, horizon + 1)


def pack_horizon(directory, model_path):
    files = [np.load(Path(directory) / f"horizon_s{i}.npz") for i in range(3)]
    d = files[0]["observations"].shape[-1]
    cat = lambda k: np.concatenate([f[k] for f in files]).reshape(-1, files[0][k].shape[-2], files[0][k].shape[-1])  # noqa: E731
    cols = np.array(KS) - 1
    model, _ = load_dynamics_model(model_path)
    s0 = np.concatenate([f["observations"] for f in files]).reshape(-1, d)
    out = {"ks": np.array(KS)}
    open_base, closed_base = cat("open_baseline"), cat("closed_baseline")
    out["closed_floor"] = norm(closed_base - cat("rollout"))[:, cols]
    for name in ["baseline"] + CONFIGS:
        closed = cat(f"closed_{name}")
        if name != "baseline":
            out[f"open_gap_{name}"] = norm(cat(f"open_{name}") - open_base)[:, cols]
            out[f"closed_gap_{name}"] = norm(closed - closed_base)[:, cols]
        multi, one = model_errors(model, s0, closed, cat(f"closed_actions_{name}"))
        key = "stock" if name == "baseline" else name
        out[f"multi_{key}"], out[f"onestep_{key}"] = multi[:, cols], one[:, cols]
    return {k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in out.items()}


def pack_b0(directory, checkpoints):
    rng = np.random.default_rng(0)
    parts, seeds, s_scale, b_scale = [], [], [], []
    for i, path in enumerate(checkpoints):
        agent = load_agent(Path(path), torch.device("cpu"))
        agent.eval()
        paired = np.load(Path(directory) / f"paired_raw_s{i}.npz")
        d = paired["observations"].shape[-1]
        flat = lambda k: paired[k].reshape(-1, d)  # noqa: E731

        def backward(x):
            with torch.no_grad():
                return agent.FB.backward_representation(torch.as_tensor(x, dtype=torch.float32)).numpy()
        s, base = flat("observations"), flat("next_baseline")
        b_s, b_base = backward(s), backward(base)
        part = {"b_step_stock": norm(b_base - b_s), "s_step_stock": norm(base - s)}
        for name in CONFIGS:
            shifted = flat(f"next_{name}")
            b_shift = backward(shifted)
            part[f"s_gap_{name}"] = norm(shifted - base)
            part[f"b_gap_{name}"] = norm(b_shift - b_base)
            part[f"b_step_{name}"] = norm(b_shift - b_s)
        i1, i2 = rng.integers(0, len(base), (2, PAIRS))
        s_scale.append(np.median(norm(base[i1] - base[i2])))
        b_scale.append(np.median(norm(b_base[i1] - b_base[i2])))
        parts.append(part)
        seeds.append(np.full(len(s), i))
    out = {k: np.concatenate([p[k] for p in parts]).astype(np.float32) for k in parts[0]}
    out.update(seed=np.concatenate(seeds), s_scale=np.array(s_scale), b_scale=np.array(b_scale))
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--horizon", type=str, required=True)
    parser.add_argument("--paired", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoints", nargs=3, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = TASK / "data/inputs"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "horizon.npz", **pack_horizon(args.horizon, args.model))
    np.savez_compressed(out / "b0.npz", **pack_b0(args.paired, args.checkpoints))
    print("written")


if __name__ == "__main__":
    main()
