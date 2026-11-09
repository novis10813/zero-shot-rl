"""Score the paired one-step transitions of zero_shot_rl.eval.paired_step with a dynamics model.

For every `next_<name>` array of each --paired NPZ (the replayed dynamics and the stock
rollout) it computes the error e = (s' - s) - f(s, a) of zero_shot_rl.latent.dynamics_model.
Files are stacked along the episode axis (`seed` gives the file of each episode). For each
`next_<name>` the output has

    norm_<name>   [episodes, steps]  |e|_2
    maha_<name>   [episodes, steps]  D_dyn
    gap_<name>    [episodes, steps]  |s'_name - s'_rollout|_2
    mse_<name>    [obs_dim]          mean of e^2 over all samples

and, for the stock rollout, `var_delta` [obs_dim], the variance of Delta s (the MSE of
predicting its mean), the baseline validation scores of the model, `val_norm`, `val_maha`, and
the mean and std of [s, a] and of Delta s that standardise its inputs and target, `x_mean`,
`x_std`, `y_mean`, `y_std`.

Usage (repo root):
    PYTHONPATH=. uv run python -m zero_shot_rl.latent.dynamics_score \
        --model <model.pt> --paired <paired_s0.npz> <paired_s1.npz> <paired_s2.npz> --out <scored.npz>
"""

from argparse import ArgumentParser

import numpy as np
import torch

from zero_shot_rl.latent.dynamics_model import load_dynamics_model


def score_file(model, paired):
    """Scores of every next_<name> array of one paired NPZ."""
    shape = paired["observations"].shape[:2]
    s64 = paired["observations"].reshape(-1, paired["observations"].shape[-1])
    a = torch.as_tensor(paired["actions"], dtype=torch.float32).reshape(-1, paired["actions"].shape[-1])
    out = {}
    for key in paired:
        if not key.startswith("next_"):
            continue
        name = key[len("next_"):]
        s2 = paired[key].reshape(-1, s64.shape[1])
        e = model.error(s64, a, s2)
        out["norm_" + name] = e.norm(dim=1).numpy().reshape(shape)
        out["maha_" + name] = model.distance(e).numpy().reshape(shape)
        out["gap_" + name] = np.linalg.norm(paired[key] - paired["next_rollout"], axis=-1)
        out["mse_" + name] = (e ** 2).mean(0).numpy()
    delta = paired["next_rollout"] - paired["observations"]
    out["var_delta"] = delta.reshape(-1, delta.shape[-1]).var(0)
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--paired", nargs="+", required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    model, meta = load_dynamics_model(args.model)
    per_file = [score_file(model, np.load(p)) for p in args.paired]
    out = {"val_norm": meta["val_norm"], "val_maha": meta["val_maha"],
           "x_mean": meta["x_mean"], "x_std": meta["x_std"],
           "y_mean": meta["y_mean"], "y_std": meta["y_std"],
           "seed": np.concatenate([np.full(len(np.load(p)["episode"]), i) for i, p in enumerate(args.paired)]),
           "episode": np.concatenate([np.load(p)["episode"] for p in args.paired])}
    for key in per_file[0]:
        parts = [f[key] for f in per_file]
        out[key] = np.concatenate(parts) if parts[0].ndim == 2 else np.mean(parts, axis=0)
    for name in sorted(k[len("norm_"):] for k in out if k.startswith("norm_")):
        print(f"{name:24s} |e| median {np.median(out['norm_' + name]):.4f}  D_dyn mean {out['maha_' + name].mean():.2f}")
    np.savez(args.out, **out)


if __name__ == "__main__":
    main()
