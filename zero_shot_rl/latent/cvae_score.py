"""Score the paired one-step transitions of zero_shot_rl.eval.paired_step with a CVAE ensemble.

For every `next_<name>` array of each --paired NPZ (the replayed dynamics and the stock
rollout) it computes the member errors e_i = s' - s'_i = (s' - s) - f_i(s, a) of
zero_shot_rl.latent.cvae_model. Files are stacked along the episode axis (`seed` gives the file
of each episode). For each `next_<name>` the output has

    member_<name>  [episodes, steps, N]  r_i = mean(e_i^2) over the observation dimensions, per member
    mean_<name>    [episodes, steps]     mean over members of r_i
    min_<name>     [episodes, steps]     min over members of r_i (the ICP statistic: OOD iff min >= threshold)
    gap_<name>     [episodes, steps]     |s'_name - s'_rollout|_2
    mse_<name>     [obs_dim]             mean of e^2 over members and samples

and, for the stock rollout, `var_delta` [obs_dim], the variance of Delta s, the baseline
validation errors of the members, `val_member` [N, samples] (the calibration set), and the mean
and std of [s, a] and of the target that standardise the model, `x_mean`, `x_std`, `y_mean`,
`y_std`. Thresholds are taken from `val_member` with zero_shot_rl.latent.cvae_model.icp_threshold.

Usage (repo root):
    PYTHONPATH=. uv run python -m zero_shot_rl.latent.cvae_score \
        --model <model.pt> --paired <paired_s0.npz> <paired_s1.npz> <paired_s2.npz> --out <scored.npz>
"""

from argparse import ArgumentParser

import numpy as np
import torch

from zero_shot_rl.latent.cvae_model import icp_threshold, is_ood, load_cvae_ensemble

CHUNK = 20000


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
        e = torch.cat([model.error(s64[i:i + CHUNK], a[i:i + CHUNK], s2[i:i + CHUNK])
                       for i in range(0, len(s64), CHUNK)], dim=1)
        member = (e ** 2).mean(-1).numpy()
        out["member_" + name] = member.T.reshape(*shape, -1)
        out["mean_" + name] = member.mean(0).reshape(shape)
        out["min_" + name] = member.min(0).reshape(shape)
        out["gap_" + name] = np.linalg.norm(paired[key] - paired["next_rollout"], axis=-1)
        out["mse_" + name] = (e ** 2).mean((0, 1)).numpy()
    delta = paired["next_rollout"] - paired["observations"]
    out["var_delta"] = delta.reshape(-1, delta.shape[-1]).var(0)
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--paired", nargs="+", required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    model, meta = load_cvae_ensemble(args.model)
    per_file = [score_file(model, np.load(p)) for p in args.paired]
    out = {"val_member": meta["val_member"],
           "x_mean": meta["x_mean"], "x_std": meta["x_std"],
           "y_mean": meta["y_mean"], "y_std": meta["y_std"],
           "seed": np.concatenate([np.full(len(np.load(p)["episode"]), i) for i, p in enumerate(args.paired)]),
           "episode": np.concatenate([np.load(p)["episode"] for p in args.paired])}
    for key in per_file[0]:
        parts = [f[key] for f in per_file]
        out[key] = np.concatenate(parts) if parts[0].ndim >= 2 else np.mean(parts, axis=0)
    thr = icp_threshold(meta["val_member"], 0.95)
    for name in sorted(k[len("mean_"):] for k in out if k.startswith("mean_")):
        rate = is_ood(out["member_" + name].transpose(2, 0, 1), thr).mean()
        print(f"{name:24s} median r (member mean) {np.median(out['mean_' + name]):.3g}  "
              f"OOD rate at delta 0.95 {rate:.3f}")
    np.savez(args.out, **out)


if __name__ == "__main__":
    main()
