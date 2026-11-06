"""Score the paired one-step transitions of zero_shot_rl.eval.paired_step with a detector.

For every `next_<name>` array of the --paired NPZ (the replayed dynamics and the stock
rollout) it computes D_t of (s_t, a_t, s') under the stored z_r of --task, with the detector
of zero_shot_rl.latent.shift_detector. All arrays of the input are copied, and for each
`next_<name>` the output adds

    D_<name>     [episodes, steps]
    mean_<name>  [z_dim]          mean of r = e - mu_0 over all samples
    cov_<name>   [z_dim, z_dim]   covariance of r over all samples

Usage (repo root):
    PYTHONPATH=. uv run python -m zero_shot_rl.eval.paired_score \
        --checkpoint <fb.pt> --detector <detector.pt> --zs <zs.npz> \
        --paired <paired.npz> --out <scored.npz>
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from zero_shot_rl.latent.shift_detector import load_detector

BATCH = 20000


def score_all(agent, detector, paired, z, device):
    out = {}
    s, a = (torch.as_tensor(paired[k], dtype=torch.float32).reshape(-1, paired[k].shape[-1]) for k in ("observations", "actions"))
    shape = paired["observations"].shape[:2]
    for key in paired:
        if not key.startswith("next_"):
            continue
        s2 = torch.as_tensor(paired[key], dtype=torch.float32).reshape(-1, s.shape[1])
        d = [detector.score(agent, s[i:i + BATCH].to(device), a[i:i + BATCH].to(device),
                            s2[i:i + BATCH].to(device), z.expand(len(s[i:i + BATCH]), -1)).cpu()
             for i in range(0, len(s), BATCH)]
        name = key[len("next_"):]
        out["D_" + name] = torch.cat(d).numpy().reshape(shape)
        r = torch.cat([detector.centered(agent, s[i:i + BATCH].to(device), a[i:i + BATCH].to(device),
                                         s2[i:i + BATCH].to(device), z.expand(len(s[i:i + BATCH]), -1)).cpu()
                       for i in range(0, len(s), BATCH)]).double().numpy()
        out["mean_" + name], out["cov_" + name] = r.mean(0), np.cov(r.T)
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--detector", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--paired", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(Path(args.checkpoint), device).eval()
    detector = load_detector(Path(args.detector), device)
    zs = np.load(args.zs)
    rows = np.flatnonzero((zs["task"] == args.task) & (zs["family"] == "z_r"))
    if len(rows) != 1:
        raise ValueError(f"expected one z_r row for {args.task}, found {len(rows)}")
    z = torch.as_tensor(zs["z"][rows[0]], dtype=torch.float32, device=device)

    paired = dict(np.load(args.paired))
    scores = score_all(agent, detector, paired, z, device)
    for name, d in scores.items():
        print(f"{name:24s} D mean {d.mean():8.2f}  median {np.median(d):8.2f}")
    np.savez(args.out, **paired, **scores)


if __name__ == "__main__":
    main()
