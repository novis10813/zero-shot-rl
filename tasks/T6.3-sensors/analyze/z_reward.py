"""Reward functions implied by task latents z: r_z(s) = B(s)^T Sigma_B^{-1} z.

FB sees a reward only through z = E_rho[B(s) r(s)], so rewards that differ by a part orthogonal to
B give the same z and the same policy. The unique reward in the span of B that produces z is

    r_z(s) = B(s)^T Sigma_B^{-1} z,    Sigma_B = E_rho[B(s) B(s)^T],

the reward whose value F(s, a, z)^T z estimates. This script evaluates r_z on the reference states
for the zero-shot latent z_r of each task and for the latents found by T6.2 runs
(`data/inputs/found_z.npz`, from `analyze.extract_found_z`).

Since ||z|| = sqrt(d_z) for every latent, all r_z of one task share one affine scale: the a, b
that best fit a * r_{z_r} + b to the true task reward on the reference states.

Outputs in `--out_dir`:
    z_reward_corr.csv  Pearson correlation of each found r_z with the true reward and with r_{z_r}
    z_reward_bins.npz  r_z, r_{z_r} and true reward averaged in 25 quantile bins of each sensor

Usage (repo root):
    PYTHONPATH=.:tasks/T6.3-sensors MUJOCO_GL=egl uv run python -m analyze.z_reward \\
        --checkpoint <fb.pt> --dataset_path datasets/walker/rnd/dataset.npz --out_dir tasks/T6.3-sensors/data
"""

import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from agents.base import load_agent
from zero_shot_rl.finetune.anfis_z import collect_reference_set
from zero_shot_rl.latent.anfis import weighted_z
from zero_shot_rl.provenance import no_clobber

SENSORS = ("torso_height", "torso_upright", "horizontal_velocity", "angmomentum")
NUM_BINS = 25


def implied_rewards(backward_repr: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """r_z(s) = B(s)^T Sigma_B^{-1} z for every row of `zs` [M, d]; returns [N, M]."""
    sigma = backward_repr.T @ backward_repr / len(backward_repr)
    return backward_repr @ np.linalg.solve(sigma, zs.T)


def affine_scale(source: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Least-squares a, b with a * source + b ~ target."""
    design = np.stack([source, np.ones_like(source)], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(a), float(b)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def bin_means(x: np.ndarray, values: np.ndarray, num_bins: int = NUM_BINS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of each column of `values` [N, M] in quantile-spaced bins of x over [p01, p99].

    Returns (bin centers [B], counts [B], means [B, M]); empty bins are NaN."""
    lo, hi = np.quantile(x, [0.01, 0.99])
    edges = np.linspace(lo, hi, num_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, num_bins - 1)
    inside = (x >= lo) & (x <= hi)
    counts = np.bincount(idx[inside], minlength=num_bins)
    sums = np.stack([np.bincount(idx[inside], weights=v[inside], minlength=num_bins) for v in values.T], axis=1)
    with np.errstate(invalid="ignore"):
        means = sums / counts[:, None]
    return 0.5 * (edges[:-1] + edges[1:]), counts, means


def main(argv: Sequence[str] = None) -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--found_z", default="tasks/T6.3-sensors/data/inputs/found_z.npz")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--reference_seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    found = np.load(args.found_z)
    z_found, conditions, envs, tasks, seeds = (found[k] for k in ("z", "conditions", "envs", "tasks", "seeds"))
    tasks = [str(t) for t in tasks]

    observations, features, rewards, _ = collect_reference_set(
        dataset_path=Path(args.dataset_path),
        domain="walker",
        sensor_names=list(SENSORS),
        tasks=tasks,
        dataset_transitions=args.dataset_transitions,
        z_inference_steps=args.z_inference_steps,
        seed=args.reference_seed,
        discount=args.discount,
        device=device,
    )
    agent = load_agent(Path(args.checkpoint), device)
    with torch.no_grad():
        b_t = agent.FB.backward_representation(observations).detach()
        z_r = torch.stack([weighted_z(rewards[t], b_t) for t in tasks]).double().cpu().numpy()  # [T, d]
    b = b_t.double().cpu().numpy()
    feats = features.double().cpu().numpy()
    true_r = np.stack([rewards[t].double().cpu().numpy()[:, 0] for t in tasks], axis=1)  # [N, T]

    nc, ne, nt, ns, d = z_found.shape
    r_zr = implied_rewards(b, z_r)  # [N, T]
    r_found = implied_rewards(b, z_found.reshape(-1, d)).reshape(len(b), nc, ne, nt, ns)

    scale = np.array([affine_scale(r_zr[:, ti], true_r[:, ti]) for ti in range(nt)])  # [T, 2]
    r_zr = r_zr * scale[:, 0] + scale[:, 1]
    r_found = r_found * scale[None, None, None, :, 0, None] + scale[None, None, None, :, 1, None]

    rows = []
    for ti, t in enumerate(tasks):
        rows.append({"condition": "z_r", "env": "-", "task": t, "seed": "-",
                     "corr_true": corr(r_zr[:, ti], true_r[:, ti]), "corr_zr": 1.0})
        for ci, c in enumerate(conditions):
            for ei, e in enumerate(envs):
                for si, s in enumerate(seeds):
                    r = r_found[:, ci, ei, ti, si]
                    rows.append({"condition": str(c), "env": str(e), "task": t, "seed": int(s),
                                 "corr_true": corr(r, true_r[:, ti]), "corr_zr": corr(r, r_zr[:, ti])})

    centers, counts, true_bins, zr_bins, found_bins = [], [], [], [], []
    for xi in range(len(SENSORS)):
        x = feats[:, xi]
        cen, cnt, m_true = bin_means(x, true_r)
        _, _, m_zr = bin_means(x, r_zr)
        _, _, m_found = bin_means(x, r_found.reshape(len(b), -1))
        centers.append(cen)
        counts.append(cnt)
        true_bins.append(m_true)
        zr_bins.append(m_zr)
        found_bins.append(m_found.reshape(NUM_BINS, nc, ne, nt, ns))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with no_clobber(out / "z_reward_corr.csv") as tmp:
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with no_clobber(out / "z_reward_bins.npz") as tmp:
        with open(tmp, "wb") as f:
            np.savez(
                f,
                sensors=np.array(SENSORS), conditions=conditions, envs=envs, tasks=np.array(tasks), seeds=seeds,
                centers=np.stack(centers),            # [sensors, bins]
                counts=np.stack(counts),              # [sensors, bins]
                true=np.stack(true_bins),             # [sensors, bins, tasks]
                z_r=np.stack(zr_bins),                # [sensors, bins, tasks]
                found=np.stack(found_bins),           # [sensors, bins, conditions, envs, tasks, seeds]
                scale=scale,                          # [tasks, 2] (a, b)
            )
    print(f"wrote {out / 'z_reward_corr.csv'} and {out / 'z_reward_bins.npz'}")


if __name__ == "__main__":
    main()
