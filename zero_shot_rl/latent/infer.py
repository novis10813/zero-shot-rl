"""Write the implicit-reward latents of a fixed standard FB checkpoint to an NPZ.

For every task: z_r, each family in zero_shot_rl.latent.implicit matched to ANGLES
degrees from z_r, the common axis AX, and the controls C1 and C2.

Usage (repo root):
    PYTHONPATH=. python -m zero_shot_rl.latent.infer \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> --out <zs.npz>
"""

import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.inference_buffer import FBInferenceReplayBuffer
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere
from zero_shot_rl.latent.implicit import DISCOUNT, FBImplicitZ, Simulator, tangent
from zero_shot_rl.latent.solver import angle, normalized

TASKS = ["stand", "walk", "run", "flip"]
ANGLES = [5, 10, 20, 35, 50, 70]  # target angles to z_r, degrees
AXIS_ANGLES = [-50, -35, -20, -10, 10, 20, 35, 50]
Z_FAMILIES = ("CQL", "ReBRAC", "ReBRAC+E2E", "E2E-bi")


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--transitions", type=int, default=100000)
    parser.add_argument("--n_inference", type=int, default=100000)
    parser.add_argument("--n_mu_states", type=int, default=2048)
    parser.add_argument("--n_mu_actions", type=int, default=32)
    parser.add_argument("--n_bi_states", type=int, default=1024)
    parser.add_argument("--cql_tau", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--bisect_iters", type=int, default=10)
    parser.add_argument("--n_random", type=int, default=3)
    parser.add_argument("--n_subsets", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sim = Simulator(args.workers)  # spawned before CUDA is initialised
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(args.seed)
    agent = load_agent(Path(args.checkpoint), device)
    reward_constructor = RewardFunctionConstructor(
        domain_name="walker", task_names=args.tasks, seed=args.seed, device=device
    )
    buffer = FBInferenceReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=Path(args.dataset_path),
        discount=DISCOUNT,
        transitions=args.transitions,
        device=device,
    )

    # inference set: transitions with a next dataset action; s' and r(s') from the
    # simulator as in the FB workspace
    rng = np.random.default_rng(args.seed)
    valid = buffer.valid_indices.cpu().numpy()
    idx = rng.choice(valid, min(args.n_inference, len(valid)), replace=False)
    storage = buffer.storage
    next_obs, rewards = reward_constructor.process_timesteps(
        batch_physics=storage["physics"][idx], batch_actions=storage["actions"][idx].cpu().numpy()
    )
    t_idx = torch.as_tensor(idx, device=device)
    data = {
        "obs": storage["observations"][t_idx],
        "actions": storage["actions"][t_idx],
        "physics": storage["physics"][idx],
        "next_obs": next_obs,
        "next_actions": storage["next_actions"][t_idx],
        "rewards": rewards,
    }
    iz = FBImplicitZ(agent, data, sim, args)
    d = iz.d

    rows = []

    def add(task, family, z_hat, z_hat_r, strength=0.0, target=np.nan, converged=True, iters=0,
            multi=False):
        z = normalized(z_hat, d)
        rows.append({
            "task": task, "family": family, "strength": float(strength),
            "target_angle": float(target), "angle": angle(z, normalized(z_hat_r, d)),
            "converged": converged, "iters": iters, "multi": multi, "z": z.cpu().numpy(),
        })
        print(f"{task:5s} {family:10s} target {target:5.1f} angle {rows[-1]['angle']:6.2f} "
              f"strength {strength:+.4g} iters {iters} converged {converged} multi {multi}",
              flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    for task in args.tasks:
        z_hat_r = iz.z_hat_r(task)
        iz.set_task(z_hat_r)
        add(task, "z_r", z_hat_r, z_hat_r)
        # zero strength reproduces z_r exactly
        for family in Z_FAMILIES + ("E2E",):
            z0, _, _ = iz.solve(family, 0.0, z_hat_r)
            assert torch.equal(normalized(z0, d), normalized(z_hat_r, d)), family

        tangents = {}
        for family in Z_FAMILIES + ("E2E",):
            grid = iz.angle_grid(family, 1, z_hat_r)
            for target in ANGLES:
                match = iz.match_angle(family, 1, target, z_hat_r, grid)
                if match is None or isinstance(match, str):
                    reason = match or f"unreachable (max angle {max(g[1] for g in grid):.1f})"
                    print(f"{task:5s} {family:10s} target {target:5.1f} {reason}", flush=True)
                    continue
                s, init = match
                z_hat, converged, iters = iz.solve(family, s, z_hat_r)
                multi = abs(angle(normalized(z_hat, d), normalized(z_hat_r, d)) - target) > 1.0
                if multi:
                    z_hat, converged, _ = iz.solve(family, s, z_hat_r, init=init)
                    if abs(angle(normalized(z_hat, d), normalized(z_hat_r, d)) - target) > 1.0:
                        print(f"{task:5s} {family:10s} target {target:5.1f} discontinuous",
                              flush=True)
                        continue
                add(task, family, z_hat, z_hat_r, s, target, converged, iters, multi)
                if target <= 10:  # 10 degrees, or 5 if 10 is not reached
                    tangents[family] = tangent(normalized(z_hat, d), normalized(z_hat_r, d))

        # AX: common pessimistic-to-optimistic axis
        unit_r = z_hat_r / z_hat_r.norm()
        pess = (tangents["CQL"] + tangents["ReBRAC"]) / 2
        axis = tangents["E2E-bi"] - pess
        axis = axis - torch.dot(axis, unit_r) * unit_r
        axis = axis / axis.norm()
        for target in AXIS_ANGLES:
            t = math.radians(target)
            add(task, "AX", math.cos(t) * unit_r + math.sin(t) * axis, z_hat_r, 0.0, target)

        # C1: random orthogonal rotations
        for _ in range(args.n_random):
            v = torch.randn(d, generator=gen).to(device)
            v = v - torch.dot(v, unit_r) * unit_r
            v = v / v.norm()
            for target in ANGLES:
                t = math.radians(target)
                add(task, "C1", math.cos(t) * unit_r + math.sin(t) * v, z_hat_r, 0.0, target)

        # C2: disjoint subsets of the inference set
        perm = torch.as_tensor(rng.permutation(len(idx)), device=device)
        for part in torch.chunk(perm, args.n_subsets):
            add(task, "C2", iz.z_hat_r(task, part), z_hat_r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        task=np.array([r["task"] for r in rows]),
        family=np.array([r["family"] for r in rows]),
        strength=np.array([r["strength"] for r in rows]),
        target_angle=np.array([r["target_angle"] for r in rows]),
        angle=np.array([r["angle"] for r in rows]),
        converged=np.array([r["converged"] for r in rows]),
        iters=np.array([r["iters"] for r in rows]),
        multi=np.array([r["multi"] for r in rows]),
        z=np.stack([r["z"] for r in rows]),
        u_1=iz.u_1.cpu().numpy(),
        checkpoint=args.checkpoint,
    )
    print(f"wrote {len(rows)} latents to {out}")
    sim.pool.close()


if __name__ == "__main__":
    main()
