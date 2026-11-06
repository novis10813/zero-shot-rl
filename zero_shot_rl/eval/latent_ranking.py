"""Predicted against actual return of a fixed set of latents, and how they rank.

For OpTI-BFM, the return of a latent w at state s is predicted by psi(s, w)^T c, with c a
reward coefficient (Rupf et al., arXiv 2510.20264, section 3.2). Per task and run i, this
entry takes K candidate latents, the initial state s_0 of the environment seeded with i, and
compares three things for every candidate:

    actual         discounted and undiscounted return of a deterministic 1000-step
                   episode that holds the latent from s_0, in the environment given by
                   --dynamics
    offline        psi(s_0, w)^T c with c the least-squares coefficient of the task reward
                   on phi over the --z_inference_steps inference transitions of the stock
                   dataset (the reward is the same under every dynamics). Its error is the
                   error of psi (and of the dynamics it was trained on), not of task
                   inference
    LABEL@epE      the same with c the belief mean of run i at the end of episode E of a
                   zero_shot_rl.eval.online_inference --predictions_dir run, so its
                   error adds the error of the online coefficient

Candidates of a task (the same for every run): the latent of offline task inference
(zero_shot_rl.latent.task_z), the direction of c, 24 latents at angles 10, 20, 30, 45, 60 and
90 degrees from the first (4 random directions each), and 6 uniform on the sphere.

--out is long format, one row per task, run, candidate and source: task, trial, candidate,
angle (degrees from the task latent), source (actual_discounted, actual_total, offline or
LABEL@epE) and value. Predictions are in discounted-return units, so they compare with
actual_discounted.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.latent_ranking \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> --dynamics <yaml> \
        --beliefs UCB pred/ucb:0.001.npz --beliefs TS pred/ts:0.1.npz --out ranking.csv
"""

import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from zero_shot_rl.env.dynamics import load_dynamics
from zero_shot_rl.env.walker import STEPS, make_env, run_episodes
from zero_shot_rl.latent import task_z
from zero_shot_rl.online.fb import FBModel
from zero_shot_rl.provenance import no_clobber

FIELDS = ["task", "trial", "candidate", "angle", "source", "value"]
ANGLES = [10, 20, 30, 45, 60, 90]
PER_ANGLE = 4
UNIFORM = 6


def candidates(z_r, w_off, rng):
    """[K, d] unit latent directions and their angles to z_r in degrees."""
    u = z_r / np.linalg.norm(z_r)
    vectors, angles = [u, w_off / np.linalg.norm(w_off)], [0.0, None]
    for angle in ANGLES:
        for _ in range(PER_ANGLE):
            v = rng.standard_normal(len(u))
            v -= (v @ u) * u
            v /= np.linalg.norm(v)
            theta = np.radians(angle)
            vectors.append(np.cos(theta) * u + np.sin(theta) * v)
            angles.append(float(angle))
    for _ in range(UNIFORM):
        v = rng.standard_normal(len(u))
        vectors.append(v / np.linalg.norm(v))
        angles.append(None)
    vectors = np.stack(vectors)
    angles = [np.degrees(np.arccos(np.clip(v @ u, -1, 1))) if a is None else a
              for v, a in zip(vectors, angles)]
    return vectors, angles


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--beliefs", nargs=2, action="append", default=[],
                        metavar=("LABEL", "NPZ"))
    parser.add_argument("--episodes", nargs="+", type=int, default=[1, 3, 10],
                        help="1-based episodes whose end-of-episode belief predicts")
    parser.add_argument("--tasks", nargs="+", default=task_z.TASKS)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--dynamics", type=str, default=None,
                        help="yaml of zero_shot_rl.env.dynamics; default is the stock environment")
    parser.add_argument("--batch", type=int, default=160, help="environments stepped at once")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    dynamics = load_dynamics(args.dynamics) if args.dynamics else None

    device = task_z.device()
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    buffer, observations_z, rewards_z = task_z.inference_set(
        Namespace(seed=args.seed, discount=args.discount, dataset_path=args.dataset_path,
                  dataset_transitions=args.dataset_transitions,
                  z_inference_steps=args.z_inference_steps),
        torch.device("cpu"),
    )
    model = FBModel(agent, buffer.storage["next_observations"].to(device))
    reward_fns = task_z.reward_functions(args, device)
    phi_z = model.phi(observations_z.to(device)).cpu()
    del buffer
    beliefs = {label: np.load(path) for label, path in args.beliefs}
    discounts = args.discount ** np.arange(STEPS)

    rows = []
    for task_index, task in enumerate(args.tasks):
        z_r = agent.infer_z(observations_z.to(device), rewards_z[task].to(device))
        w_off = model.lstsq(phi_z, rewards_z[task].squeeze(1)).numpy()
        vectors, angles = candidates(z_r, w_off, np.random.default_rng(task_index))
        k = len(vectors)
        n = args.trials

        starts = np.stack([make_env(i, dynamics).reset().observation["observations"]
                           for i in range(n)])
        w = torch.as_tensor(vectors, dtype=torch.float64, device=device)
        sf = model.psi(torch.as_tensor(starts, dtype=torch.float32, device=device),
                       w.expand(n, -1, -1)).double().mean(2)  # [n, K, d]
        coefficients = {"offline": np.broadcast_to(w_off, (n, len(w_off)))}
        for label, data in beliefs.items():
            for episode in args.episodes:
                coefficients[f"{label}@ep{episode}"] = data[f"{task}_belief"][:n, episode - 1]
        predicted = {source: torch.einsum("nkd,nd->nk", sf.cpu(), torch.as_tensor(c)).numpy()
                     for source, c in coefficients.items()}

        env_seeds = [i for i in range(n) for _ in range(k)]
        latents = model.unit(w.expand(n, -1, -1).reshape(n * k, -1))
        discounted, total = [], []
        for c in range(0, len(env_seeds), args.batch):
            envs = [make_env(s, dynamics) for s in env_seeds[c : c + args.batch]]
            rewards = run_episodes(agent, latents[c : c + args.batch], envs,
                                   reward_fns[task])["rewards"]
            discounted.extend((rewards * discounts).sum(1))
            total.extend(rewards.sum(1))
            for env in envs:
                env.close()
            print(f"{task}: {c + len(envs)} / {len(env_seeds)} episodes", flush=True)
        values = {"actual_discounted": np.array(discounted).reshape(n, k),
                  "actual_total": np.array(total).reshape(n, k), **predicted}
        for source, table in values.items():
            for i in range(n):
                for j in range(k):
                    rows.append({"task": task, "trial": i, "candidate": j,
                                 "angle": angles[j], "source": source, "value": table[i, j]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(out) as path, open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                             for key, v in row.items()})


if __name__ == "__main__":
    main()
