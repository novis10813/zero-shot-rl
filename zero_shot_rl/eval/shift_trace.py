"""Score every step of FB rollouts in (shifted) simulators with a dynamic-shift detector.

For each --dynamics yaml (zero_shot_rl.env.dynamics) it runs --episodes deterministic
episodes of 1000 steps with the clipped actor mean at the stored z_r of --task, as in
zero_shot_rl.eval.dynamics_sweep, and scores each transition (s_t, a_t, s_{t+1}) with the
detector of zero_shot_rl.latent.shift_detector:

    D_t = (e_t - mu_0)^T (Sigma_0 + eps I)^{-1} (e_t - mu_0)

Writes one NPZ per yaml, <out_dir>/<yaml stem>.npz, with D, reward and height of shape
[episodes, 1000], episode (the environment seeds) and the return per episode.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.shift_trace \
        --checkpoint <fb.pt> --detector <detector.pt> --zs <zs.npz> \
        --dynamics <a.yaml> <b.yaml> --out_dir <dir>
"""

import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from rewards import RewardFunctionConstructor
from zero_shot_rl.env.dynamics import load_dynamics
from zero_shot_rl.env.walker import STEPS, make_env, policy
from zero_shot_rl.latent.shift_detector import load_detector

G = {}  # per-process state, filled before the pool forks


def rollout(path):
    torch.set_num_threads(1)
    args, agent, detector, reward_fn = G["args"], G["agent"], G["detector"], G["reward_fn"]
    dynamics = load_dynamics(path)
    n = args.episodes
    envs = [make_env(args.first_episode + e, dynamics) for e in range(n)]
    obs = np.stack([env.reset().observation["observations"] for env in envs])
    z = torch.as_tensor(G["z"], dtype=torch.float32).expand(n, -1)
    d, reward, height = (np.zeros((n, STEPS)) for _ in range(3))
    for t in range(STEPS):
        s = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a = policy(agent, s, z)
        for i, env in enumerate(envs):
            obs[i] = env.step(a[i].numpy()).observation["observations"]
            reward[i, t] = reward_fn(env.physics)
            height[i, t] = env.physics.torso_height()
        d[:, t] = detector.score(agent, s, a, torch.as_tensor(obs, dtype=torch.float32), z).numpy()
    return Path(path).stem, {
        "D": d, "reward": reward, "height": height, "return": reward.sum(1),
        "episode": args.first_episode + np.arange(n),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--detector", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--dynamics", nargs="+", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first_episode", type=int, default=0, help="environment seed of the first episode")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    agent = load_agent(Path(args.checkpoint), cpu)
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    detector = load_detector(Path(args.detector), cpu)
    zs = np.load(args.zs)
    rows = np.flatnonzero((zs["task"] == args.task) & (zs["family"] == "z_r"))
    if len(rows) != 1:
        raise ValueError(f"expected one z_r row for {args.task}, found {len(rows)}")
    reward_constructor = RewardFunctionConstructor(domain_name="walker", task_names=[args.task], seed=0, device=cpu)
    G.update(args=args, agent=agent, detector=detector, z=zs["z"][rows[0]],
             reward_fn=reward_constructor.reward_functions[args.task])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for name, result in pool.imap_unordered(rollout, args.dynamics):
            np.savez(out / f"{name}.npz", **result)
            print(f"{name:24s} D mean {result['D'].mean():8.2f}  median {np.median(result['D']):8.2f}  "
                  f"return {result['return'].mean():6.1f}", flush=True)


if __name__ == "__main__":
    main()
