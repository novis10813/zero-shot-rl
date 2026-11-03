"""Roll out the standard FB actor at the stored z_r of a task in shifted environments.

Each --dynamics yaml (zero_shot_rl.env.dynamics) runs --episodes deterministic episodes of
1000 steps (clipped mean action). Episode e (from --first_episode) starts from the reset of
an environment seeded with e, so all yamls share the same start states. Per episode it records

    return     sum of the task reward
    speed      mean horizontal velocity of the centre of mass
    fall_frac  fraction of steps with torso height < 0.6

--zs is a zero_shot_rl.latent.infer file; its `z_r` row for --task is the latent.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.dynamics_sweep \
        --checkpoint <fb.pt> --zs <zs.npz> --dynamics <a.yaml> <b.yaml> --out <rollouts.csv>
"""

import csv
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from rewards import RewardFunctionConstructor
from zero_shot_rl.env.dynamics import load_dynamics
from zero_shot_rl.env.walker import make_env, policy

FALL_HEIGHT = 0.6
FIELDS = ["dynamics", "episode", "return", "speed", "fall_frac"]

G = {}  # per-process state, filled before the pool forks


def rollout(path):
    torch.set_num_threads(1)
    args, agent, reward_fn = G["args"], G["agent"], G["reward_fn"]
    dynamics = load_dynamics(path)
    n = args.episodes
    envs = [make_env(args.first_episode + e, dynamics) for e in range(n)]
    obs = np.stack([env.reset().observation["observations"] for env in envs])
    z = torch.as_tensor(G["z"], dtype=torch.float32).expand(n, -1)
    rewards, heights, speeds = (np.zeros((n, 1000)) for _ in range(3))
    for t in range(1000):
        with torch.no_grad():
            actions = policy(agent, torch.as_tensor(obs, dtype=torch.float32), z).numpy()
        for i, env in enumerate(envs):
            obs[i] = env.step(actions[i]).observation["observations"]
            p = env.physics
            rewards[i, t] = reward_fn(p)
            heights[i, t] = p.torso_height()
            speeds[i, t] = p.horizontal_velocity()
    return [
        {"dynamics": Path(path).stem, "episode": args.first_episode + e, "return": rewards[e].sum(),
         "speed": speeds[e].mean(), "fall_frac": (heights[e] < FALL_HEIGHT).mean()}
        for e in range(n)
    ]


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--dynamics", nargs="+", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--first_episode", type=int, default=0, help="environment seed of the first episode")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(1)
    agent = load_agent(Path(args.checkpoint), torch.device("cpu"))
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    zs = np.load(args.zs)
    rows = np.flatnonzero((zs["task"] == args.task) & (zs["family"] == "z_r"))
    if len(rows) != 1:
        raise ValueError(f"expected one z_r row for {args.task}, found {len(rows)}")
    reward_constructor = RewardFunctionConstructor(
        domain_name="walker", task_names=[args.task], seed=0, device=torch.device("cpu")
    )
    G.update(args=args, agent=agent, z=zs["z"][rows[0]],
             reward_fn=reward_constructor.reward_functions[args.task])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f, mp.get_context("fork").Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for result in pool.imap_unordered(rollout, args.dynamics):
            for rec in result:
                writer.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                                 for k, v in rec.items()})
            f.flush()
            print(f"{result[0]['dynamics']:24s} return {np.mean([r['return'] for r in result]):6.1f}", flush=True)


if __name__ == "__main__":
    main()
