"""One-step transitions of the same (s, a) in the stock and in shifted simulators.

Runs --episodes deterministic 1000-step episodes of the standard FB actor at the stored z_r
of --task in the stock environment (environment seeds from --first_episode). At every step
t = 0 .. 998 it takes (s_t, physics_t, a_t) and, for each --dynamics yaml
(zero_shot_rl.env.dynamics), restores physics_t in an environment with that dynamics, applies
a_t once and records the next observation. A yaml with the stock dynamics reproduces the
rollout, so its output is the check of the replay. Yamls with `events` are not supported:
the replay resets the episode step counter.

Writes <out> (NPZ) with
    observations         [episodes, 999, d_s]   s_t
    actions              [episodes, 999, d_a]   a_t
    physics              [episodes, 999, d_p]   physics_t before a_t
    next_rollout         [episodes, 999, d_s]   s_{t+1} of the stock rollout
    physics_rollout      [episodes, 999, d_p]   physics_{t+1} of the stock rollout
    rewards_rollout      [episodes, 999]        r_t of --task in the stock rollout
    next_<yaml stem>     [episodes, 999, d_s]   s' after the replayed step, one per yaml
    physics_<yaml stem>  [episodes, 999, d_p]   physics' after the replayed step, one per yaml
    rewards_<yaml stem>  [episodes, 999]        r' of --task after the replayed step, one per yaml
    episode              [episodes]             environment seeds

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.paired_step \
        --checkpoint <fb.pt> --zs <zs.npz> --dynamics <a.yaml> <b.yaml> --out <paired.npz>
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
from zero_shot_rl.env.walker import make_env, run_episodes, set_physics

G = {}  # per-process state, filled before the pool forks


def replay(path):
    """s', physics' and task reward of every recorded (physics, action) in `path`."""
    torch.set_num_threads(1)
    physics, actions, reward_fn = G["physics"], G["actions"], G["reward_fn"]
    env = make_env(0, load_dynamics(path))
    out_obs = np.zeros_like(G["next_rollout"])
    out_physics = np.zeros_like(physics)
    out_rewards = np.zeros(physics.shape[:2], dtype=np.float64)
    for i in range(physics.shape[0]):
        for t in range(physics.shape[1]):
            set_physics(env, physics[i, t])
            out_obs[i, t] = env.step(actions[i, t]).observation["observations"]
            out_physics[i, t] = env.physics.get_state().copy()
            out_rewards[i, t] = reward_fn(env.physics)
    return Path(path).stem, out_obs, out_physics, out_rewards


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--dynamics", nargs="+", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first_episode", type=int, default=0, help="environment seed of the first episode")
    parser.add_argument("--workers", type=int, default=5)
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
    z = torch.as_tensor(zs["z"][rows[0]], dtype=torch.float32)

    reward_constructor = RewardFunctionConstructor(
        domain_name="walker",
        task_names=[args.task],
        seed=0,
        device=torch.device("cpu"),
    )
    reward_fn = reward_constructor.reward_functions[args.task]

    envs = [make_env(args.first_episode + e) for e in range(args.episodes)]
    run = run_episodes(agent, z, envs, reward_fn=reward_fn, stride=1, physics=True)
    obs = run["observations"]
    phys = run["physics"]
    G.update(
        physics=phys[:, :-1],
        actions=run["actions"][:, :-1],
        next_rollout=obs[:, 1:],
        reward_fn=reward_fn,
    )
    arrays = {
        "observations": obs[:, :-1],
        "actions": G["actions"],
        "physics": phys[:, :-1],
        "next_rollout": obs[:, 1:],
        "physics_rollout": phys[:, 1:],
        "rewards_rollout": run["rewards"][:, :-1],
        "episode": args.first_episode + np.arange(args.episodes),
    }

    with mp.get_context("fork").Pool(args.workers) as pool:
        for name, nxt_obs, nxt_phys, nxt_rew in pool.imap_unordered(replay, args.dynamics):
            arrays[f"next_{name}"] = nxt_obs
            arrays[f"physics_{name}"] = nxt_phys
            arrays[f"rewards_{name}"] = nxt_rew
            gap = np.abs(nxt_obs - arrays["next_rollout"]).max(-1)
            print(f"{name:24s} |s' - s'_rollout|  median {np.median(gap):.4g}  max {gap.max():.4g}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)


if __name__ == "__main__":
    main()
