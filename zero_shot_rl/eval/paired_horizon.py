"""K-step trajectories from the same states in the stock and in shifted simulators.

Runs --episodes deterministic 1000-step episodes of the standard FB actor at the stored z_r of
--task in the stock environment (environment seeds from --first_episode) and records the
physics state of every step. From each start step t0 in --starts of each episode, it restores
physics_t0 in an environment with each --dynamics yaml (zero_shot_rl.env.dynamics) and runs
--horizon steps twice:

    open loop     the actions a_t0 .. a_t0+K-1 of the stock rollout
    closed loop   the actor's own action at every step, a = pi(s, z_r) of the current observation

so the trajectories of two yamls differ only by the dynamics (open loop), or by the dynamics and
the policy's response to it (closed loop). A yaml with the stock dynamics reproduces the rollout
in open loop, which is the check of the replay. Yamls with `events` are not supported: the replay
resets the episode step counter. Start steps must leave room for the horizon (t0 + K <= 1000).

Writes <out> (NPZ) with
    observations      [episodes, starts, d_s]        s_t0
    starts            [starts]                       t0
    episode           [episodes]                     environment seeds
    rollout           [episodes, starts, K, d_s]     s_t0+1 .. s_t0+K of the stock rollout
    open_<stem>       [episodes, starts, K, d_s]     open-loop observations after each step, one per yaml
    closed_<stem>     [episodes, starts, K, d_s]     closed-loop observations after each step
    closed_actions_<stem>  [episodes, starts, K, d_a]  the actions of the closed loop

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.paired_horizon \
        --checkpoint <fb.pt> --zs <zs.npz> --dynamics <a.yaml> <b.yaml> --out <horizon.npz>
"""

import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from zero_shot_rl.env.dynamics import load_dynamics
from zero_shot_rl.env.walker import STEPS, act, make_env, run_episodes, set_physics

G = {}  # per-process state, filled before the pool forks


def open_loop(env, state, actions):
    """Observations after each of `actions` [K, d_a] applied from the physics `state`."""
    set_physics(env, state)
    return np.stack([env.step(a).observation["observations"] for a in actions])


def closed_loop(env, state, obs, policy, horizon):
    """Observations and actions of `horizon` steps of `policy` (obs -> action) from `state` and its observation `obs`."""
    set_physics(env, state)
    observations, actions = [], []
    for _ in range(horizon):
        a = policy(obs)
        obs = env.step(a).observation["observations"]
        observations.append(obs)
        actions.append(a)
    return np.stack(observations), np.stack(actions)


def replay(path):
    """Open- and closed-loop trajectories of every (episode, start) in the dynamics of the yaml `path`."""
    torch.set_num_threads(1)
    physics, actions, obs, starts, horizon = G["physics"], G["actions"], G["observations"], G["starts"], G["horizon"]
    env = make_env(0, load_dynamics(path))
    policy = lambda o: act(G["agent"], o, G["z"])  # noqa: E731
    n, d_s, d_a = physics.shape[0], obs.shape[-1], actions.shape[-1]
    opened = np.zeros((n, len(starts), horizon, d_s))
    closed = np.zeros_like(opened)
    closed_actions = np.zeros((n, len(starts), horizon, d_a), np.float32)
    for i in range(n):
        for j, t0 in enumerate(starts):
            opened[i, j] = open_loop(env, physics[i, t0], actions[i, t0:t0 + horizon])
            closed[i, j], closed_actions[i, j] = closed_loop(env, physics[i, t0], obs[i, t0], policy, horizon)
    return Path(path).stem, opened, closed, closed_actions


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--dynamics", nargs="+", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first_episode", type=int, default=0, help="environment seed of the first episode")
    parser.add_argument("--starts", nargs="+", type=int, default=list(range(0, 900, 50)))
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if max(args.starts) + args.horizon > STEPS:
        raise ValueError(f"start + horizon must be <= {STEPS}")

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

    envs = [make_env(args.first_episode + e) for e in range(args.episodes)]
    run = run_episodes(agent, z, envs, stride=1, physics=True)
    obs = run["observations"]
    starts = np.array(args.starts)
    G.update(agent=agent, z=z, physics=run["physics"], actions=run["actions"], observations=obs,
             starts=starts, horizon=args.horizon)
    after = np.concatenate([obs[:, 1:], np.full_like(obs[:, :1], np.nan)], 1)  # s_t+1; the last is not recorded
    steps = starts[:, None] + np.arange(args.horizon)
    arrays = {"observations": obs[:, starts], "starts": starts, "rollout": after[:, steps],
              "episode": args.first_episode + np.arange(args.episodes)}

    with mp.get_context("fork").Pool(args.workers) as pool:
        for name, opened, closed, closed_actions in pool.imap_unordered(replay, args.dynamics):
            arrays[f"open_{name}"] = opened
            arrays[f"closed_{name}"] = closed
            arrays[f"closed_actions_{name}"] = closed_actions
            gap = np.nanmax(np.abs(opened - arrays["rollout"]), -1)
            print(f"{name:24s} open-loop |s - s_rollout|  median {np.nanmedian(gap):.4g}  "
                  f"max {np.nanmax(gap):.4g}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)


if __name__ == "__main__":
    main()
