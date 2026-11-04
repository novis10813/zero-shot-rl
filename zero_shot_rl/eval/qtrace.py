"""Does the ensemble Q rank a_opt and a_pess correctly, and how large is its spread, in a
(shifted) environment?

Per episode (environment seed e from --first_episode, optional --dynamics yaml) the
deterministic actor at z_r drives the episode for 1000 steps. At every step it records, with
F_k the members of a zero_shot_rl.train.ensemble ensemble,

    q_opt, q_pess  mean_k F_k(s, a, z_r)^T z_r for a_opt = pi(s, z_r), a_pess = pi(s, z_cql)
    u_opt, u_pess  std_k of the same
    reward         task reward of the next state

z_r is the unit latent, so q and u are in units of |z_hat_r| (zero_shot_rl.eval.switch).
Every --every steps, up to the last step that still has --horizon steps left, it also
branches from the state: step once with a_opt, or once with a_pess, then follow
pi(., z_r) for the remaining steps of --horizon, and records

    g_opt, g_pess  sum_k gamma^k r_{t+k+1}, gamma = 0.98, over --horizon steps

Q(s, a) estimates this return, so the sign and size of (q_opt - q_pess) can be compared with
g_opt - g_pess. The simulator is deterministic, so the two branches differ only by the first
action. The return after --horizon steps is not counted.

Writes an NPZ with q_opt, q_pess, u_opt, u_pess, reward of shape [episodes, 1000], g_opt,
g_pess of shape [episodes, branches], and branch_steps, episodes.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.qtrace \
        --checkpoint <fb.pt> --ensemble <ensemble.pt> --zs <zs_sweep.npz> \
        [--dynamics <yaml>] --out <qtrace.npz>
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
from zero_shot_rl.env.walker import make_env, policy, set_physics
from zero_shot_rl.latent.switch import latents, values
from zero_shot_rl.provenance import sha256
from zero_shot_rl.train.ensemble import load_ensemble

DISCOUNT = 0.98  # agents/fb/config.yaml
STEPS = 1000

G = {}  # per-process state, filled before the pool forks


def trace(episodes):
    torch.set_num_threads(1)
    args, agent, ensemble, reward_fn = G["args"], G["agent"], G["ensemble"], G["reward_fn"]
    n = len(episodes)
    envs = [make_env(e, args.dynamics) for e in episodes]
    branches = [make_env(e, args.dynamics) for e in episodes]
    obs = np.stack([env.reset().observation["observations"] for env in envs])
    z_r, z_cql = (torch.as_tensor(z, dtype=torch.float32).expand(n, -1) for z in G["z"])
    branch_steps = list(range(0, STEPS - args.horizon + 1, args.every))

    out = {k: np.zeros((n, STEPS)) for k in ("q_opt", "q_pess", "u_opt", "u_pess", "reward")}
    g_opt, g_pess = (np.zeros((n, len(branch_steps))) for _ in range(2))
    for t in range(STEPS):
        s = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a_opt, a_pess = policy(agent, s, z_r), policy(agent, s, z_cql)
            out["q_opt"][:, t], out["u_opt"][:, t] = values(ensemble, s, a_opt, z_r)
            out["q_pess"][:, t], out["u_pess"][:, t] = values(ensemble, s, a_pess, z_r)
        if t in branch_steps:
            states = [env.physics.get_state().copy() for env in envs]
            for first, g in ((a_opt, g_opt), (a_pess, g_pess)):
                g[:, branch_steps.index(t)] = branch(agent, branches, states, first, z_r, reward_fn, args.horizon)
        for i, env in enumerate(envs):
            obs[i] = env.step(a_opt[i].numpy()).observation["observations"]
            out["reward"][i, t] = reward_fn(env.physics)
    return {**out, "g_opt": g_opt, "g_pess": g_pess}


def branch(agent, envs, states, first_actions, z, reward_fn, horizon):
    """Return of each (state, first action), then pi(., z) for `horizon` steps in total."""
    returns = np.zeros(len(envs))
    obs = np.zeros((len(envs), G["obs_dim"]), dtype=np.float32)
    for i, (env, state) in enumerate(zip(envs, states)):
        set_physics(env, state)
        obs[i] = env.step(first_actions[i].numpy()).observation["observations"]
        returns[i] = reward_fn(env.physics)
    for k in range(1, horizon):
        with torch.no_grad():
            actions = policy(agent, torch.as_tensor(obs), z).numpy()
        for i, env in enumerate(envs):
            obs[i] = env.step(actions[i]).observation["observations"]
            returns[i] += DISCOUNT**k * reward_fn(env.physics)
    return returns


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ensemble", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--angle", type=float, default=7.5)
    parser.add_argument("--dynamics", type=str, default=None,
                        help="yaml of zero_shot_rl.env.dynamics; default is the stock environment")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--first_episode", type=int, default=0, help="environment seed of the first episode")
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.dynamics = load_dynamics(args.dynamics) if args.dynamics else None

    torch.set_num_threads(1)
    device = torch.device("cpu")
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    meta = torch.load(args.ensemble, map_location=device, weights_only=True)
    if meta["checkpoint_sha256"] != sha256(Path(args.checkpoint)):
        raise ValueError(f"{args.ensemble} was trained for another checkpoint")
    ensemble = load_ensemble(Path(args.ensemble), device).eval()
    reward_fn = RewardFunctionConstructor(
        domain_name="walker", task_names=[args.task], seed=0, device=device
    ).reward_functions[args.task]
    probe_env = make_env(0)
    G.update(args=args, agent=agent, ensemble=ensemble, reward_fn=reward_fn,
             z=latents(np.load(args.zs), args.task, args.angle),
             obs_dim=probe_env.reset().observation["observations"].shape[0])

    episodes = np.arange(args.first_episode, args.first_episode + args.episodes)
    chunks = [c for c in np.array_split(episodes, args.workers) if len(c)]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        parts = pool.map(trace, chunks)
    result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    result.update(episodes=episodes, branch_steps=np.arange(0, STEPS - args.horizon + 1, args.every))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in result.items()})
    print(f"{out} q gap mean {np.mean(result['q_opt'] - result['q_pess']):.4f} "
          f"g gap mean {np.mean(result['g_opt'] - result['g_pess']):.4f}", flush=True)


if __name__ == "__main__":
    main()
