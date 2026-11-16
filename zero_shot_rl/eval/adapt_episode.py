"""Closed-loop return of the action adaptation of zero_shot_rl.latent.action_adapt under shifted gear.

For one FB training seed, every --tasks task (z_r from --latents), --episodes episodes per task
(environment seeds --first-env-seed + episodes * task index + j), 1000 steps in each dynamics of
--dynamics listed in --configs (the first being the stock walker). The same episodes are run with

    none    the FB actor's action a
    star    gear_inverse(a, m): the best action inside [-1, 1] for the gear multiplier m of the
            dynamics, an oracle that knows m
    adapt   a' = adapt(model, s_t, a, c1, c0, --k, --lr) with c1 = E(the last --h executed
            transitions (s, a', s_next - s)) as soon as --h steps exist (a' = a before), c0 the
            context of the stock walker: the mean of the window contexts of --calib-episodes
            unadapted stock episodes per task (environment seeds --calib-seed + ...)

The return is the task reward summed over the episode (rewards.RewardFunctionConstructor).
Writes --out (NPZ):

    configs [C], gear [C], methods ["none", "star", "adapt"]
    task, episode [N]                 the episodes (shared by all configs and methods)
    ret [C, 3, N]                     returns
    reward [C, 3, N, 1000]            reward after each step
    da [C, 3, N, 1000]                |a_executed - a| of each step

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.adapt_episode --checkpoint <fb.pt> \\
        --latents <zs_fb.npz> --model <context_best.pt> \\
        --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml --out <adapt_episode.npz>
"""

import json
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from agents.base import load_agent
from rewards import RewardFunctionConstructor
from zero_shot_rl.data.multi_dynamics import dynamics_config, sample_sets
from zero_shot_rl.env.walker import STEPS, make_env, policy
from zero_shot_rl.latent.action_adapt import adapt, gear_inverse
from zero_shot_rl.latent.archive import matched_rows
from zero_shot_rl.latent.context_encoder import load_context_model
from zero_shot_rl.provenance import no_clobber, sha256

METHODS = ["none", "star", "adapt"]
TASKS = ["stand", "walk", "run", "flip"]


def window(states, actions, t, h):
    """The h transitions before step t: states [n, T + 1, d_s], actions [n, T, d_a] -> (s, a, s_next - s)."""
    s = states[:, t - h:t]
    return s, actions[:, t - h:t], states[:, t - h + 1:t + 1] - s


def rollout(agent, envs, z, reward_fns, method, gear, model, c0, h, k, lr):
    """STEPS-step episodes of `envs` in lockstep; z [n, d_z], reward_fns one per env.

    Returns states [n, STEPS + 1, d_s], actions (executed) [n, STEPS, d_a], rewards and da [n, STEPS].
    """
    n = len(envs)
    obs = np.stack([env.reset().observation["observations"] for env in envs]).astype(np.float32)
    states = np.zeros((n, STEPS + 1, obs.shape[1]), np.float32)
    states[:, 0] = obs
    actions, rewards, da = None, np.zeros((n, STEPS)), np.zeros((n, STEPS))
    for t in range(STEPS):
        with torch.no_grad():
            a = policy(agent, torch.as_tensor(obs), z).numpy()
        if actions is None:
            actions = np.zeros((n, STEPS, a.shape[1]), np.float32)
        executed = a
        if method == "star":
            executed = gear_inverse(torch.as_tensor(a), gear).numpy()
        elif method == "adapt" and t >= h:
            s_w, a_w, d_w = (torch.as_tensor(x) for x in window(states, actions, t, h))
            with torch.no_grad():
                c1 = model.encode(s_w, a_w, d_w)
            executed = adapt(model, torch.as_tensor(obs), torch.as_tensor(a), c1, c0, k, lr)[0].numpy()
        actions[:, t] = executed
        da[:, t] = np.linalg.norm(executed - a, axis=-1)
        for i, env in enumerate(envs):
            obs[i] = env.step(executed[i]).observation["observations"]
            rewards[i, t] = reward_fns[i](env.physics)
        states[:, t + 1] = obs
    return states, actions, rewards, da


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=None, help="names of --dynamics (default all)")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--episodes", type=int, default=5, help="per task")
    parser.add_argument("--first-env-seed", type=int, default=300000)
    parser.add_argument("--calib-episodes", type=int, default=5, help="per task")
    parser.add_argument("--calib-seed", type=int, default=310000)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8, help="of the calibration windows")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    torch.set_num_threads(4)
    start_time = time.monotonic()

    config = yaml.safe_load(args.dynamics.read_text())
    friction, gear = sample_sets(config)
    all_names = [v["name"] for v in config["values"]]
    names = args.configs or all_names
    if names[0] != all_names[0]:
        raise ValueError("the first config must be the stock walker")
    index = [all_names.index(n) for n in names]
    dynamics = [dynamics_config(friction[i], np.full(6, gear[i])) for i in index]
    agent = load_agent(args.checkpoint, torch.device("cpu"))
    agent.eval()
    model, _ = load_context_model(args.model)
    model.requires_grad_(False)
    reward = RewardFunctionConstructor(domain_name="walker", task_names=TASKS, seed=42,
                                       device=torch.device("cpu")).reward_functions
    with np.load(args.latents, allow_pickle=False) as archive:
        data = dict(archive)
    z_of = {t: torch.as_tensor(data["z"][matched_rows(data, [t], ["z_r"], None)[0]], dtype=torch.float32)
            for t in args.tasks}

    def episodes(first_seed, per_task, dyn):
        seeds = np.concatenate([first_seed + per_task * i + np.arange(per_task) for i in range(len(args.tasks))])
        tasks = np.repeat(args.tasks, per_task)
        z = torch.stack([z_of[t] for t in tasks])
        return [make_env(int(s), dyn) for s in seeds], z, [reward[t] for t in tasks], tasks, seeds

    # Stock context c0 from unadapted stock episodes
    envs, z, fns, _, _ = episodes(args.calib_seed, args.calib_episodes, dynamics[0])
    states, actions, _, _ = rollout(agent, envs, z, fns, "none", 1.0, None, None, args.h, args.k, args.lr)
    with torch.no_grad():
        c = [model.encode(*(torch.as_tensor(x) for x in window(states, actions, t, args.h)))
             for t in range(args.h, STEPS, args.stride)]
    c0 = F.normalize(torch.cat(c).mean(0, keepdim=True), dim=-1)
    print(f"calibration {len(envs)} episodes elapsed {time.monotonic() - start_time:.0f}s", flush=True)

    n = len(args.tasks) * args.episodes
    ret = np.zeros((len(names), len(METHODS), n), np.float32)
    rew = np.zeros((len(names), len(METHODS), n, STEPS), np.float32)
    da = np.zeros_like(rew)
    for ci, name in enumerate(names):
        for mi, method in enumerate(METHODS):
            envs, z, fns, tasks, seeds = episodes(args.first_env_seed, args.episodes, dynamics[ci])
            _, _, r, d = rollout(agent, envs, z, fns, method, float(gear[index[ci]]), model, c0,
                                 args.h, args.k, args.lr)
            rew[ci, mi], da[ci, mi], ret[ci, mi] = r, d, r.sum(1)
            print(f"{name:22s} {method:6s} mean return {ret[ci, mi].mean():7.1f}  mean |a'-a| {d.mean():.3f} "
                  f"elapsed {time.monotonic() - start_time:.0f}s", flush=True)
            for env in envs:
                env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(args.out) as temporary:
        np.savez_compressed(temporary, configs=np.array(names), gear=np.asarray(gear, np.float32)[index],
                            methods=np.array(METHODS), task=tasks, episode=seeds, ret=ret, reward=rew, da=da)
    meta = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_sha256=sha256(args.checkpoint), latents_sha256=sha256(args.latents),
        model_sha256=sha256(args.model),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        out_sha256=sha256(args.out))
    with args.out.with_suffix(".json").open("x") as stream:
        json.dump(meta, stream, indent=2)
    print(f"Wrote {args.out} elapsed {time.monotonic() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
