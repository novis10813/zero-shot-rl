"""Evaluate two-predictor Action Transformation (g_0, g_1, T_theta) on shifted dynamics.

For each dynamics config in --dynamics (filtered by --configs):
    1. Finetune phase (disjoint --finetune-seed, --finetune-episodes per task):
       Collects transitions (s, a, s') in P_1 using the unadapted FB policy (optionally with
       Gaussian exploration noise --explore-std), computes frozen B_0(s'), fine-tunes g_1 from g_0,
       and trains T_theta(s, a) to minimize |g_1(s, T(s, a)) - g_0(s, a)|^2 + lam |T(s, a) - a|^2.
    2. Evaluation phase (--first-env-seed, --episodes per task):
       Runs the same evaluation episodes with:
           none       unadapted FB actor action a
           star       gear_inverse(a, m): oracle clip(a / m, -1, 1)
           transform  T_theta(s, a)

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.eval_action_transform \\
        --checkpoint <fb.pt> --latents <zs_fb.npz> --g0 <g0_best.pt> \\
        --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml --out <eval.npz>
"""

import json
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.base import load_agent
from rewards import RewardFunctionConstructor
from zero_shot_rl.data.multi_dynamics import dynamics_config, sample_sets
from zero_shot_rl.env.walker import STEPS, make_env, policy
from zero_shot_rl.latent.action_adapt import gear_inverse
from zero_shot_rl.latent.action_transform import (
    compute_backward_features,
    finetune_g1,
    load_feature_predictor,
    train_action_transformer,
)
from zero_shot_rl.latent.archive import matched_rows
from zero_shot_rl.provenance import no_clobber, sha256

METHODS = ["none", "star", "transform"]
TASKS = ["stand", "walk", "run", "flip"]


def collect_finetune_transitions(agent, envs, z, explore_std=0.0, rng=None):
    """Collect (s, a_policy, a_exec, s') from STEPS-step episodes in P_1."""
    obs = np.stack([env.reset().observation["observations"] for env in envs]).astype(np.float32)
    s_list, a_pol_list, a_exec_list, s2_list = [], [], [], []
    for _ in range(STEPS):
        with torch.no_grad():
            a_pol = policy(agent, torch.as_tensor(obs, device=z.device), z).cpu().numpy()
        if explore_std > 0.0:
            noise = rng.normal(0.0, explore_std, size=a_pol.shape).astype(np.float32)
            a_exec = np.clip(a_pol + noise, -1.0, 1.0)
        else:
            a_exec = a_pol
        next_obs = np.zeros_like(obs)
        for i, env in enumerate(envs):
            next_obs[i] = env.step(a_exec[i]).observation["observations"]
        s_list.append(obs.copy())
        a_pol_list.append(a_pol)
        a_exec_list.append(a_exec)
        s2_list.append(next_obs.copy())
        obs = next_obs
    return dict(
        s=np.concatenate(s_list, axis=0),
        a_pol=np.concatenate(a_pol_list, axis=0),
        a_exec=np.concatenate(a_exec_list, axis=0),
        s2=np.concatenate(s2_list, axis=0),
    )


def rollout_eval(agent, envs, z, reward_fns, method, gear, transformer=None):
    """Evaluate one method on `envs` for STEPS steps."""
    n = len(envs)
    device = z.device
    obs = np.stack([env.reset().observation["observations"] for env in envs]).astype(np.float32)
    rewards = np.zeros((n, STEPS), dtype=np.float32)
    da = np.zeros((n, STEPS), dtype=np.float32)
    for t in range(STEPS):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=device)
            a_t = policy(agent, obs_t, z)
            if method == "star":
                exec_t = gear_inverse(a_t, gear)
            elif method == "transform":
                exec_t = transformer(obs_t, a_t)
            else:
                exec_t = a_t
            a = a_t.cpu().numpy()
            executed = exec_t.cpu().numpy()
        da[:, t] = np.linalg.norm(executed - a, axis=-1)
        for i, env in enumerate(envs):
            obs[i] = env.step(executed[i]).observation["observations"]
            rewards[i, t] = reward_fns[i](env.physics)
    return rewards, da


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=None, help="names of --dynamics (default all)")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--episodes", type=int, default=5, help="eval episodes per task")
    parser.add_argument("--first-env-seed", type=int, default=300000)
    parser.add_argument("--finetune-episodes", type=int, default=5, help="finetune episodes per task")
    parser.add_argument("--finetune-seed", type=int, default=320000)
    parser.add_argument("--explore-std", type=float, default=0.0)
    parser.add_argument("--g1-steps", type=int, default=2000)
    parser.add_argument("--g1-lr", type=float, default=1e-4)
    parser.add_argument("--t-steps", type=int, default=2000)
    parser.add_argument("--t-lr", type=float, default=3e-4)
    parser.add_argument("--t-hidden", type=int, default=256)
    parser.add_argument("--t-layers", type=int, default=3)
    parser.add_argument("--pretrain-steps", type=int, default=500)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")

    eval_seeds = set(range(args.first_env_seed, args.first_env_seed + len(args.tasks) * args.episodes))
    ft_seeds = set(range(args.finetune_seed, args.finetune_seed + len(args.tasks) * args.finetune_episodes))
    if eval_seeds & ft_seeds:
        raise ValueError("finetune seeds must not overlap evaluation seeds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.monotonic()

    config = yaml.safe_load(args.dynamics.read_text())
    friction, gear = sample_sets(config)
    all_names = [v["name"] for v in config["values"]]
    names = args.configs or all_names
    index = [all_names.index(n) for n in names]
    dynamics = [dynamics_config(friction[i], np.full(6, gear[i])) for i in index]

    agent = load_agent(args.checkpoint, device)
    agent.eval()
    g0, _ = load_feature_predictor(args.g0, device=device)
    g0.requires_grad_(False)
    reward = RewardFunctionConstructor(domain_name="walker", task_names=TASKS, seed=42,
                                       device=torch.device("cpu")).reward_functions
    with np.load(args.latents, allow_pickle=False) as archive:
        data = dict(archive)
    z_of = {t: torch.as_tensor(data["z"][matched_rows(data, [t], ["z_r"], None)[0]],
                               dtype=torch.float32, device=device)
            for t in args.tasks}

    def make_episodes(first_seed, per_task, dyn):
        seeds = np.concatenate([first_seed + per_task * i + np.arange(per_task) for i in range(len(args.tasks))])
        tasks = np.repeat(args.tasks, per_task)
        z = torch.stack([z_of[t] for t in tasks])
        return [make_env(int(s), dyn) for s in seeds], z, [reward[t] for t in tasks], tasks, seeds

    rng = np.random.default_rng(args.seed)
    n_eval = len(args.tasks) * args.episodes
    ret = np.zeros((len(names), len(METHODS), n_eval), np.float32)
    rew = np.zeros((len(names), len(METHODS), n_eval, STEPS), np.float32)
    da = np.zeros_like(rew)
    train_metrics = []

    for ci, name in enumerate(names):
        ft_envs, ft_z, _, _, _ = make_episodes(args.finetune_seed, args.finetune_episodes, dynamics[ci])
        ft_data = collect_finetune_transitions(agent, ft_envs, ft_z, explore_std=args.explore_std, rng=rng)
        for env in ft_envs:
            env.close()

        b_next = compute_backward_features(agent, ft_data["s2"], device)
        g1, g1_loss = finetune_g1(g0, ft_data["s"], ft_data["a_exec"], b_next,
                                  steps=args.g1_steps, batch_size=args.batch_size, lr=args.g1_lr)
        transformer, t_info = train_action_transformer(
            g0, g1, ft_data["s"], ft_data["a_pol"], lam=args.lam, steps=args.t_steps,
            batch_size=args.batch_size, lr=args.t_lr, hidden=args.t_hidden, layers=args.t_layers,
            pretrain_steps=args.pretrain_steps, seed=args.seed,
        )
        train_metrics.append({"config": name, "g1_loss": g1_loss, **t_info})
        print(f"{name:22s} finetune g1_loss={g1_loss:.4f} dyn={t_info['dyn_loss']:.4f} "
              f"reg={t_info['reg_loss']:.4f}", flush=True)

        for mi, method in enumerate(METHODS):
            envs, z, fns, tasks, seeds = make_episodes(args.first_env_seed, args.episodes, dynamics[ci])
            r, d = rollout_eval(agent, envs, z, fns, method, float(gear[index[ci]]), transformer)
            rew[ci, mi], da[ci, mi], ret[ci, mi] = r, d, r.sum(1)
            print(f"{name:22s} {method:9s} mean return {ret[ci, mi].mean():7.1f}  mean |a'-a| {d.mean():.3f} "
                  f"elapsed {time.monotonic() - start_time:.0f}s", flush=True)
            for env in envs:
                env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(args.out) as temporary:
        np.savez_compressed(temporary, configs=np.array(names), gear=np.asarray(gear, np.float32)[index],
                            methods=np.array(METHODS), task=tasks, episode=seeds, ret=ret, reward=rew, da=da)
    meta = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_sha256=sha256(args.checkpoint),
        latents_sha256=sha256(args.latents),
        g0_sha256=sha256(args.g0),
        train_metrics=train_metrics,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        out_sha256=sha256(args.out),
    )
    with args.out.with_suffix(".json").open("x") as stream:
        json.dump(meta, stream, indent=2)
    print(f"Wrote {args.out} elapsed {time.monotonic() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
