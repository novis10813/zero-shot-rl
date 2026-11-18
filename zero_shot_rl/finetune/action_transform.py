"""Test-time finetuning of two-predictor Action Transformation (g_1, T_theta) on shifted dynamics.

Given a pretrained FB checkpoint and offline stock predictor g_0:
    1. Collects finetune transitions (s, a_pol, a_exec, s') in each shifted dynamics config
       using --finetune-seed .. --finetune-seed + --finetune-episodes - 1.
    2. Fine-tunes g_1 from g_0 on (s, a_exec) -> B_0(s').
    3. Trains ActionTransformer T_theta(s, a) to minimize
       |g_1(s, T(s, a)) - g_0(s, a)|^2 + lam |T(s, a) - a|^2.
    4. Saves <out-dir>/<config>/transformer.pt (and optionally g1.pt) plus finetune_meta.json.

Evaluate saved transformers separately with zero_shot_rl.eval.action_transform.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.finetune.action_transform \
        --checkpoint <fb.pt> --dataset <dataset.npz> --g0 <g0_best.pt> \
        --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml --out-dir out/ft_aa
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
from zero_shot_rl.latent.action_transform import (
    compute_backward_features,
    finetune_g1,
    load_feature_predictor,
    train_action_transformer,
)
from zero_shot_rl.latent.archive import matched_rows
from zero_shot_rl.provenance import sha256

TASKS = ["stand", "walk", "run", "flip"]


def resolve_latents(agent, latents_path, dataset_path, tasks, device):
    """Resolve task vectors z_of[task] from either a stored latent NPZ or an ExORL dataset."""
    if latents_path is not None:
        with np.load(latents_path, allow_pickle=False) as archive:
            data = dict(archive)
        z_of = {
            t: torch.as_tensor(
                data["z"][matched_rows(data, [t], ["z_r"], None)[0]],
                dtype=torch.float32,
                device=device,
            )
            for t in tasks
        }
        data_hash = {"latents_sha256": sha256(latents_path)}
        return z_of, data_hash

    from agents.fb.replay_buffer import FBReplayBuffer
    from zero_shot_rl.utils import set_seed_everywhere

    set_seed_everywhere(42)
    rc = RewardFunctionConstructor(domain_name="walker", task_names=TASKS, seed=42, device=device)
    buf = FBReplayBuffer(
        reward_constructor=rc,
        dataset_path=dataset_path,
        transitions=100000,
        relabel=False,
        task=None,
        device=device,
        discount=0.98,
        action_condition=None,
    )
    obs_z, rew_z = buf.sample_task_inference_transitions(inference_steps=10000)
    with torch.no_grad():
        z_of = {
            t: torch.as_tensor(
                agent.infer_z(obs_z.to(device), rew_z[t].to(device)),
                dtype=torch.float32,
                device=device,
            ).reshape(-1)
            for t in tasks
        }
    return z_of, {"dataset_sha256": sha256(dataset_path)}


def make_episode_envs(first_seed, per_task, dyn, tasks, z_of, reward_fns):
    """Build deterministic Walker environments, stacked task latents, and reward functions."""
    seeds = np.tile(first_seed + np.arange(per_task), len(tasks))
    ep_tasks = np.repeat(tasks, per_task)
    z = torch.stack([z_of[t] for t in ep_tasks])
    envs = [make_env(int(s), dyn) for s in seeds]
    fns = [reward_fns[t] for t in ep_tasks]
    return envs, z, fns, ep_tasks, seeds


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


def finetune_config(
    agent,
    g0,
    dyn,
    z_of,
    tasks,
    reward_fns,
    finetune_episodes=10,
    finetune_seed=1000,
    explore_std=0.0,
    rng=None,
    g1_steps=2000,
    g1_lr=1e-4,
    t_steps=2000,
    t_lr=3e-4,
    t_hidden=256,
    t_layers=3,
    pretrain_steps=0,
    lam=0.01,
    batch_size=256,
    seed=0,
):
    """Run test-time g_1 + T_theta finetuning for one shifted dynamics config."""
    device = z_of[tasks[0]].device
    ft_envs, ft_z, _, _, _ = make_episode_envs(
        finetune_seed, finetune_episodes, dyn, tasks, z_of, reward_fns
    )
    try:
        ft_data = collect_finetune_transitions(agent, ft_envs, ft_z, explore_std=explore_std, rng=rng)
    finally:
        for env in ft_envs:
            env.close()

    b_next = compute_backward_features(agent, ft_data["s2"], device)
    g1, g1_loss = finetune_g1(
        g0, ft_data["s"], ft_data["a_exec"], b_next,
        steps=g1_steps, batch_size=batch_size, lr=g1_lr,
    )
    transformer, t_info = train_action_transformer(
        g0, g1, ft_data["s"], ft_data["a_pol"],
        lam=lam, steps=t_steps, batch_size=batch_size, lr=t_lr,
        hidden=t_hidden, layers=t_layers, pretrain_steps=pretrain_steps, seed=seed,
    )
    metrics = {"g1_loss": g1_loss, **t_info}
    return g1, transformer, metrics


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latents", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Infer z_r from dataset.npz when --latents is omitted")
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=None, help="names of --dynamics (default all)")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--finetune-episodes", type=int, default=10, help="finetune episodes per task")
    parser.add_argument("--finetune-seed", type=int, default=1000)
    parser.add_argument("--explore-std", type=float, default=0.0)
    parser.add_argument("--g1-steps", type=int, default=2000)
    parser.add_argument("--g1-lr", type=float, default=1e-4)
    parser.add_argument("--t-steps", type=int, default=2000)
    parser.add_argument("--t-lr", type=float, default=3e-4)
    parser.add_argument("--t-hidden", type=int, default=256)
    parser.add_argument("--t-layers", type=int, default=3)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-g1", action="store_true", help="Also save fine-tuned g_1 checkpoints")
    args = parser.parse_args()
    if args.latents is None and args.dataset is None:
        parser.error("Provide --latents or --dataset")

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
    reward = RewardFunctionConstructor(
        domain_name="walker",
        task_names=TASKS,
        seed=42,
        device=device if args.latents is None else torch.device("cpu"),
    ).reward_functions
    z_of, data_hash = resolve_latents(agent, args.latents, args.dataset, args.tasks, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train_metrics = []
    for ci, name in enumerate(names):
        g1, transformer, info = finetune_config(
            agent, g0, dynamics[ci], z_of, args.tasks, reward,
            finetune_episodes=args.finetune_episodes,
            finetune_seed=args.finetune_seed,
            explore_std=args.explore_std,
            rng=rng,
            g1_steps=args.g1_steps,
            g1_lr=args.g1_lr,
            t_steps=args.t_steps,
            t_lr=args.t_lr,
            t_hidden=args.t_hidden,
            t_layers=args.t_layers,
            pretrain_steps=args.pretrain_steps,
            lam=args.lam,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        cfg_dir = args.out_dir / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_meta = {"config": name, "gear": float(gear[index[ci]]), "lam": args.lam,
                    "seed": args.seed, **info}
        transformer.cpu().save(cfg_dir / "transformer.pt", **cfg_meta)
        if args.save_g1:
            g1.cpu().save(cfg_dir / "g1.pt", **cfg_meta)
        train_metrics.append(cfg_meta)
        print(f"{name:22s} finetune g1_loss={info['g1_loss']:.4f} dyn={info['dyn_loss']:.4f} "
              f"reg={info['reg_loss']:.4f} elapsed {time.monotonic() - start_time:.0f}s", flush=True)

    meta = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_sha256=sha256(args.checkpoint),
        **data_hash,
        g0_sha256=sha256(args.g0),
        configs=names,
        train_metrics=train_metrics,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
    )
    with (args.out_dir / "finetune_meta.json").open("w") as stream:
        json.dump(meta, stream, indent=2)
    print(f"Wrote {args.out_dir} elapsed {time.monotonic() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
