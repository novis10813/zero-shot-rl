"""End-to-end finetune + evaluate entry point for Action Transformation (g_0, g_1, T_theta).

Prefer running zero_shot_rl.finetune.action_transform followed by zero_shot_rl.eval.action_transform
for separated finetuning and evaluation; this module preserves the combined CLI interface.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.eval_action_transform \
        --checkpoint <fb.pt> --latents <zs_fb.npz> --g0 <g0_best.pt> \
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
from zero_shot_rl.env.walker import STEPS
from zero_shot_rl.eval.action_transform import METHODS, rollout_eval
from zero_shot_rl.finetune.action_transform import (
    TASKS,
    collect_finetune_transitions,
    finetune_config,
    make_episode_envs,
    resolve_latents,
)
from zero_shot_rl.latent.action_transform import load_feature_predictor
from zero_shot_rl.provenance import no_clobber, sha256

__all__ = ["METHODS", "TASKS", "collect_finetune_transitions", "main", "rollout_eval"]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latents", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Infer z_r from dataset.npz when --latents is omitted")
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=None, help="names of --dynamics (default all)")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--episodes", type=int, default=20, help="eval episodes per task (shared seeds)")
    parser.add_argument("--first-env-seed", type=int, default=5)
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
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.latents is None and args.dataset is None:
        parser.error("Provide --latents or --dataset")

    eval_seeds = set(range(args.first_env_seed, args.first_env_seed + args.episodes))
    ft_seeds = set(range(args.finetune_seed, args.finetune_seed + args.finetune_episodes))
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
    reward = RewardFunctionConstructor(
        domain_name="walker",
        task_names=TASKS,
        seed=42,
        device=device if args.latents is None else torch.device("cpu"),
    ).reward_functions
    z_of, data_hash = resolve_latents(agent, args.latents, args.dataset, args.tasks, device)

    rng = np.random.default_rng(args.seed)
    n_eval = len(args.tasks) * args.episodes
    ret = np.zeros((len(names), len(METHODS), n_eval), np.float32)
    rew = np.zeros((len(names), len(METHODS), n_eval, STEPS), np.float32)
    da = np.zeros_like(rew)
    train_metrics = []
    tasks = seeds = None

    for ci, name in enumerate(names):
        _, transformer, t_info = finetune_config(
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
        train_metrics.append({"config": name, **t_info})
        print(f"{name:22s} finetune g1_loss={t_info['g1_loss']:.4f} dyn={t_info['dyn_loss']:.4f} "
              f"reg={t_info['reg_loss']:.4f}", flush=True)

        for mi, method in enumerate(METHODS):
            envs, z, fns, tasks, seeds = make_episode_envs(
                args.first_env_seed, args.episodes, dynamics[ci], args.tasks, z_of, reward
            )
            try:
                r, d = rollout_eval(agent, envs, z, fns, method, float(gear[index[ci]]), transformer)
            finally:
                for env in envs:
                    env.close()
            rew[ci, mi], da[ci, mi], ret[ci, mi] = r, d, r.sum(1)
            print(f"{name:22s} {method:9s} mean return {ret[ci, mi].mean():7.1f}  "
                  f"mean |a'-a| {d.mean():.3f} elapsed {time.monotonic() - start_time:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(args.out) as temporary:
        np.savez_compressed(
            temporary,
            configs=np.array(names),
            gear=np.asarray(gear, np.float32)[index],
            methods=np.array(METHODS),
            task=tasks,
            episode=seeds,
            ret=ret,
            reward=rew,
            da=da,
        )
    meta = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_sha256=sha256(args.checkpoint),
        **data_hash,
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
