"""Evaluate saved ActionTransformer checkpoints (from zero_shot_rl.finetune.action_transform) on shifted dynamics.

For each dynamics config in --dynamics (filtered by --configs), loads <ft-dir>/<config>/transformer.pt
and evaluates:
    none       unadapted FB actor action a
    star       gear_inverse(a, m): oracle clip(a / m, -1, 1)
    transform  T_theta(s, a) loaded from <ft-dir>/<config>/transformer.pt

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.action_transform \
        --checkpoint <fb.pt> --dataset <dataset.npz> --ft-dir out/ft_aa \
        --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml --out out/eval_aa.npz
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
from zero_shot_rl.env.walker import STEPS, policy
from zero_shot_rl.finetune.action_transform import TASKS, make_episode_envs, resolve_latents
from zero_shot_rl.latent.action_adapt import gear_inverse
from zero_shot_rl.latent.action_transform import load_action_transformer
from zero_shot_rl.provenance import no_clobber, sha256

METHODS = ["none", "star", "transform"]


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
    parser.add_argument("--latents", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Infer z_r from dataset.npz when --latents is omitted")
    parser.add_argument("--ft-dir", type=Path, required=True,
                        help="Directory written by zero_shot_rl.finetune.action_transform")
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=None, help="names of --dynamics (default all)")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--episodes", type=int, default=20, help="eval episodes per task (shared seeds)")
    parser.add_argument("--first-env-seed", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
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
    reward = RewardFunctionConstructor(
        domain_name="walker",
        task_names=TASKS,
        seed=42,
        device=device if args.latents is None else torch.device("cpu"),
    ).reward_functions
    z_of, data_hash = resolve_latents(agent, args.latents, args.dataset, args.tasks, device)

    n_eval = len(args.tasks) * args.episodes
    ret = np.zeros((len(names), len(args.methods), n_eval), np.float32)
    rew = np.zeros((len(names), len(args.methods), n_eval, STEPS), np.float32)
    da = np.zeros_like(rew)
    transformer_hashes = {}
    tasks = seeds = None

    for ci, name in enumerate(names):
        transformer = None
        if "transform" in args.methods:
            t_path = args.ft_dir / name / "transformer.pt"
            transformer, _ = load_action_transformer(t_path, device=device)
            transformer_hashes[name] = sha256(t_path)
        for mi, method in enumerate(args.methods):
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
            methods=np.array(args.methods),
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
        transformer_sha256=transformer_hashes,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        out_sha256=sha256(args.out),
    )
    with args.out.with_suffix(".json").open("x") as stream:
        json.dump(meta, stream, indent=2)
    print(f"Wrote {args.out} elapsed {time.monotonic() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
