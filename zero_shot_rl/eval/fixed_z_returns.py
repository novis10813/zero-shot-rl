"""Returns of fixed task vectors inferred from the first n transitions of a data source.

Rupf et al. (arXiv 2510.20264) section 5.2: the least-squares task vector of eq. 4 from the
first n transitions a source collected, evaluated as a fixed latent. Sources:

    --z LABEL npz   a file of zero_shot_rl.eval.online_inference --z_dir, one task vector
                    per run and n
    --rnd LABEL     --trials draws of whole episodes from the dataset (the RND data the
                    checkpoint was pre-trained on), each draw the first n transitions of
                    ceil(max n / 1000) random episodes, labelled with the task reward of the
                    next state; n are the snapshots of the first --z file
    fb              z_r of offline task inference (zero_shot_rl.latent.task_z), always added

Every task vector of run i plays --episodes_per_z deterministic 1000-step episodes from env
seeds i * episodes_per_z + e, so each n and source sees the same initial states; z_r plays
all of them. --out gets one row per episode: source, task, trial, n, env_seed, return.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.fixed_z_returns \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> \
        --z UCB z/ucb:0.1.npz --z Random z/random.npz --rnd RND --out returns.csv
"""

import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from zero_shot_rl.env.walker import STEPS, make_env, run_episodes, set_physics
from zero_shot_rl.latent import task_z
from zero_shot_rl.online.fb import FBModel
from zero_shot_rl.provenance import no_clobber

FIELDS = ["source", "task", "trial", "n", "env_seed", "return"]


def rnd_vectors(args, model, reward_fns, snapshots, device):
    """task -> [trials, len(snapshots), d] from random dataset episodes."""
    episodes = np.load(args.dataset_path, allow_pickle=True)
    keys = sorted(episodes.files)
    rng = np.random.default_rng(args.rnd_seed)
    count = -(-max(snapshots) // STEPS)
    env = make_env(0)
    out = {task: np.zeros((args.trials, len(snapshots), model.d)) for task in args.tasks}
    for i in range(args.trials):
        obs, rewards = [], {task: [] for task in args.tasks}
        for key in rng.choice(keys, count, replace=False):
            episode = episodes[key].item()
            obs.append(episode["observation"][1:])
            for state in episode["physics"][1:]:
                set_physics(env, state)
                for task in args.tasks:
                    rewards[task].append(reward_fns[task](env.physics))
        obs = np.concatenate(obs)
        phi = model.phi(torch.as_tensor(obs, dtype=torch.float32, device=device))
        for task in args.tasks:
            r = torch.as_tensor(rewards[task])
            out[task][i] = [model.lstsq(phi[:n], r[:n]).numpy() for n in snapshots]
        print(f"RND draw {i}", flush=True)
    env.close()
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--z", nargs=2, action="append", default=[], metavar=("LABEL", "NPZ"))
    parser.add_argument("--rnd", type=str, default=None, metavar="LABEL")
    parser.add_argument("--rnd_seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", default=task_z.TASKS)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--episodes_per_z", type=int, default=2)
    parser.add_argument("--batch", type=int, default=200, help="environments stepped at once")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = task_z.device()
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    buffer, observations_z, rewards_z = task_z.inference_set(
        Namespace(seed=args.seed, discount=0.98, dataset_path=args.dataset_path,
                  dataset_transitions=args.dataset_transitions,
                  z_inference_steps=args.z_inference_steps),
        torch.device("cpu"),
    )
    model = FBModel(agent, buffer.storage["next_observations"].to(device))
    reward_fns = task_z.reward_functions(args, device)
    del buffer

    sources = {label: np.load(path) for label, path in args.z}
    snapshots = next(iter(sources.values()))["snapshots"] if sources else np.array([1000])
    for label, data in sources.items():
        if not np.array_equal(data["snapshots"], snapshots):
            raise ValueError(f"{label}: snapshots differ from the first --z file")
    vectors = {label: {t: data[t][: args.trials] for t in args.tasks}
               for label, data in sources.items()}
    if args.rnd:
        vectors[args.rnd] = rnd_vectors(args, model, reward_fns, list(snapshots), device)

    k = args.episodes_per_z
    rows = []
    for task in args.tasks:
        z_r = torch.as_tensor(
            agent.infer_z(observations_z.to(device), rewards_z[task].to(device)),
            dtype=torch.float64,
        )
        jobs = [("fb", i, np.nan, z_r) for i in range(args.trials)]
        for label, by_task in vectors.items():
            jobs += [(label, i, int(n), torch.as_tensor(by_task[task][i, j]))
                     for i in range(args.trials) for j, n in enumerate(snapshots)]
        env_seeds = [i * k + e for _, i, _, _ in jobs for e in range(k)]
        w = torch.stack([v for *_, v in jobs for _ in range(k)]).to(device)
        returns = []
        for c in range(0, len(env_seeds), args.batch):
            envs = [make_env(s) for s in env_seeds[c : c + args.batch]]
            z = model.unit(w[c : c + args.batch])
            returns.extend(run_episodes(agent, z, envs, reward_fns[task])["return"])
            for env in envs:
                env.close()
        for j, (label, i, n, _) in enumerate(jobs):
            for e in range(k):
                rows.append({"source": label, "task": task, "trial": i, "n": n,
                             "env_seed": i * k + e, "return": returns[j * k + e]})
        fb = np.mean([r["return"] for r in rows if r["task"] == task and r["source"] == "fb"])
        print(f"{task}: fb {fb:.1f} over {args.trials * k} episodes", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(out) as path, open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                             for key, v in row.items()})


if __name__ == "__main__":
    main()
