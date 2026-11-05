"""
Distance of FB and VC-FB actions to the dataset behaviour along whole episodes.

Every checkpoint runs --episodes deterministic episodes of 1000 steps per task (episode e
starts from the reset of an environment seeded with e, so start states are shared). The
states s_t visited every --stride steps are kept. Two readings, both scored with the
distances of zero_shot_rl.eval.behavior_distance:

    own    D(s_t, a_t) of the checkpoint's own visited states and actions.
    cross  every checkpoint acts at the states visited by every other checkpoint, so FB and
           VC-FB are compared at identical states (rows with the same pool).

Writes one CSV row per (pool checkpoint, acting checkpoint, task, episode, step): the distances
of the acting checkpoint's action at that visited state, plus the episode return of the pool
checkpoint. Figures are scatter plots, one panel per task and one dot per visited state:
FB seed k on x against VC-FB seed k on y, both acting at the states visited by the pool
algorithm's seed k (one figure per pool algorithm and metric). Below the diagonal means the
VC-FB action is closer to the data.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.behavior_rollout \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out rollout_distances.csv --figure_prefix rollout
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rewards import RewardFunctionConstructor
from zero_shot_rl.env.walker import make_env, policy
from zero_shot_rl.eval.behavior_distance import METRICS, Behavior
from zero_shot_rl.eval.first_action import TASKS, inference_set, load_with_zs

STEPS = 1000


def rollout(agent, z, reward_fn, episodes, stride):
    """Visited states [episodes, steps // stride, d_s] and returns [episodes]."""
    envs = [make_env(e) for e in range(episodes)]
    obs = np.stack([env.reset().observation["observations"] for env in envs])
    z = z.expand(episodes, -1)
    returns = np.zeros(episodes)
    kept = []
    for t in range(STEPS):
        if t % stride == 0:
            kept.append(obs.copy())
        with torch.no_grad():
            actions = policy(agent, torch.as_tensor(obs, dtype=torch.float32, device=z.device), z)
        actions = actions.cpu().numpy()
        for i, env in enumerate(envs):
            obs[i] = env.step(actions[i]).observation["observations"]
            returns[i] += reward_fn(env.physics)
    return np.stack(kept, 1), returns


def plot_pairs(df, metric, pool_algorithm, path):
    """Scatter of FB seed k (x) against VC-FB seed k (y) at the states visited by pool seed k."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    colours = plt.cm.tab10(np.arange(3))
    for ax, task in zip(axes.ravel(), TASKS):
        deltas = []
        top = 0.0
        for k, colour in enumerate(colours):
            sub = df[(df["task"] == task) & (df["pool"] == f"{pool_algorithm} seed {k}")]
            fb = sub[sub["actor"] == f"FB seed {k}"].set_index("step")[metric]
            vc = sub[sub["actor"] == f"VC-FB seed {k}"].set_index("step")[metric].loc[fb.index]
            ax.scatter(fb, vc, s=14, color=colour, label=f"seed {k}")
            deltas.append((vc - fb).to_numpy())
            top = max(top, fb.max(), vc.max())
        deltas = np.concatenate(deltas)
        ax.plot([0, top], [0, top], color="gray", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel(f"FB {metric} distance")
        ax.set_ylabel(f"VC-FB {metric} distance")
        ax.text(
            0.03, 0.97,
            f"VC-FB closer: {(deltas < 0).mean():.0%}\n"
            f"median VC-FB minus FB: {np.median(deltas):+.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
        )
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"states visited by {pool_algorithm}", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", action="append", required=True,
                        help="LABEL path [path ...]; repeat per algorithm")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--figure_prefix", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--std_dev_eval", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer, observations_z, rewards_z = inference_set(args, device)
    behavior = Behavior(
        buffer.storage["observations"].float().to(device),
        buffer.storage["actions"].float().to(device),
        args.lam,
        args.k,
    )
    reward_fns = RewardFunctionConstructor(
        domain_name="walker", task_names=TASKS, seed=args.seed, device=device
    ).reward_functions

    agents = {}
    for label, *paths in args.checkpoints:
        for index, path in enumerate(paths):
            agent, zs = load_with_zs(path, observations_z, rewards_z, args, device)
            zs = {t: torch.as_tensor(z, dtype=torch.float32, device=device) for t, z in zs.items()}
            agents[f"{label} seed {index}"] = (agent, zs)

    rows = []
    for pool, (pool_agent, pool_zs) in agents.items():
        for task in TASKS:
            states, returns = rollout(pool_agent, pool_zs[task], reward_fns[task], args.episodes, args.stride)
            n_kept = states.shape[1]
            s = torch.as_tensor(states.reshape(-1, states.shape[-1]), dtype=torch.float32, device=device)
            state_term = behavior.state_term(s)
            for actor, (agent, zs) in agents.items():
                with torch.no_grad():
                    a = policy(agent, s, zs[task].expand(len(s), -1))
                joint, conditional = behavior.distances(state_term, a)
                for e in range(args.episodes):
                    for i in range(n_kept):
                        j = e * n_kept + i
                        rows.append({
                            "pool": pool, "actor": actor, "task": task, "episode": e,
                            "step": i * args.stride, "return": returns[e],
                            "joint": joint[j], "conditional": conditional[j],
                        })
            print(f"{pool} {task}: return {returns.mean():.0f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    own = df[df["pool"] == df["actor"]]
    print(own.groupby(["task", "actor"])[METRICS].mean().round(3))
    if args.figure_prefix:
        for metric in METRICS:
            for pool_algorithm in ("FB", "VC-FB"):
                plot_pairs(df, metric, pool_algorithm,
                           f"{args.figure_prefix}_{pool_algorithm}_states_{metric}.png")


if __name__ == "__main__":
    main()
