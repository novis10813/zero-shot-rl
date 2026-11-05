"""
Reward of the action each checkpoint takes at the states of an episode, against the
distance of that action to the dataset behaviour.

For env seed --env_seed, every checkpoint runs one deterministic 1000-step episode per task
and keeps the state (observation and physics) every --stride steps (the "pool" of states).
At each kept state every checkpoint (the "actor") acts with its own policy, scored with the
joint and conditional distances of zero_shot_rl.eval.behavior_distance. The same action is then
executed from that exact physics state in the simulator and the task reward is summed over
--horizon steps (the first step uses the action above, later steps the actor's own policy).
--horizon 1 is the reward r(s') after taking a at s.

Writes one CSV row per (pool, actor, task, step) and, per pool algorithm and distance, a figure
with one dot per state: x = distance of VC-FB seed k minus FB seed k, y = reward of VC-FB
seed k minus FB seed k, both acting at the states of the pool algorithm's seed k.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.behavior_reward \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out rewards.csv --figure_prefix reward
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from rewards import RewardFunctionConstructor
from zero_shot_rl.env.walker import make_env, policy, set_physics
from zero_shot_rl.eval.behavior_distance import METRICS, Behavior
from zero_shot_rl.eval.first_action import TASKS, inference_set, load_with_zs

STEPS = 1000


def act(agent, obs, z, device):
    with torch.no_grad():
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return policy(agent, o, z.unsqueeze(0))[0].cpu().numpy()


def collect_states(agent, z, env, stride, device):
    """Observations [n, d_s] and physics states [n, d_p] every `stride` steps of an episode."""
    obs = env.reset().observation["observations"]
    observations, physics = [], []
    for t in range(STEPS):
        if t % stride == 0:
            observations.append(obs.copy())
            physics.append(env.physics.get_state().copy())
        obs = env.step(act(agent, obs, z, device)).observation["observations"]
    return np.stack(observations), np.stack(physics)


def reward_from(env, physics, obs, agent, z, reward_fn, horizon, device):
    """Reward summed over `horizon` steps starting by acting at `obs` from `physics`."""
    set_physics(env, physics)
    total = 0.0
    for _ in range(horizon):
        obs = env.step(act(agent, obs, z, device)).observation["observations"]
        total += reward_fn(env.physics)
    return total


def plot(df, metric, pool_algorithm, path, horizon):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    colours = plt.cm.tab10(np.arange(3))
    for ax, task in zip(axes.ravel(), TASKS):
        xs, ys = [], []
        for k, colour in enumerate(colours):
            sub = df[(df["task"] == task) & (df["pool"] == f"{pool_algorithm} seed {k}")]
            fb = sub[sub["actor"] == f"FB seed {k}"].set_index("step")
            vc = sub[sub["actor"] == f"VC-FB seed {k}"].set_index("step").loc[fb.index]
            x, y = (vc[metric] - fb[metric]).to_numpy(), (vc["reward"] - fb["reward"]).to_numpy()
            ax.scatter(x, y, s=14, color=colour, label=f"seed {k}")
            xs.append(x)
            ys.append(y)
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        rho, p = stats.spearmanr(xs, ys)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel(f"{metric} distance: VC-FB minus FB")
        ax.set_ylabel(f"reward over {horizon} step(s): VC-FB minus FB")
        ax.text(0.03, 0.97, f"Spearman {rho:+.2f} (p={p:.2f}), n={len(xs)}",
                transform=ax.transAxes, va="top", fontsize=8)
        print(f"{pool_algorithm} states {metric:11s} {task:5s} rho {rho:+.2f} p {p:.3f} n {len(xs)}")
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
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1)
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

    env = make_env(args.env_seed)
    rows = []
    for pool, (pool_agent, pool_zs) in agents.items():
        for task in TASKS:
            observations, physics = collect_states(pool_agent, pool_zs[task], env, args.stride, device)
            s = torch.as_tensor(observations, dtype=torch.float32, device=device)
            state_term = behavior.state_term(s)
            for actor, (agent, zs) in agents.items():
                z = zs[task]
                with torch.no_grad():
                    a = policy(agent, s, z.expand(len(s), -1))
                joint, conditional = behavior.distances(state_term, a)
                for i in range(len(s)):
                    rows.append({
                        "pool": pool, "actor": actor, "task": task, "step": i * args.stride,
                        "reward": reward_from(env, physics[i], observations[i], agent, z,
                                              reward_fns[task], args.horizon, device),
                        "joint": joint[i], "conditional": conditional[i],
                    })
            print(f"{pool} {task} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    own = df[df["pool"] == df["actor"]]
    print(own.groupby(["task", "actor"])[METRICS + ["reward"]].mean().round(3))
    if args.figure_prefix:
        for metric in METRICS:
            for pool_algorithm in ("FB", "VC-FB"):
                plot(df, metric, pool_algorithm,
                     f"{args.figure_prefix}_{pool_algorithm}_states_{metric}.png", args.horizon)


if __name__ == "__main__":
    main()
