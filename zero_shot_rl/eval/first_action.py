"""
Action of each checkpoint at the first state of an episode, per task and joint.

Every checkpoint sees the same state: the reset of a walker environment with the same seed.
Per task, z is inferred from 10k reward-labelled transitions of the same 100k-transition
dataset subsample (same seed for every checkpoint), then the deterministic actor action
at that state is recorded, exactly as in OfflineRLWorkspace.eval.

Writes a CSV (algorithm, checkpoint index, task, joint, action) and a bar figure: one panel
per task, one bar per checkpoint at each joint (checkpoints of one algorithm share a colour family).

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.first_action \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out first_action.csv --figure first_action.png
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agents.base import load_agent
from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor
from zero_shot_rl.env.walker import make_env
from zero_shot_rl.utils import set_seed_everywhere

TASKS = ["stand", "walk", "run", "flip"]


def first_state(env_seed):
    env = make_env(env_seed)
    obs = env.reset().observation["observations"]
    joints = list(env.physics.named.data.ctrl.axes.row.names)
    return obs, joints


def inference_set(args, device):
    """The dataset buffer and the task inference transitions, identical for every checkpoint."""
    set_seed_everywhere(args.seed)
    reward_constructor = RewardFunctionConstructor(
        domain_name="walker", task_names=TASKS, seed=args.seed, device=device
    )
    buffer = FBReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=Path(args.dataset_path),
        transitions=args.dataset_transitions,
        relabel=False,
        task=None,
        device=device,
        discount=args.discount,
        action_condition=None,
    )
    observations_z, rewards_z = buffer.sample_task_inference_transitions(
        inference_steps=args.z_inference_steps
    )
    return buffer, observations_z, rewards_z


def load_with_zs(checkpoint, observations_z, rewards_z, args, device):
    """The agent and its z per task."""
    agent = load_agent(Path(checkpoint), device)
    agent.eval()
    agent.std_dev_schedule = args.std_dev_eval
    return agent, {t: agent.infer_z(observations_z, rewards_z[t]) for t in TASKS}


def act(agent, obs, z):
    action, _ = agent.act(obs, task=z, step=None, sample=False)
    return np.clip(action, -1.0, 1.0)


def first_actions(checkpoint, obs, args, device):
    """Action at `obs` of one checkpoint for every task: dict task -> [action_length]."""
    _, observations_z, rewards_z = inference_set(args, device)
    agent, zs = load_with_zs(checkpoint, observations_z, rewards_z, args, device)
    return {task: act(agent, obs, zs[task]) for task in TASKS}


def plot(df, joints, path):
    """One bar per (algorithm, checkpoint) at each joint; one colour family per algorithm."""
    algorithms = list(dict.fromkeys(df["algorithm"]))
    groups = [(a, c) for a in algorithms for c in sorted(df[df["algorithm"] == a]["checkpoint"].unique())]
    cmaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples]
    colours = {}
    for k, algorithm in enumerate(algorithms):
        members = [g for g in groups if g[0] == algorithm]
        shades = np.linspace(0.45, 0.9, len(members))
        colours.update({g: cmaps[k % len(cmaps)](v) for g, v in zip(members, shades)})
    width = 0.9 / len(groups)
    x = np.arange(len(joints))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.ravel()
    for ax, task in zip(axes, TASKS):
        for k, (algorithm, checkpoint) in enumerate(groups):
            sub = df[(df["task"] == task) & (df["algorithm"] == algorithm) & (df["checkpoint"] == checkpoint)]
            values = [sub[sub["joint"] == j]["action"].item() for j in joints]
            offset = x + (k - (len(groups) - 1) / 2) * width
            ax.bar(offset, values, width, color=colours[(algorithm, checkpoint)],
                   label=f"{algorithm} seed {checkpoint}")
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels(joints, rotation=45, ha="right")
        ax.axhline(0, color="gray", linewidth=0.5)
    for ax in axes[::2]:
        ax.set_ylabel("action")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        action="append",
        required=True,
        help="LABEL path [path ...]; repeat per algorithm",
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--figure", type=str, default=None)
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--std_dev_eval", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs, joints = first_state(args.env_seed)

    rows = []
    for label, *paths in args.checkpoints:
        for index, path in enumerate(paths):
            for task, action in first_actions(path, obs, args, device).items():
                for joint, value in zip(joints, action):
                    rows.append(
                        {
                            "algorithm": label,
                            "checkpoint": index,
                            "task": task,
                            "joint": joint,
                            "action": float(value),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    if args.figure:
        plot(df, joints, args.figure)
    print(df.groupby(["task", "joint", "algorithm"])["action"].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
