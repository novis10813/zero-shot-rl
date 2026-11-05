"""
How far the first-step actions of FB and VC-FB are from the behaviour in the dataset.

For env seeds 0..N-1 the walker is reset and every checkpoint picks its deterministic action
at that state for each task (z inferred as in zero_shot_rl.eval.first_action). Each action is
scored against the (s_i, a_i) of the dataset used for training:

    joint        min_i [ |s - s_i|^2 / (d_s) + lam |a - a_i|^2 / (d_a) ]
    conditional  min over the k dataset states nearest to s of |a - a_i|^2 / d_a

Every coordinate is standardised by its dataset standard deviation and each squared norm is
divided by the dimension count, so one unit is one standard deviation per coordinate.
Reference actions at the same states: uniform random actions in [-1, 1], and the action of
the nearest dataset state ("nn_action").

Writes a CSV with one row per (algorithm, checkpoint, task, env seed) and one figure per
metric: a panel per task with FB on x and VC-FB on y, one dot per state, FB seed k paired with
VC-FB seed k. Below the diagonal means the VC-FB action is closer to the data.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.behavior_distance \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out distances.csv --figure_prefix distance
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from zero_shot_rl.eval.first_action import (
    TASKS,
    act,
    first_state,
    inference_set,
    load_with_zs,
)

METRICS = ["joint", "conditional"]


class Behavior:
    """Dataset (s_i, a_i) with the standardisation and the two distances."""

    def __init__(self, states, actions, lam, k):
        self.states, self.actions, self.lam, self.k = states, actions, lam, k
        self.sigma_s = states.std(0).clamp(min=1e-6)
        self.sigma_a = actions.std(0).clamp(min=1e-6)

    def state_term(self, s):
        """[Q, N]: mean squared standardised state difference to every dataset state."""
        d = torch.cdist(s / self.sigma_s, self.states / self.sigma_s) ** 2
        return d / s.shape[1]

    def action_term(self, a):
        d = torch.cdist(a / self.sigma_a, self.actions / self.sigma_a) ** 2
        return d / a.shape[1]

    def distances(self, state_term, a):
        """joint and conditional distance of actions `a` [Q, d_a] at the states of `state_term`."""
        action_term = self.action_term(a)
        joint = (state_term + self.lam * action_term).min(1).values
        nearest = state_term.topk(self.k, dim=1, largest=False).indices
        conditional = action_term.gather(1, nearest).min(1).values
        return joint.cpu().numpy(), conditional.cpu().numpy()


def rows_for(label, checkpoint, task, seeds, joint, conditional):
    return [
        {
            "algorithm": label,
            "checkpoint": checkpoint,
            "task": task,
            "env_seed": seed,
            "joint": j,
            "conditional": c,
        }
        for seed, j, c in zip(seeds, joint, conditional)
    ]


def plot(df, metric, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    checkpoints = sorted(df[df["algorithm"] == "FB"]["checkpoint"].unique())
    colours = plt.cm.tab10(np.arange(len(checkpoints)))
    for ax, task in zip(axes.ravel(), TASKS):
        sub = df[df["task"] == task]
        deltas = []
        top = 0.0
        for colour, k in zip(colours, checkpoints):
            fb = sub[(sub["algorithm"] == "FB") & (sub["checkpoint"] == k)].set_index("env_seed")[metric]
            vc = sub[(sub["algorithm"] == "VC-FB") & (sub["checkpoint"] == k)].set_index("env_seed")[metric]
            ax.scatter(fb, vc.loc[fb.index], s=14, color=colour, label=f"seed {k}")
            deltas.append((vc.loc[fb.index] - fb).to_numpy())
            top = max(top, fb.max(), vc.max())
        deltas = np.concatenate(deltas)
        ax.plot([0, top], [0, top], color="gray", linewidth=0.8)
        uniform = sub[sub["algorithm"] == "uniform"][metric].median()
        nn_action = sub[sub["algorithm"] == "nn_action"][metric].median()
        ax.set_title(task)
        ax.set_xlabel(f"FB {metric} distance")
        ax.set_ylabel(f"VC-FB {metric} distance")
        ax.text(
            0.03, 0.97,
            f"VC-FB closer: {(deltas < 0).mean():.0%}\n"
            f"median VC-FB minus FB: {np.median(deltas):+.3f}\n"
            f"median uniform: {uniform:.2f}, nn_action: {nn_action:.2f}",
            transform=ax.transAxes, va="top", fontsize=8,
        )
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", action="append", required=True,
                        help="LABEL path [path ...]; FB first, VC-FB second, paired by position")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--figure_prefix", type=str, default=None)
    parser.add_argument("--n_states", type=int, default=50)
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

    env_seeds = list(range(args.n_states))
    obs = np.stack([first_state(seed)[0] for seed in env_seeds])
    s = torch.as_tensor(obs, dtype=torch.float32, device=device)
    state_term = behavior.state_term(s)

    rows = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    uniform = torch.rand(s.shape[0], behavior.actions.shape[1], device=device, generator=generator) * 2 - 1
    nn_action = behavior.actions[state_term.argmin(1)]
    for task in TASKS:
        for label, a in [("uniform", uniform), ("nn_action", nn_action)]:
            rows += rows_for(label, 0, task, env_seeds, *behavior.distances(state_term, a))

    for label, *paths in args.checkpoints:
        for index, path in enumerate(paths):
            agent, zs = load_with_zs(path, observations_z, rewards_z, args, device)
            for task in TASKS:
                a = torch.as_tensor(
                    np.stack([act(agent, o, zs[task]) for o in obs]), dtype=torch.float32, device=device
                )
                rows += rows_for(label, index, task, env_seeds, *behavior.distances(state_term, a))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby(["task", "algorithm"])[METRICS].median().round(3))
    if args.figure_prefix:
        for metric in METRICS:
            plot(df, metric, f"{args.figure_prefix}_{metric}.png")


if __name__ == "__main__":
    main()
