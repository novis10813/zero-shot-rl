"""
Action difference heatmap of VC-FB and FB at states from the dataset.

Rows are --n_states states sampled from the training dataset subsample (the same states for
every panel), columns are the actuators, and a cell is

    delta_a(s_i) = pi_VC(s_i, z_VC) - pi_FB(s_i, z_FB)

with the deterministic (clipped mean) actions and each agent's own z for the task. One panel
per (task, seed pair): VC-FB seed k against FB seed k.

Writes a long CSV (task, pair, state index, actuator, FB action, VC-FB action, delta) and one
figure with a row of panels per task.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.action_difference \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out action_difference.csv --figure action_difference.png
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from zero_shot_rl.env.walker import policy
from zero_shot_rl.eval.first_action import TASKS, first_state, inference_set, load_with_zs


def actions_at(agent, zs, states, task, device):
    z = torch.as_tensor(zs[task], dtype=torch.float32, device=device)
    with torch.no_grad():
        return policy(agent, states, z.expand(len(states), -1)).cpu().numpy()


def plot(df, joints, path):
    pairs = sorted(df["pair"].unique())
    fig, axes = plt.subplots(len(TASKS), len(pairs), figsize=(3.4 * len(pairs), 3.6 * len(TASKS)),
                             squeeze=False)
    image = None
    for r, task in enumerate(TASKS):
        for c, pair in enumerate(pairs):
            sub = df[(df["task"] == task) & (df["pair"] == pair)]
            delta = sub.pivot(index="state", columns="actuator", values="delta")[joints]
            ax = axes[r, c]
            image = ax.imshow(delta.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2,
                              interpolation="nearest")
            ax.set_title(f"{task}, seed pair {pair}", fontsize=9)
            ax.set_xticks(range(len(joints)))
            ax.set_xticklabels(joints if r == len(TASKS) - 1 else [], rotation=60, ha="right",
                               fontsize=8)
            ax.set_ylabel("dataset state" if c == 0 else "")
    fig.colorbar(image, ax=axes, shrink=0.6, label="VC-FB action minus FB action")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", action="append", required=True,
                        help="LABEL path [path ...]; FB first, VC-FB second, paired by position")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--figure", type=str, default=None)
    parser.add_argument("--n_states", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--std_dev_eval", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer, observations_z, rewards_z = inference_set(args, device)
    joints = first_state(0)[1]

    storage = buffer.storage["observations"]
    index = np.random.default_rng(args.seed).choice(len(storage), args.n_states, replace=False)
    states = storage[torch.as_tensor(index, device=storage.device)].float().to(device)

    (_, *fb_paths), (_, *vc_paths) = args.checkpoints
    rows = []
    for pair, (fb_path, vc_path) in enumerate(zip(fb_paths, vc_paths)):
        fb, fb_zs = load_with_zs(fb_path, observations_z, rewards_z, args, device)
        vc, vc_zs = load_with_zs(vc_path, observations_z, rewards_z, args, device)
        for task in TASKS:
            a_fb = actions_at(fb, fb_zs, states, task, device)
            a_vc = actions_at(vc, vc_zs, states, task, device)
            for i in range(args.n_states):
                for j, joint in enumerate(joints):
                    rows.append({
                        "task": task, "pair": pair, "state": i, "dataset_index": int(index[i]),
                        "actuator": joint, "fb": a_fb[i, j], "vc": a_vc[i, j],
                        "delta": a_vc[i, j] - a_fb[i, j],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["task", "actuator"])["delta"].agg(
        mean="mean", mean_abs=lambda x: x.abs().mean()
    ).round(3)
    print(summary.unstack(0))
    if args.figure:
        plot(df, joints, args.figure)


if __name__ == "__main__":
    main()
