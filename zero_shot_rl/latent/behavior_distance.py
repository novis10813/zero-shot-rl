"""
How far the first-step actions of the checkpoints are from the behaviour in the dataset.

For env seeds 0..N-1 the walker is reset and every checkpoint picks its deterministic action
at that state for each task (z inferred as in zero_shot_rl.latent.task_z). Each action is
scored with the joint and conditional distances of zero_shot_rl.latent.behavior against the
dataset used for training. Reference actions at the same states: uniform random actions in
[-1, 1] ("uniform"), and the action of the nearest dataset state ("nn_action"), both with
checkpoint index 0.

Writes a CSV with one row per (algorithm, checkpoint, task, env seed).

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.latent.behavior_distance \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out distances.csv
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

from zero_shot_rl.env.walker import act, first_state
from zero_shot_rl.latent import behavior as behavior_module
from zero_shot_rl.latent import task_z
from zero_shot_rl.latent.behavior import METRICS, Behavior
from zero_shot_rl.latent.task_z import TASKS


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


def main():
    parser = ArgumentParser()
    task_z.add_arguments(parser)
    behavior_module.add_arguments(parser)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_states", type=int, default=50)
    args = parser.parse_args()

    device = task_z.device()
    buffer, observations_z, rewards_z = task_z.inference_set(args, device)
    behavior = Behavior.from_buffer(buffer, args.lam, args.k, device)

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

    for label, index, agent, zs in task_z.load_checkpoints(args, observations_z, rewards_z, device):
        for task in TASKS:
            a = torch.as_tensor(
                np.stack([act(agent, o, zs[task]) for o in obs]), dtype=torch.float32, device=device
            )
            rows += rows_for(label, index, task, env_seeds, *behavior.distances(state_term, a))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby(["task", "algorithm"])[METRICS].median().round(3))


if __name__ == "__main__":
    main()
