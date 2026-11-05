"""
Action of each checkpoint at the first state of an episode, per task and joint.

Every checkpoint sees the same state: the reset of a walker environment with the same seed.
z per task is inferred as in zero_shot_rl.latent.task_z, then the deterministic actor action
at that state is recorded.

Writes a CSV (algorithm, checkpoint index, task, joint, action).

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.latent.first_action \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out first_action.csv
"""

from argparse import ArgumentParser

import pandas as pd

from zero_shot_rl.env.walker import act, first_state
from zero_shot_rl.latent import task_z
from zero_shot_rl.latent.task_z import TASKS


def main():
    parser = ArgumentParser()
    task_z.add_arguments(parser)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--env_seed", type=int, default=0)
    args = parser.parse_args()

    device = task_z.device()
    obs, joints = first_state(args.env_seed)
    _, observations_z, rewards_z = task_z.inference_set(args, device)

    rows = []
    for label, index, agent, zs in task_z.load_checkpoints(args, observations_z, rewards_z, device):
        for task in TASKS:
            for joint, value in zip(joints, act(agent, obs, zs[task])):
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
    print(df.groupby(["task", "joint", "algorithm"])["action"].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
