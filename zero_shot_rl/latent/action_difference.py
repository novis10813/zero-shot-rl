"""
Action difference of VC-FB and FB at states from the dataset.

At --n_states states sampled from the training dataset subsample (the same states for every
pair), per actuator

    delta_a(s_i) = pi_VC(s_i, z_VC) - pi_FB(s_i, z_FB)

with the deterministic (clipped mean) actions and each agent's own z for the task (inferred
as in zero_shot_rl.latent.task_z). --checkpoints FB first and VC-FB second; FB seed k is
paired with VC-FB seed k.

Writes a long CSV (task, pair, state index, actuator, FB action, VC-FB action, delta).

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.latent.action_difference \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out action_difference.csv
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

from zero_shot_rl.env.walker import first_state, policy
from zero_shot_rl.latent import task_z
from zero_shot_rl.latent.task_z import TASKS


def actions_at(agent, z, states):
    with torch.no_grad():
        return policy(agent, states, z.expand(len(states), -1)).cpu().numpy()


def main():
    parser = ArgumentParser()
    task_z.add_arguments(parser)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_states", type=int, default=200)
    args = parser.parse_args()

    device = task_z.device()
    buffer, observations_z, rewards_z = task_z.inference_set(args, device)
    joints = first_state(0)[1]

    storage = buffer.storage["observations"]
    index = np.random.default_rng(args.seed).choice(len(storage), args.n_states, replace=False)
    states = storage[torch.as_tensor(index, device=storage.device)].float().to(device)

    checkpoints = task_z.load_checkpoints(args, observations_z, rewards_z, device)
    (fb_label, *_), (vc_label, *_) = args.checkpoints
    fbs = [c for c in checkpoints if c[0] == fb_label]
    vcs = [c for c in checkpoints if c[0] == vc_label]
    rows = []
    for pair, ((_, _, fb, fb_zs), (_, _, vc, vc_zs)) in enumerate(zip(fbs, vcs)):
        for task in TASKS:
            a_fb = actions_at(fb, fb_zs[task], states)
            a_vc = actions_at(vc, vc_zs[task], states)
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


if __name__ == "__main__":
    main()
