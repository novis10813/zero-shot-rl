"""
Distance of FB and VC-FB actions to the dataset behaviour along whole episodes.

Every checkpoint runs --episodes deterministic episodes of 1000 steps per task (episode e
starts from the reset of an environment seeded with e, so start states are shared; z inferred
as in zero_shot_rl.latent.task_z). The states s_t visited every --stride steps are kept. Two
readings, both scored with the distances of zero_shot_rl.latent.behavior:

    own    D(s_t, a_t) of the checkpoint's own visited states and actions.
    cross  every checkpoint acts at the states visited by every other checkpoint, so FB and
           VC-FB are compared at identical states (rows with the same pool).

Writes one CSV row per (pool checkpoint, acting checkpoint, task, episode, step): the distances
of the acting checkpoint's action at that visited state, plus the episode return of the pool
checkpoint.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.behavior_rollout \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out rollout_distances.csv
"""

from argparse import ArgumentParser

import pandas as pd
import torch

from zero_shot_rl.env.walker import make_env, policy, run_episodes
from zero_shot_rl.latent import behavior as behavior_module
from zero_shot_rl.latent import task_z
from zero_shot_rl.latent.behavior import METRICS, Behavior
from zero_shot_rl.latent.task_z import TASKS


def main():
    parser = ArgumentParser()
    task_z.add_arguments(parser)
    behavior_module.add_arguments(parser)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    device = task_z.device()
    buffer, observations_z, rewards_z = task_z.inference_set(args, device)
    behavior = Behavior.from_buffer(buffer, args.lam, args.k, device)
    reward_fns = task_z.reward_functions(args, device)
    agents = {
        f"{label} seed {index}": (agent, zs)
        for label, index, agent, zs in task_z.load_checkpoints(args, observations_z, rewards_z, device)
    }

    rows = []
    for pool, (pool_agent, pool_zs) in agents.items():
        for task in TASKS:
            envs = [make_env(e) for e in range(args.episodes)]
            run = run_episodes(pool_agent, pool_zs[task], envs, reward_fns[task], args.stride)
            states, returns = run["observations"], run["return"]
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


if __name__ == "__main__":
    main()
