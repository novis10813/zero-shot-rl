"""
Episode return of every checkpoint from each initial state of env seeds 0..N-1.

Env seed e resets to the same state as in zero_shot_rl.latent.behavior_distance, so a return
here belongs to the same (task, env seed, checkpoint) as a distance there. Each checkpoint runs
one deterministic 1000-step episode per env seed and task, z inferred as in
zero_shot_rl.latent.task_z.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.episode_returns \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out returns.csv
"""

from argparse import ArgumentParser

import pandas as pd

from zero_shot_rl.env.walker import make_env, run_episodes
from zero_shot_rl.latent import task_z
from zero_shot_rl.latent.task_z import TASKS


def main():
    parser = ArgumentParser()
    task_z.add_arguments(parser)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_states", type=int, default=50)
    args = parser.parse_args()

    device = task_z.device()
    _, observations_z, rewards_z = task_z.inference_set(args, device)
    reward_fns = task_z.reward_functions(args, device)

    rows = []
    for label, index, agent, zs in task_z.load_checkpoints(args, observations_z, rewards_z, device):
        for task in TASKS:
            envs = [make_env(e) for e in range(args.n_states)]
            returns = run_episodes(agent, zs[task], envs, reward_fns[task])["return"]
            rows += [
                {"algorithm": label, "checkpoint": index, "task": task,
                 "env_seed": e, "return": r}
                for e, r in enumerate(returns)
            ]
            print(f"{label} seed {index} {task}: mean return {returns.mean():.0f}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
