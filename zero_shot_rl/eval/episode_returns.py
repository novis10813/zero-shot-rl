"""
Episode return of every checkpoint from each initial state of env seeds 0..N-1.

Env seed e resets to the same state as in zero_shot_rl.eval.behavior_distance, so a return
here belongs to the same (task, env seed, checkpoint) as a distance there. Each checkpoint runs
one deterministic 1000-step episode per env seed and task, z inferred as in
zero_shot_rl.eval.first_action.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.episode_returns \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out returns.csv
"""

from argparse import ArgumentParser

import pandas as pd
import torch

from rewards import RewardFunctionConstructor
from zero_shot_rl.eval.behavior_rollout import STEPS, rollout
from zero_shot_rl.eval.first_action import TASKS, inference_set, load_with_zs


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", action="append", required=True,
                        help="LABEL path [path ...]; repeat per algorithm")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--std_dev_eval", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, observations_z, rewards_z = inference_set(args, device)
    reward_fns = RewardFunctionConstructor(
        domain_name="walker", task_names=TASKS, seed=args.seed, device=device
    ).reward_functions

    rows = []
    for label, *paths in args.checkpoints:
        for index, path in enumerate(paths):
            agent, zs = load_with_zs(path, observations_z, rewards_z, args, device)
            for task in TASKS:
                z = torch.as_tensor(zs[task], dtype=torch.float32, device=device)
                # stride = STEPS keeps only the start state; the states are not used here
                _, returns = rollout(agent, z, reward_fns[task], args.n_states, STEPS)
                rows += [
                    {"algorithm": label, "checkpoint": index, "task": task,
                     "env_seed": e, "return": r}
                    for e, r in enumerate(returns)
                ]
                print(f"{label} seed {index} {task}: mean return {returns.mean():.0f}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
