"""
Record one deterministic 1000-step episode per checkpoint and task from env seed --env_seed.

Per (algorithm, seed index, task) it writes
    <out_dir>/actions.npz   key "<algorithm>/seed<k>/<task>": actions [1000, 6] and
                            "<...>/reward": task reward per step [1000]
    <video_dir>/<algorithm>_seed<k>_<task>.mp4   every --frame_stride-th step, 30 fps
z is inferred as in zero_shot_rl.latent.task_z. Videos are not meant for git; put
--video_dir under artifacts/. Rendering needs a single process.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.record_episode \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out episode_actions.npz --video_dir <dir>
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from zero_shot_rl.env.walker import make_env, run_episodes, write_video
from zero_shot_rl.latent import task_z
from zero_shot_rl.latent.task_z import TASKS


def main():
    parser = ArgumentParser()
    task_z.add_arguments(parser)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=2)
    args = parser.parse_args()

    device = task_z.device()
    _, observations_z, rewards_z = task_z.inference_set(args, device)
    reward_fns = task_z.reward_functions(args, device)
    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    arrays = {}
    for label, index, agent, zs in task_z.load_checkpoints(args, observations_z, rewards_z, device):
        for task in TASKS:
            env = make_env(args.env_seed)
            try:
                run = run_episodes(agent, zs[task], [env], reward_fns[task],
                                   frame_stride=args.frame_stride if args.video_dir else None)
            finally:
                env.close()
            key = f"{label}/seed{index}/{task}"
            arrays[key] = run["actions"][0]
            arrays[f"{key}/reward"] = run["rewards"][0]
            if args.video_dir:
                write_video(run["frames"][0], Path(args.video_dir) / f"{label}_seed{index}_{task}.mp4")
            print(f"{key}: return {run['return'][0]:.0f}", flush=True)
    np.savez(args.out, **arrays)


if __name__ == "__main__":
    main()
