"""
Record one deterministic 1000-step episode per checkpoint and task from env seed --env_seed.

Per (algorithm, seed index, task) it writes
    <out_dir>/actions.npz   key "<algorithm>/seed<k>/<task>": actions [1000, 6] and
                            "<...>/reward": task reward per step [1000]
    <video_dir>/<algorithm>_seed<k>_<task>.mp4   every --frame_stride-th step, 30 fps
z is inferred as in zero_shot_rl.eval.first_action. Videos are not meant for git; put
--video_dir under artifacts/. Rendering needs a single process.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.record_episode \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out episode_actions.npz --video_dir <dir>
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from rewards import RewardFunctionConstructor
from zero_shot_rl.env.walker import make_env, policy, write_video
from zero_shot_rl.eval.first_action import TASKS, inference_set, load_with_zs

STEPS = 1000


def record(agent, z, reward_fn, env_seed, frame_stride, device):
    env = make_env(env_seed)
    z = torch.as_tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
    actions, rewards, frames = [], [], []
    try:
        obs = env.reset().observation["observations"]
        for t in range(STEPS):
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a = policy(agent, o, z)[0].cpu().numpy()
            obs = env.step(a).observation["observations"]
            actions.append(a)
            rewards.append(reward_fn(env.physics))
            if t % frame_stride == 0:
                frames.append(env.physics.render(height=240, width=320, camera_id=0))
    finally:
        env.close()
    return np.stack(actions), np.array(rewards), frames


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", action="append", required=True,
                        help="LABEL path [path ...]; repeat per algorithm")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=2)
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
    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    arrays = {}
    for label, *paths in args.checkpoints:
        for index, path in enumerate(paths):
            agent, zs = load_with_zs(path, observations_z, rewards_z, args, device)
            for task in TASKS:
                actions, rewards, frames = record(
                    agent, zs[task], reward_fns[task], args.env_seed, args.frame_stride, device
                )
                key = f"{label}/seed{index}/{task}"
                arrays[key] = actions
                arrays[f"{key}/reward"] = rewards
                if args.video_dir:
                    write_video(frames, Path(args.video_dir) / f"{label}_seed{index}_{task}.mp4")
                print(f"{key}: return {rewards.sum():.0f}", flush=True)
    np.savez(args.out, **arrays)


if __name__ == "__main__":
    main()
