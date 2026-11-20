"""Render side-by-side comparison videos: left = original z_r, right = ANFIS-z.

Reads `z_r` and `z` stored in each `<models_dir>/<task>_best.pt` and rolls out the frozen FB
checkpoint from the same environment seed for both latents.

Usage (repo root):
    PYTHONPATH=.:tasks/T6-anfis MUJOCO_GL=egl uv run python -m analyze.render_compare \\
        --checkpoint artifacts/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt \\
        --models_dir artifacts/T6-anfis/zero-shot-rl-314/anfis_z_297/models \\
        --out_dir artifacts/T6-anfis/zero-shot-rl-314/videos
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from agents.base import load_agent
from rewards import RewardFunctionConstructor
from zero_shot_rl.env.dynamics import load_dynamics
from zero_shot_rl.env.walker import STEPS, make_env, run_episodes, write_video
from zero_shot_rl.latent.anfis import load_anfis

TASKS = ["stand", "walk", "run", "flip"]
HEADER_H = 32
DIVIDER_W = 4


def compose_frames(task, frames_left, rewards_left, frames_right, rewards_right, frame_stride):
    """Stack left (original z_r) and right (ANFIS-z) frames with live return headers."""
    cum_left = np.cumsum(rewards_left)
    cum_right = np.cumsum(rewards_right)
    h, w, _ = frames_left[0].shape
    total_w = w * 2 + DIVIDER_W
    total_h = h + HEADER_H
    # H.264 yuv420p requires even width and height
    total_w += total_w % 2
    total_h += total_h % 2

    out = []
    for k, (fl, fr) in enumerate(zip(frames_left, frames_right)):
        step = min(k * frame_stride, STEPS - 1)
        canvas = np.full((total_h, total_w, 3), 24, dtype=np.uint8)
        canvas[HEADER_H : HEADER_H + h, 0:w] = fl
        canvas[HEADER_H : HEADER_H + h, w + DIVIDER_W : w + DIVIDER_W + w] = fr
        canvas[HEADER_H : HEADER_H + h, w : w + DIVIDER_W] = 90

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.text((8, 10), f"{task} | Original z_r | R={cum_left[step]:6.1f}", fill=(170, 210, 255))
        draw.text((w + DIVIDER_W + 8, 10), f"{task} | ANFIS-z | R={cum_right[step]:6.1f}", fill=(255, 200, 130))
        draw.text((total_w - 86, 10), f"t={step:4d}", fill=(200, 200, 200))
        out.append(np.asarray(img))
    return out


def render_task(agent, task, z_orig, z_new, reward_fn, seed, dynamics, frame_stride, out_path):
    env_l = make_env(seed, dynamics)
    env_r = make_env(seed, dynamics)
    try:
        run_l = run_episodes(agent, z_orig, [env_l], reward_fn=reward_fn, frame_stride=frame_stride)
        run_r = run_episodes(agent, z_new, [env_r], reward_fn=reward_fn, frame_stride=frame_stride)
    finally:
        env_l.close()
        env_r.close()

    frames = compose_frames(
        task,
        run_l["frames"][0],
        run_l["rewards"][0],
        run_r["frames"][0],
        run_r["rewards"][0],
        frame_stride,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(frames, out_path)
    return float(run_l["return"][0]), float(run_r["return"][0])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--models_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--dynamics", type=str, default=None)
    parser.add_argument("--model_suffix", type=str, default="best", choices=("best", "final"))
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(4)
    agent = load_agent(args.checkpoint, device).eval()
    dynamics = load_dynamics(args.dynamics) if args.dynamics else None
    rc = RewardFunctionConstructor(domain_name="walker", task_names=list(args.tasks), seed=args.seed, device=device)

    for task in args.tasks:
        _, meta = load_anfis(args.models_dir / f"{task}_{args.model_suffix}.pt", device=device)
        z_orig = torch.as_tensor(meta["z_r"], dtype=torch.float32, device=device)
        z_new = torch.as_tensor(meta["z"], dtype=torch.float32, device=device)
        out_path = args.out_dir / f"{task}_compare.mp4"
        ret_l, ret_r = render_task(
            agent, task, z_orig, z_new, rc.reward_functions[task],
            args.seed, dynamics, args.frame_stride, out_path,
        )
        print(f"[{task:5s}] seed={args.seed}  Original={ret_l:6.1f}  ANFIS-z={ret_r:6.1f}  delta={ret_r - ret_l:+6.1f}  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
