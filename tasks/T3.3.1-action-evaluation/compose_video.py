"""
Side-by-side video of one task: VC-FB on the left, FB on the right.

Each column, top to bottom: the recorded episode video, the 6 actuator actions as stacked
step traces (ECG style), and the reward of each step. Inputs are the outputs of
zero_shot_rl.eval.record_episode: the per-step actions and rewards in episode_actions.npz and
the per-checkpoint mp4 files (one frame every --frame_stride steps). Video frame i is the state
after the action of step frame_stride * i, so the traces end at that step. The traces show the
last --window steps (--window 1000 shows the whole episode).

Usage (repo root):
    uv run python tasks/T3.3.1-action-evaluation/compose_video.py \
        --video_dir artifacts/T3.3.1-action-evaluation/videos --task walk --seed 1 \
        --window 100 --out artifacts/T3.3.1-action-evaluation/videos/walk_seed1_w100.mp4
"""

import subprocess
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
JOINTS = ["right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]
ALGORITHMS = ["VC-FB", "FB"]  # left column, right column
OFFSET = 2.5  # vertical distance between stacked traces
STEPS = 1000


def read_video(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    width, height = map(int, probe.stdout.strip().split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def main():
    parser = ArgumentParser()
    parser.add_argument("--actions", type=str, default=str(HERE / "data" / "episode_actions.npz"))
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.actions)
    runs = {}
    for algorithm in ALGORITHMS:
        key = f"{algorithm}/seed{args.seed}/{args.task}"
        frames = read_video(Path(args.video_dir) / f"{algorithm}_seed{args.seed}_{args.task}.mp4")
        runs[algorithm] = (frames, data[key], data[f"{key}/reward"])
    n_frames = min(len(frames) for frames, _, _ in runs.values())

    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    grid = fig.add_gridspec(3, 2, height_ratios=[3.4, 2.4, 1.0], hspace=0.25, wspace=0.08,
                            left=0.1, right=0.98, top=0.95, bottom=0.06)
    steps = np.arange(STEPS)
    colours = plt.cm.tab10(np.arange(len(JOINTS)))
    artists = {}
    for c, algorithm in enumerate(ALGORITHMS):
        frames, actions, rewards = runs[algorithm]
        video_ax = fig.add_subplot(grid[0, c])
        image = video_ax.imshow(frames[0])
        video_ax.axis("off")
        title = video_ax.set_title("", fontsize=10)

        action_ax = fig.add_subplot(grid[1, c])
        traces = []
        for j, colour in enumerate(colours):
            base = (len(JOINTS) - 1 - j) * OFFSET
            action_ax.axhline(base, color="lightgray", linewidth=0.5)
            (line,) = action_ax.plot([], [], drawstyle="steps-post", color=colour, linewidth=0.9)
            traces.append((line, actions[:, j] + base))
        (heads,) = action_ax.plot([], [], "o", color="black", markersize=3)
        action_ax.set_ylim(-1.4, (len(JOINTS) - 1) * OFFSET + 1.4)
        action_ax.set_yticks([(len(JOINTS) - 1 - j) * OFFSET for j in range(len(JOINTS))])
        action_ax.set_yticklabels(JOINTS if c == 0 else [], fontsize=8)
        action_ax.tick_params(axis="x", labelbottom=False)

        reward_ax = fig.add_subplot(grid[2, c])
        (reward_line,) = reward_ax.plot([], [], color="black", linewidth=0.9)
        reward_ax.set_ylim(-0.05, 1.05)
        reward_ax.set_yticks([0, 1])
        reward_ax.set_ylabel("reward" if c == 0 else "", fontsize=8)
        if c != 0:
            reward_ax.set_yticklabels([])
        reward_ax.set_xlabel("step", fontsize=8)
        reward_ax.tick_params(labelsize=8)
        artists[algorithm] = (image, title, action_ax, traces, heads, reward_ax, reward_line)

    width, height = fig.canvas.get_width_height()
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p",
         str(args.out)],
        stdin=subprocess.PIPE,
    )
    for i in range(n_frames):
        t = min(args.frame_stride * i, STEPS - 1)
        start = max(0, t - args.window + 1)
        right = max(t, args.window - 1) + 1
        for algorithm in ALGORITHMS:
            frames, actions, rewards = runs[algorithm]
            image, title, action_ax, traces, heads, reward_ax, reward_line = artists[algorithm]
            image.set_data(frames[i])
            title.set_text(f"{algorithm} seed {args.seed}, {args.task}: step {t}, "
                           f"return {rewards[: t + 1].sum():.0f}")
            for line, values in traces:
                line.set_data(steps[start: t + 1], values[start: t + 1])
            heads.set_data([t] * len(traces), [values[t] for _, values in traces])
            reward_line.set_data(steps[start: t + 1], rewards[start: t + 1])
            action_ax.set_xlim(right - args.window, right)
            reward_ax.set_xlim(right - args.window, right)
        fig.canvas.draw()
        encoder.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
    encoder.stdin.close()
    encoder.wait()


if __name__ == "__main__":
    main()
