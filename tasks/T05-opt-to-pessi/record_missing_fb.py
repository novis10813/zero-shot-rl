"""Record missing standard-FB videos only. Never train, infer z, or rewrite CSVs.

Run with PYTHONPATH=. MUJOCO_GL=egl from the repository root.
Videos replay deterministic, single-observation actions from environment seed 0.
This is not guaranteed to reproduce historical, batched evaluation trajectories.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from analyze_q1_fb import METHODS, ROOT, SOURCES, validate
from rollout_eval import make_env, policy
from agents.base import load_agent

REPO = ROOT.parents[1]
# Checkpoint hashes recorded in exp/T05-zero-shot-rl-{30,31,32}.
CHECKPOINTS = {
    0: ("artifacts/T04-mcfb-pessimism/zsrl-32/800000.pt",
        "2797c6587ded0677d1f57575f516683613b7734b05f40ac8b2ffb330ec52e81a"),
    1: ("artifacts/T04-mcfb-pessimism/zsrl-41/680000.pt",
        "7f6e641cc44244787a599c0b2de4b14e6310eaf0a36cda9c50162bf9c7cc91be"),
    2: ("artifacts/T04-mcfb-pessimism/zsrl-42/200000.pt",
        "229f7dae43ef6a52c288d2724aa303d7b3a4564eb9d557e6799a6e1a683b6e08"),
}
FAMILIES = ("z_r", *METHODS)


def video_name(data, row):
    return f"{data['task'][row]}_{data['family'][row]}_{data['target_angle'][row]:g}_{row}.mp4"


def selected_rows(data):
    return [row for row, family in enumerate(data["family"]) if family in FAMILIES]


def verify_video(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,nb_read_frames,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    if (stream["codec_name"], stream["width"], stream["height"],
        stream["nb_read_frames"], stream["r_frame_rate"]) != ("h264", 320, 240, "500", "30/1"):
        raise ValueError(f"Unexpected video format: {path}: {stream}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(agent, z_np, path):
    """Same reset, camera, action rule and frame stride as record_video; atomic output."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    z = torch.as_tensor(z_np, dtype=torch.float32).unsqueeze(0)
    env = make_env(0)
    frames = []
    try:
        obs = env.reset().observation["observations"]
        for step in range(1000):
            with torch.no_grad():
                action = policy(agent, torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), z)
            obs = env.step(action.numpy()[0]).observation["observations"]
            if step % 2 == 0:
                frames.append(env.physics.render(height=240, width=320, camera_id=0))
    finally:
        env.close()
    temporary = path.with_suffix(".partial.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "320x240", "-r", "30", "-i", "-", "-pix_fmt", "yuv420p", str(temporary)],
            input=np.stack(frames).tobytes(), check=True,
        )
        verify_video(temporary)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        # Atomic no-clobber publication, even if another process creates path.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify complete video coverage, no rollouts")
    parser.add_argument("--dry-run", action="store_true", help="List missing videos, no rollouts")
    args = parser.parse_args()
    if args.check and args.dry_run:
        parser.error("Choose --check or --dry-run")
    torch.set_num_threads(1)
    manifest = REPO / "artifacts/T05-opt-to-pessi/videos/supplement_fb.jsonl"
    missing_total = 0
    for seed, source in SOURCES:
        with np.load(ROOT / "data" / source, allow_pickle=False) as archive:
            data = dict(archive)
        validate(data, source)
        directory = manifest.parent / ("fb_" + source[len("zs_fb_"):-4])
        rows = selected_rows(data)
        missing = [row for row in rows if not (directory / video_name(data, row)).exists()]
        missing_total += len(missing)
        print(f"seed {seed}: {len(rows)} required, {len(missing)} missing", flush=True)
        if args.dry_run:
            for row in missing:
                print(video_name(data, row), flush=True)
            continue
        if args.check:
            if missing:
                for row in missing:
                    print(f"Missing: {directory / video_name(data, row)}", flush=True)
                continue
            for row in rows:
                verify_video(directory / video_name(data, row))
            # Newly added combined-reward videos must retain their provenance.
            entries = []
            if manifest.exists():
                entries = [json.loads(line) for line in manifest.read_text().splitlines() if line]
            by_video = {entry["video"]: entry for entry in entries}
            for row in rows:
                if data["family"][row] != "ReBRAC+E2E":
                    continue
                path = directory / video_name(data, row)
                entry = by_video.get(str(path.relative_to(REPO)))
                if entry is None or entry["video_sha256"] != sha256(path):
                    raise ValueError(f"Missing or mismatched supplement provenance: {path}")
            continue
        if not missing:
            continue
        checkpoint, expected_hash = CHECKPOINTS[seed]
        checkpoint = REPO / checkpoint
        if sha256(checkpoint) != expected_hash:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        agent = load_agent(checkpoint, torch.device("cpu"))
        agent.eval()
        directory.mkdir(parents=True, exist_ok=True)
        for row in missing:
            path = directory / video_name(data, row)
            start = time.monotonic()
            record(agent, data["z"][row], path)
            entry = dict(
                video=str(path.relative_to(REPO)), video_sha256=sha256(path),
                checkpoint=str(checkpoint.relative_to(REPO)), checkpoint_sha256=expected_hash,
                latent_source=str((ROOT / "data" / source).relative_to(REPO)),
                latent_source_sha256=sha256(ROOT / "data" / source), row=row,
                training_seed=seed, task=str(data["task"][row]), family=str(data["family"][row]),
                target_angle=float(data["target_angle"][row]) if np.isfinite(data["target_angle"][row]) else None,
                environment_seed=0, action_batch_size=1, steps=1000, frame_stride=2,
                torch_version=torch.__version__, numpy_version=np.__version__,
                mujoco_gl=os.environ.get("MUJOCO_GL"),
                replay_note="New replay, not guaranteed identical to historical evaluation trajectory",
            )
            with manifest.open("a") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"Recorded {path.name} in {time.monotonic() - start:.1f}s", flush=True)
    if args.check and missing_total:
        raise SystemExit(f"Missing {missing_total} videos")


if __name__ == "__main__":
    main()
