"""Record missing rollout videos of stored latents. Never trains, infers z or rewrites CSVs.

Videos replay deterministic, single-observation actions from environment seed 0.
This is not guaranteed to reproduce historical, batched evaluation trajectories.
Each model's videos go to <video-dir>/fb_<job>/; every new video is appended with its
provenance to --manifest.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl python -m zero_shot_rl.rollout.video --config <sources.json> \
        --video-dir <dir> --manifest <dir>/<name>.jsonl --families <family> ... [--check | --dry-run]
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from agents.base import load_agent
from zero_shot_rl.latent.archive import family_rows, validate
from zero_shot_rl.provenance import no_clobber, sha256
from zero_shot_rl.rollout.env import render_episode, write_video

REPO = Path(__file__).resolve().parents[2]


def video_name(data, row):
    return f"{data['task'][row]}_{data['family'][row]}_{data['target_angle'][row]:g}_{row}.mp4"


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


def record(agent, z_np, path):
    """Render, encode and verify one video; atomic, never overwrites."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    frames = render_episode(agent, z_np)
    with no_clobber(path) as temporary:
        write_video(frames, temporary)
        verify_video(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--manifest-families", nargs="*", default=[],
                        help="--check also requires a matching manifest entry for these families")
    parser.add_argument("--tasks", nargs="+", default=["stand", "walk", "run", "flip"])
    parser.add_argument("--check", action="store_true", help="Verify complete video coverage, no rollouts")
    parser.add_argument("--dry-run", action="store_true", help="List missing videos, no rollouts")
    args = parser.parse_args()
    if args.check and args.dry_run:
        parser.error("Choose --check or --dry-run")
    torch.set_num_threads(1)
    video_dir, manifest = args.video_dir.resolve(), args.manifest.resolve()
    missing_total = 0
    for model in json.loads(args.config.read_text())["models"]:
        seed, latents = model["training_seed"], REPO / model["latents"]
        with np.load(latents, allow_pickle=False) as archive:
            data = dict(archive)
        validate(data, latents.name, args.tasks, data["z"].shape[1])
        directory = video_dir / f"fb_{model['job']}"
        rows = family_rows(data, args.families)
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
            entries = []
            if manifest.exists():
                entries = [json.loads(line) for line in manifest.read_text().splitlines() if line]
            by_video = {entry["video"]: entry for entry in entries}
            for row in rows:
                if data["family"][row] not in args.manifest_families:
                    continue
                path = directory / video_name(data, row)
                entry = by_video.get(str(path.relative_to(REPO)))
                if entry is None or entry["video_sha256"] != sha256(path):
                    raise ValueError(f"Missing or mismatched video provenance: {path}")
            continue
        if not missing:
            continue
        checkpoint, expected_hash = REPO / model["checkpoint"], model["checkpoint_sha256"]
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
                latent_source=str(latents.relative_to(REPO)),
                latent_source_sha256=sha256(latents), row=row,
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
