"""Replay stored standard-FB latents for task-specific behavior, without training.

Writes NEW traces and provenance. Refuses to overwrite traces or legacy CSVs.
Run from the repo root with PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python.
"""

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from analyze_q1_fb import ROOT, SOURCES, validate
from analyze_task_behavior_fb import DT, EPISODES, STEPS, TRACE_KEYS, selected_rows, trace_name, validate_trace
from record_missing_fb import CHECKPOINTS, REPO, sha256
from rollout_eval import make_env, policy

G = {}


def torso_angle(physics):
    """Rotation about world +Y. For a pure Y rotation: xz=sin(theta), zz=cos(theta)."""
    return float(np.arctan2(physics.named.data.xmat["torso", "xz"],
                           physics.named.data.xmat["torso", "zz"]))


def rollout(row):
    torch.set_num_threads(1)
    agent, data = G["agent"], G["data"]
    envs = []
    try:
        envs = [make_env(e) for e in range(EPISODES)]
        if any(not np.isclose(env.control_timestep(), DT) for env in envs):
            raise ValueError("Unexpected timestep")
        obs = np.stack([env.reset().observation["observations"] for env in envs])
        result = {key: np.zeros((EPISODES, STEPS), dtype=np.float64) for key in TRACE_KEYS}
        result["angle"] = np.zeros((EPISODES, STEPS + 1), dtype=np.float64)
        result["angle"][:, 0] = [torso_angle(env.physics) for env in envs]
        z = torch.as_tensor(data["z"][row], dtype=torch.float32).expand(EPISODES, -1)
        reward_fn = importlib.import_module(f"rewards.walker.{data['task'][row]}").reward_function
        with torch.inference_mode():
            for step in range(STEPS):
                actions = policy(agent, torch.as_tensor(obs, dtype=torch.float32), z).numpy()
                for episode, env in enumerate(envs):
                    obs[episode] = env.step(actions[episode]).observation["observations"]
                    physics = env.physics
                    result["reward"][episode, step] = reward_fn(physics)
                    result["speed"][episode, step] = physics.horizontal_velocity()
                    result["height"][episode, step] = physics.torso_height()
                    result["upright"][episode, step] = physics.torso_upright()
                    result["angular_momentum"][episode, step] = physics.angmomentum()
                    result["angle"][episode, step + 1] = torso_angle(physics)
        return result
    finally:
        for env in envs:
            env.close()


def publish_npz(path, trace):
    temporary = path.with_suffix(".partial.npz")
    try:
        np.savez_compressed(temporary, **trace)
        os.link(temporary, path)  # atomic no-clobber
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=[s for s, _ in SOURCES], default=None)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.data_dir = args.data_dir.resolve()
    if not args.data_dir.is_relative_to(REPO):
        parser.error("--data-dir must be inside the repository")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    directory = REPO / "artifacts/T05-opt-to-pessi/task-behavior"
    directory.mkdir(parents=True, exist_ok=True)
    for seed, source in SOURCES:
        if args.seed is not None and seed != args.seed:
            continue
        path = args.data_dir / trace_name(seed)
        manifest = directory / f"behavior_fb_seed{seed}.json"
        if path.exists() or manifest.exists():
            raise FileExistsError(f"Refusing to overwrite {path} or {manifest}")
        with np.load(args.data_dir / source, allow_pickle=False) as archive:
            data = dict(archive)
        validate(data, source)
        checkpoint, expected = CHECKPOINTS[seed]
        if sha256(REPO / checkpoint) != expected:
            raise ValueError(f"Checkpoint hash mismatch: seed {seed}")
        agent = load_agent(REPO / checkpoint, torch.device("cpu"))
        if not isinstance(agent, FB):
            raise ValueError("Only standard FB is allowed")
        agent.eval()
        rows = selected_rows(data)
        G.update(agent=agent, data=data)
        results = []
        start = time.monotonic()
        # Linux fork shares the read-only model. Each worker uses one Torch thread.
        with mp.get_context("fork").Pool(args.workers) as pool:
            for row, result in zip(rows, pool.imap(rollout, rows)):
                results.append(result)
                print(f"seed {seed}: {len(results)}/{len(rows)} {data['task'][row]} "
                      f"{data['family'][row]} return {result['reward'].sum(1).mean():.3f} "
                      f"elapsed {time.monotonic() - start:.1f}s", flush=True)
        trace = {key: np.stack([r[key] for r in results]) for key in (*TRACE_KEYS, "angle")}
        trace.update(rows=np.array(rows), episodes=np.arange(EPISODES), dt=np.array(DT), z=data["z"][rows])
        validate_trace(trace, data)
        files = [Path(__file__), ROOT / "analyze_task_behavior_fb.py", ROOT / "rollout_eval.py",
                 ROOT / "analyze_q1_fb.py", ROOT / "record_missing_fb.py", REPO / "agents/base.py",
                 REPO / "agents/utils.py", REPO / "utils.py", REPO / "dmc.py"]
        for folder in ("agents/fb", "custom_dmc_tasks", "rewards/walker"):
            files += sorted((REPO / folder).rglob("*.py"))
        files += sorted((REPO / "custom_dmc_tasks").rglob("*.xml"))
        files = sorted(set(files))
        metadata = dict(
            training_seed=seed, trace=str(path.relative_to(REPO)),
            git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
            git_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
            checkpoint=checkpoint, checkpoint_sha256=expected,
            latent_source=str((args.data_dir / source).relative_to(REPO)),
            latent_source_sha256=sha256(args.data_dir / source),
            code_sha256={str(p.relative_to(REPO)): sha256(p) for p in files},
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            torch_version=torch.__version__, numpy_version=np.__version__, mujoco_version=mujoco.__version__,
            steps=STEPS, episodes=EPISODES, control_timestep=DT, action_batch_size=EPISODES,
            workers=args.workers, policy="deterministic mean, clamped [-1,1]",
            note="New local replay, not an exact reconstruction of historical evaluation or videos",
        )
        # Keep actual source bytes, including untracked evaluation code, locally.
        snapshot = directory / f"source_seed{seed}"
        for file in files:
            target = snapshot / file.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
        publish_npz(path, trace)
        manifest_created = False
        try:
            metadata["trace_sha256"] = sha256(path)
            with manifest.open("x") as stream:
                manifest_created = True
                json.dump(metadata, stream, indent=2)
        except BaseException:
            path.unlink()
            if manifest_created:
                manifest.unlink()
            raise
        print(f"Wrote {path} and provenance", flush=True)


if __name__ == "__main__":
    main()
