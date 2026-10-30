"""Replay stored standard FB latents and write per-step behavior traces, without training.

For every model in --config, replays one converged latent per task and family
(zero_shot_rl.latent.archive.matched_rows) for EPISODES episodes and writes the trace to
the model's "trace" path, with a provenance manifest and source snapshot in
--provenance-dir. Refuses to overwrite traces or manifests.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl python -m zero_shot_rl.rollout.trace --config <sources.json> \
        --provenance-dir <dir> --families z_r <family> ... --target-angle 10
"""

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from zero_shot_rl.latent.archive import matched_rows, validate
from zero_shot_rl.provenance import no_clobber, sha256, snapshot
from zero_shot_rl.rollout.env import make_env, policy
from zero_shot_rl.rollout.trace_format import DT, EPISODES, STEPS, TRACE_KEYS, validate_trace

REPO = Path(__file__).resolve().parents[2]

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
    with no_clobber(path) as temporary:
        np.savez_compressed(temporary, **trace)


def source_files():
    files = [REPO / "agents/base.py", REPO / "agents/utils.py", REPO / "utils.py", REPO / "dmc.py"]
    for folder in ("zero_shot_rl", "agents/fb", "custom_dmc_tasks", "rewards/walker"):
        files += sorted((REPO / folder).rglob("*.py"))
    files += sorted((REPO / "custom_dmc_tasks").rglob("*.xml"))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--target-angle", type=float, required=True)
    parser.add_argument("--tasks", nargs="+", default=["stand", "walk", "run", "flip"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None, help="training seed; default: all")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(1)
    directory = args.provenance_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for model in json.loads(args.config.read_text())["models"]:
        seed = model["training_seed"]
        if args.seed is not None and seed != args.seed:
            continue
        path, latents = REPO / model["trace"], REPO / model["latents"]
        manifest = directory / f"{path.stem}.json"
        if path.exists() or manifest.exists():
            raise FileExistsError(f"Refusing to overwrite {path} or {manifest}")
        with np.load(latents, allow_pickle=False) as archive:
            data = dict(archive)
        validate(data, latents.name, args.tasks, data["z"].shape[1])
        checkpoint, expected = model["checkpoint"], model["checkpoint_sha256"]
        if sha256(REPO / checkpoint) != expected:
            raise ValueError(f"Checkpoint hash mismatch: seed {seed}")
        agent = load_agent(REPO / checkpoint, torch.device("cpu"))
        if not isinstance(agent, FB):
            raise ValueError("Only standard FB is allowed")
        agent.eval()
        rows = matched_rows(data, args.tasks, args.families, args.target_angle)
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
        validate_trace(trace, data, rows)
        metadata = dict(
            training_seed=seed, trace=str(path.relative_to(REPO)),
            git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
            git_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
            checkpoint=checkpoint, checkpoint_sha256=expected,
            latent_source=str(latents.relative_to(REPO)),
            latent_source_sha256=sha256(latents),
            # Keep actual source bytes, including untracked code, locally.
            code_sha256=snapshot(source_files(), REPO, directory / f"source_seed{seed}"),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            torch_version=torch.__version__, numpy_version=np.__version__, mujoco_version=mujoco.__version__,
            steps=STEPS, episodes=EPISODES, control_timestep=DT, action_batch_size=EPISODES,
            workers=args.workers, policy="deterministic mean, clamped [-1,1]",
            note="New local replay, not an exact reconstruction of historical evaluation or videos",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
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
