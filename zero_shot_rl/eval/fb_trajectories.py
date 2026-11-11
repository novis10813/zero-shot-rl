"""Full walker trajectories (observation, action, reward) of standard FB with its original z_r.

For every model in --config and every task, replays the stored z_r latent (the one inferred
from the unmodified task reward) for --episodes deterministic 1000-step episodes. Episode e
starts from the reset of make_env(e), so episode e is the same start state for every model.
Writes one NPZ per model with arrays [task, episode, step, ...]:

    observations  [T, E, 1000, d_s] observation before the action of every step
    actions       [T, E, 1000, d_a]
    rewards       [T, E, 1000]      task reward after each step
    z             [T, d_z]          latent used per task
    tasks, episodes, dt

and a provenance JSON in --out-dir. Refuses to overwrite either. --compare-trace checks the
rewards of the shared episodes against a trace written by zero_shot_rl.eval.trace and records
the largest difference without failing.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.fb_trajectories \
        --config <sources.json> --out-dir <dir> --episodes 50 [--tasks stand walk run flip] \
        [--seed 0] [--workers 4] [--batch 20] [--compare-trace <behavior_fb_seed0.npz> ...]
"""

import argparse
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
from zero_shot_rl.env.walker import STEPS, make_env, run_episodes
from zero_shot_rl.latent.archive import matched_rows, validate
from zero_shot_rl.provenance import no_clobber, sha256

REPO = Path(__file__).resolve().parents[2]
DT = 0.025
ALL_TASKS = ["stand", "walk", "run", "flip"]

G = {}


def rollout(row):
    """All episodes of one latent row, as a dict of arrays with the episode on axis 0.

    Episodes run in batches of G["batch"] (one policy call per step for a batch). The batch
    size changes floating point results, and trajectories diverge within tens of steps, so
    reproducing zero_shot_rl.eval.trace (batch 20) needs batch 20.
    """
    torch.set_num_threads(1)
    agent, data, episodes, batch = G["agent"], G["data"], G["episodes"], G["batch"]
    reward_fn = importlib.import_module(f"rewards.walker.{data['task'][row]}").reward_function
    z = torch.as_tensor(data["z"][row], dtype=torch.float32).unsqueeze(0)
    chunks = []
    for start in range(0, episodes, batch):
        envs = [make_env(e) for e in range(start, min(start + batch, episodes))]
        try:
            if any(not np.isclose(env.control_timestep(), DT) for env in envs):
                raise ValueError("Unexpected timestep")
            chunks.append(run_episodes(agent, z, envs, reward_fn=reward_fn, stride=1))
        finally:
            for env in envs:
                env.close()
    return {
        "observations": np.concatenate([c["observations"] for c in chunks]).astype(np.float32),
        "actions": np.concatenate([c["actions"] for c in chunks]).astype(np.float32),
        "rewards": np.concatenate([c["rewards"] for c in chunks]),
    }


def compare_with_trace(trace_path, data, tasks, rows, rewards):
    """Largest absolute reward difference on the episodes shared with `trace_path`."""
    with np.load(trace_path, allow_pickle=False) as trace:
        trace_rows = list(trace["rows"])
        reference = trace["reward"]
    worst = 0.0
    for i, task in enumerate(tasks):
        index = [j for j, r in enumerate(trace_rows)
                 if data["task"][r] == task and data["family"][r] == "z_r"]
        if len(index) != 1:
            raise ValueError(f"Trace has no unique z_r row for {task}")
        shared = min(reference.shape[1], rewards.shape[1])
        difference = reference[index[0], :shared] - rewards[i, :shared]
        worst = max(worst, float(np.abs(difference).max()))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=20, help="episodes per policy batch")
    parser.add_argument("--seed", type=int, default=None, help="training seed; default: all")
    parser.add_argument("--compare-trace", nargs="*", type=Path, default=[],
                        help="one trace per model in config order (with --seed: that model's)")
    args = parser.parse_args()
    if min(args.episodes, args.workers, args.batch) < 1:
        parser.error("--episodes, --workers and --batch must be positive")
    torch.set_num_threads(1)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in json.loads(args.config.read_text())["models"]
              if args.seed is None or m["training_seed"] == args.seed]
    if args.compare_trace and len(args.compare_trace) != len(models):
        parser.error("--compare-trace needs one trace per selected model")
    for n, model in enumerate(models):
        seed = model["training_seed"]
        path = out_dir / f"fb_trajectories_seed{seed}.npz"
        manifest = path.with_suffix(".json")
        if path.exists() or manifest.exists():
            raise FileExistsError(f"Refusing to overwrite {path} or {manifest}")
        latents = REPO / model["latents"]
        with np.load(latents, allow_pickle=False) as archive:
            data = dict(archive)
        validate(data, latents.name, ALL_TASKS, data["z"].shape[1])
        checkpoint = REPO / model["checkpoint"]
        if sha256(checkpoint) != model["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: seed {seed}")
        agent = load_agent(checkpoint, torch.device("cpu"))
        if not isinstance(agent, FB):
            raise ValueError("Only standard FB is allowed")
        agent.eval()
        rows = [matched_rows(data, [task], ["z_r"], None)[0] for task in args.tasks]
        G.update(agent=agent, data=data, episodes=args.episodes, batch=args.batch)
        results = []
        start = time.monotonic()
        with mp.get_context("fork").Pool(min(args.workers, len(rows))) as pool:
            for row, result in zip(rows, pool.imap(rollout, rows)):
                results.append(result)
                print(f"seed {seed}: {len(results)}/{len(rows)} {data['task'][row]} "
                      f"return {result['rewards'].sum(1).mean():.3f} "
                      f"elapsed {time.monotonic() - start:.1f}s", flush=True)
        arrays = {key: np.stack([r[key] for r in results]) for key in results[0]}
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Non-finite values in trajectories")
        if arrays["rewards"].shape != (len(rows), args.episodes, STEPS):
            raise ValueError("Unexpected reward shape")
        arrays.update(z=data["z"][rows], tasks=np.array(args.tasks),
                      episodes=np.arange(args.episodes), dt=np.array(DT))
        metadata = dict(
            training_seed=seed, trajectories=path.name, tasks=args.tasks, rows=rows,
            episodes=args.episodes, steps=STEPS, control_timestep=DT,
            checkpoint=model["checkpoint"], checkpoint_sha256=model["checkpoint_sha256"],
            latent_source=model["latents"], latent_source_sha256=sha256(latents),
            policy="deterministic mean, clamped [-1,1]", workers=args.workers, batch=args.batch,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
            torch_version=torch.__version__, numpy_version=np.__version__,
            mujoco_version=mujoco.__version__,
        )
        if args.compare_trace:
            metadata["max_reward_difference_vs_trace"] = compare_with_trace(
                args.compare_trace[n], data, args.tasks, rows, arrays["rewards"])
            print(f"seed {seed}: max reward difference vs trace "
                  f"{metadata['max_reward_difference_vs_trace']:.3g}", flush=True)
        with no_clobber(path) as temporary:
            np.savez_compressed(temporary, **arrays)
        metadata["trajectories_sha256"] = sha256(path)
        with manifest.open("x") as stream:
            json.dump(metadata, stream, indent=2)
        print(f"Wrote {path} and {manifest}", flush=True)


if __name__ == "__main__":
    main()
