"""Walker transitions under many sampled dynamics, for training a dynamics context encoder.

`sample_sets(config)` draws the dynamics sets of a config (for example
tasks/T4.1.3.2-fb-aa/configs/multi_dynamics.yaml): sliding friction of all geoms and one gear
multiplier for the six actuators, log-uniform in their ranges, redrawn within a factor `margin`
of any test value. The first third of the sets change friction only, the second gear only, the
last both.

The entry runs, for one FB training seed, the episodes the config assigns to it: episode j of
set i belongs to number (episodes * i + j) mod 12, seed number // 4 and task number % 4, and
starts from the reset with environment seed first_env_seed + episodes * i + j. The policy is the
deterministic clamped actor mean with the stored z_r of the task (zero_shot_rl.env.walker.policy).
Episodes of the same task run in batches of --batch, one policy call per step. Writes
<out-dir>/multi_dynamics_seed<seed>.npz:

    observations  [E, 1001, d_s]  float32, the observation before every action and after the last
    actions       [E, 1000, d_a]  float32
    set           [E]             index of the dynamics set
    task          [E]             task name
    env_seed      [E]
    friction, gear [S]            parameters of every set (gear as multiplier of the XML gear)

and multi_dynamics_seed<seed>.json with the config and provenance. Refuses to overwrite.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.data.multi_dynamics \
        --config tasks/T4.1.3.2-fb-aa/configs/multi_dynamics.yaml --seed 0 \
        --checkpoint <fb.pt> --latents <zs_fb.npz> --out-dir out [--workers 5] [--batch 20] [--sets N]

--sets keeps only the first N sets (smoke tests).
"""

import argparse
import json
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

import custom_dmc_tasks.walker as cw
from agents.base import load_agent
from zero_shot_rl.env.dynamics import make_shifted_env
from zero_shot_rl.env.walker import STEPS, policy
from zero_shot_rl.latent.archive import matched_rows
from zero_shot_rl.provenance import no_clobber, sha256

SEEDS = 3
G = {}


def _draw(rng, low, high, tests, margin):
    while True:
        x = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        if all(abs(np.log(x / t)) >= np.log(margin) for t in tests):
            return x


def sample_sets(config):
    """Friction [S] and gear multiplier [S] of the config's dynamics sets."""
    rng = np.random.default_rng(config["seed"])
    n, f, g = config["sets"], config["friction"], config["gear"]
    friction, gear = np.full(n, nominal_friction()), np.ones(n)
    for i in range(n):
        group = 3 * i // n  # 0 friction only, 1 gear only, 2 both
        if group != 1:
            friction[i] = _draw(rng, *f["range"], f["test"], config["margin"])
        if group != 0:
            gear[i] = _draw(rng, *g["range"], g["test"], config["margin"])
    return friction, gear


def _xml_model():
    return cw.Physics.from_xml_string(*cw.get_model_and_assets()).model


def nominal_friction():
    friction = _xml_model().geom_friction[:, 0]
    if not np.allclose(friction, friction[0]):
        raise ValueError("Expected one sliding friction for all geoms")
    return float(friction[0])


def dynamics_config(friction, gear):
    """Config of zero_shot_rl.env.dynamics for one set."""
    return {"base": {"geom_friction": {"all": float(friction)},
                     "actuator_gear": [float(x) for x in gear * _xml_model().actuator_gear[:, 0]]}}


def assignment(config):
    """(set, seed, task, env_seed) of every episode, in set and episode order."""
    k, tasks = config["episodes"], config["tasks"]
    out = []
    for i in range(config["sets"]):
        for j in range(k):
            number = (k * i + j) % (SEEDS * len(tasks))
            out.append((i, number // len(tasks), tasks[number % len(tasks)], config["first_env_seed"] + k * i + j))
    return out


def run_batch(job):
    """Episodes of one task: [(set, env_seed)] -> observations [n, STEPS+1, d_s], actions [n, STEPS, d_a]."""
    torch.set_num_threads(1)
    task, episodes = job
    agent, friction, gear = G["agent"], G["friction"], G["gear"]
    z = G["z"][task].unsqueeze(0).expand(len(episodes), -1)
    envs = [make_shifted_env(seed, dynamics_config(friction[s], gear[s])) for s, seed in episodes]
    try:
        obs = np.stack([env.reset().observation["observations"] for env in envs])
        observations, actions = [obs.copy()], []
        for _ in range(STEPS):
            with torch.no_grad():
                a = policy(agent, torch.as_tensor(obs, dtype=torch.float32), z).numpy()
            actions.append(a)
            for e, env in enumerate(envs):
                obs[e] = env.step(a[e]).observation["observations"]
            observations.append(obs.copy())
    finally:
        for env in envs:
            env.close()
    return np.stack(observations, 1).astype(np.float32), np.stack(actions, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="FB training seed")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--sets", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = yaml.safe_load(args.config.read_text())
    friction, gear = sample_sets(config)
    episodes = [e for e in assignment(config) if e[1] == args.seed and (args.sets is None or e[0] < args.sets)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"multi_dynamics_seed{args.seed}.npz"
    manifest = path.with_suffix(".json")
    if path.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {path} or {manifest}")

    agent = load_agent(args.checkpoint, torch.device("cpu"))
    agent.eval()
    with np.load(args.latents, allow_pickle=False) as archive:
        data = dict(archive)
    z = {t: torch.as_tensor(data["z"][matched_rows(data, [t], ["z_r"], None)[0]], dtype=torch.float32)
         for t in config["tasks"]}
    G.update(agent=agent, friction=friction, gear=gear, z=z)

    jobs, order = [], []
    for task in config["tasks"]:
        mine = [e for e in episodes if e[2] == task]
        for start in range(0, len(mine), args.batch):
            chunk = mine[start:start + args.batch]
            jobs.append((task, [(e[0], e[3]) for e in chunk]))
            order.extend(chunk)
    start = time.monotonic()
    results = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for n, result in enumerate(pool.imap(run_batch, jobs), 1):
            results.append(result)
            print(f"seed {args.seed}: batch {n}/{len(jobs)} elapsed {time.monotonic() - start:.0f}s", flush=True)
    observations = np.concatenate([r[0] for r in results])
    actions = np.concatenate([r[1] for r in results])
    if not (np.isfinite(observations).all() and np.isfinite(actions).all()):
        raise ValueError("Non-finite values in trajectories")
    with no_clobber(path) as temporary:
        np.savez_compressed(
            temporary, observations=observations, actions=actions,
            set=np.array([e[0] for e in order]), task=np.array([e[2] for e in order]),
            env_seed=np.array([e[3] for e in order]), friction=friction, gear=gear)
    metadata = dict(
        config=config, config_path=str(args.config), training_seed=args.seed, sets_kept=args.sets,
        episodes=len(order), checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        latents=str(args.latents), latents_sha256=sha256(args.latents), batch=args.batch,
        policy="deterministic mean, clamped [-1,1]",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        torch_version=torch.__version__, mujoco_version=mujoco.__version__,
        trajectories_sha256=sha256(path))
    with manifest.open("x") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Wrote {path} ({len(order)} episodes) and {manifest}", flush=True)


if __name__ == "__main__":
    main()
