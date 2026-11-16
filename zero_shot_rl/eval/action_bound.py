"""Action-bound ceiling of the one-step action adaptation under a gear shift (no training).

Same stock rollouts as zero_shot_rl.eval.adapt_step (FB actor at the stored z_r of each --tasks,
--episodes per task, environment seeds --first-env-seed + episodes * task index + j). Every
recorded (physics_t, a_t) is replayed one step in each dynamics of --dynamics (the first being the
stock walker) with the stock action a_t and with a*_t = gear_inverse(a_t, gear of that dynamics),
the best action inside [-1, 1] for a gear multiplier. Rows with gear 1 have a* = a and are not
replayed twice. Writes --out (NPZ), N = episodes * tasks * steps samples over all steps:

    configs [C], gear [C]
    task, episode, t [N]
    a [N, d_a]                  the stock action
    b_none, s_none [C, N]       |x(s'_k(a)) - x(s'_0(a))|, x = B_0 (same FB checkpoint) or the observation
    b_star, s_star [C, N]       |x(s'_k(a*)) - x(s'_0(a))|

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.action_bound --checkpoint <fb.pt> \
        --latents <zs_fb.npz> --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml --out <action_bound.npz>
"""

import json
import multiprocessing as mp
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.base import load_agent
from zero_shot_rl.data.multi_dynamics import dynamics_config, sample_sets
from zero_shot_rl.env.walker import make_env, run_episodes
from zero_shot_rl.eval.adapt_step import G, replay_all
from zero_shot_rl.latent.action_adapt import gear_inverse
from zero_shot_rl.latent.archive import matched_rows
from zero_shot_rl.provenance import no_clobber, sha256


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["stand", "walk", "run", "flip"])
    parser.add_argument("--episodes", type=int, default=5, help="per task")
    parser.add_argument("--first-env-seed", type=int, default=300000)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    torch.set_num_threads(4)
    start_time = time.monotonic()

    config = yaml.safe_load(args.dynamics.read_text())
    names = [v["name"] for v in config["values"]]
    friction, gear = sample_sets(config)
    dynamics = [dynamics_config(f, np.full(6, g)) for f, g in zip(friction, gear)]
    agent = load_agent(args.checkpoint, torch.device("cpu"))
    agent.eval()
    with np.load(args.latents, allow_pickle=False) as archive:
        data = dict(archive)

    act, phys, task_of, episode_of = [], [], [], []
    for i, task in enumerate(args.tasks):
        z = torch.as_tensor(data["z"][matched_rows(data, [task], ["z_r"], None)[0]], dtype=torch.float32)
        seeds = args.first_env_seed + args.episodes * i + np.arange(args.episodes)
        run = run_episodes(agent, z, [make_env(int(s), dynamics[0]) for s in seeds], stride=1, physics=True)
        act.append(run["actions"])
        phys.append(run["physics"])
        task_of += [task] * args.episodes
        episode_of += list(seeds)
    act, phys = np.concatenate(act).astype(np.float32), np.concatenate(phys)
    n_ep, steps, d_a = act.shape
    d_s = run["observations"].shape[-1]
    print(f"rollouts {n_ep} x {steps} elapsed {time.monotonic() - start_time:.0f}s", flush=True)
    G.update(dynamics=dynamics, physics=phys.reshape(n_ep * steps, -1), obs_dim=d_s)
    flat_act = act.reshape(n_ep * steps, d_a)
    every = np.arange(n_ep * steps)

    shifted = [k for k in range(len(names)) if gear[k] != 1.0]
    with mp.get_context("fork").Pool(args.workers) as pool:
        nxt = replay_all(pool, [(k, every, flat_act) for k in range(len(names))])
        print(f"replays elapsed {time.monotonic() - start_time:.0f}s", flush=True)
        star = replay_all(pool, [(k, every, gear_inverse(flat_act, gear[k])) for k in shifted])
        print(f"a* replays elapsed {time.monotonic() - start_time:.0f}s", flush=True)
    star = dict(zip(shifted, star))

    def backward(x):
        with torch.no_grad():
            return agent.FB.backward_representation(torch.as_tensor(x, dtype=torch.float32)).numpy()

    norm = lambda x: np.linalg.norm(x, axis=-1)  # noqa: E731
    b_stock = backward(nxt[0])
    b_none, s_none = np.zeros((len(names), len(every)), np.float32), np.zeros((len(names), len(every)), np.float32)
    b_star, s_star = np.zeros_like(b_none), np.zeros_like(s_none)
    for k in range(len(names)):
        b_none[k], s_none[k] = norm(backward(nxt[k]) - b_stock), norm(nxt[k] - nxt[0])
        x = star.get(k, nxt[k])
        b_star[k], s_star[k] = norm(backward(x) - b_stock), norm(x - nxt[0])
    for k, name in enumerate(names):
        print(f"{name:22s} median |dB| none {np.median(b_none[k]):.4f}  a* {np.median(b_star[k]):.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(args.out) as temporary:
        np.savez_compressed(
            temporary, configs=np.array(names), gear=np.asarray(gear, np.float32),
            task=np.repeat(task_of, steps), episode=np.repeat(episode_of, steps), t=np.tile(np.arange(steps), n_ep),
            a=flat_act, b_none=b_none, s_none=s_none, b_star=b_star, s_star=s_star)
    meta = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_sha256=sha256(args.checkpoint), latents_sha256=sha256(args.latents),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        out_sha256=sha256(args.out))
    with args.out.with_suffix(".json").open("x") as stream:
        json.dump(meta, stream, indent=2)
    print(f"Wrote {args.out} elapsed {time.monotonic() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
