"""One-step check of the action adaptation of zero_shot_rl.latent.action_adapt.

For one FB training seed: deterministic stock rollouts of the FB actor at the stored z_r of each
--tasks (--episodes per task, environment seeds --first-env-seed + episodes * task index + j).
Every recorded (physics_t, a_t) is replayed one step in each dynamics of --dynamics (a config with
listed `values`, the first being the stock walker, as tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml),
giving s'_k(a_t). At the steps t = H, H + stride, ... the context of dynamics k is

    window  c1 = E(the H transitions (s_j, a_j, s'_k(a_j)), j = t - H .. t - 1)
    oracle  c1 = normalise(mean of the window contexts of dynamics k over all sampled steps)

and c0 the oracle context of the stock dynamics. For each context kind and each (K, lr) of --ks x
--lrs, a' = adapt(model, s_t, a_t, c1, c0, K, lr) is replayed in dynamics k, giving s'_k(a').
The check is whether s'_k(a') is closer than s'_k(a) to the stock outcome s'_0(a), in B_0 (of the
same FB checkpoint) and in the observation. Writes --out (NPZ):

    configs [C], kinds ["window", "oracle"], ks [G], lrs [G]   (G = len(ks) * len(lrs), ks major)
    task, episode, t [N]                                       the sampled steps
    b_none, s_none     [C, N]        |x(s'_k(a)) - x(s'_0(a))|, x = B_0 or the observation
    b_adapt, s_adapt   [C, 2, G, N]  |x(s'_k(a')) - x(s'_0(a))|
    da                 [C, 2, G, N]  |a' - a|
    obj_before, obj_after [C, 2, G, N]  |g(s, ., c1) - g(s, a, c0)|^2 at a and a'
    ctx_dist           [C, N]        |c1_window - c0|

Row 0 (the stock walker) measures what the adaptation costs without a shift.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.adapt_step --checkpoint <fb.pt> \
        --latents <zs_fb.npz> --model <context_best.pt> \
        --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml --out <adapt_step.npz>
"""

import json
import multiprocessing as mp
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from agents.base import load_agent
from zero_shot_rl.data.multi_dynamics import dynamics_config, sample_sets
from zero_shot_rl.env.walker import make_env, run_episodes, set_physics
from zero_shot_rl.latent.action_adapt import adapt
from zero_shot_rl.latent.archive import matched_rows
from zero_shot_rl.latent.context_encoder import load_context_model
from zero_shot_rl.provenance import no_clobber, sha256

G = {}  # per-process state, filled before the pool forks


def replay(job):
    """(dynamics index, indices into G['physics'] rows, actions [n, d_a]) -> s' [n, d_s]."""
    torch.set_num_threads(1)
    k, rows, actions = job
    env = make_env(0, G["dynamics"][k])
    out = np.zeros((len(rows), G["obs_dim"]), np.float32)
    for i, (row, a) in enumerate(zip(rows, actions)):
        set_physics(env, G["physics"][row])
        out[i] = env.step(a).observation["observations"]
    env.close()
    return out


def replay_all(pool, k_rows_actions, chunk=2000):
    """Replays [(k, rows, actions)] in chunks over the pool, returns one array per entry."""
    jobs, owner = [], []
    for n, (k, rows, actions) in enumerate(k_rows_actions):
        for i in range(0, len(rows), chunk):
            jobs.append((k, rows[i:i + chunk], actions[i:i + chunk]))
            owner.append(n)
    parts = [[] for _ in k_rows_actions]
    for n, out in zip(owner, pool.imap(replay, jobs)):
        parts[n].append(out)
    return [np.concatenate(p) for p in parts]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["stand", "walk", "run", "flip"])
    parser.add_argument("--episodes", type=int, default=5, help="per task")
    parser.add_argument("--first-env-seed", type=int, default=300000)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 20, 50])
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.01, 0.05])
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
    model, _ = load_context_model(args.model)
    model.requires_grad_(False)
    with np.load(args.latents, allow_pickle=False) as archive:
        data = dict(archive)

    # Stock rollouts with physics
    obs, act, phys, task_of, episode_of = [], [], [], [], []
    for i, task in enumerate(args.tasks):
        z = torch.as_tensor(data["z"][matched_rows(data, [task], ["z_r"], None)[0]], dtype=torch.float32)
        seeds = args.first_env_seed + args.episodes * i + np.arange(args.episodes)
        run = run_episodes(agent, z, [make_env(int(s), dynamics[0]) for s in seeds], stride=1, physics=True)
        obs.append(run["observations"])
        act.append(run["actions"])
        phys.append(run["physics"])
        task_of += [task] * args.episodes
        episode_of += list(seeds)
    obs, act, phys = np.concatenate(obs).astype(np.float32), np.concatenate(act).astype(np.float32), np.concatenate(phys)
    n_ep, steps, d_s = obs.shape
    print(f"rollouts {n_ep} x {steps} elapsed {time.monotonic() - start_time:.0f}s", flush=True)
    G.update(dynamics=dynamics, physics=phys.reshape(n_ep * steps, -1), obs_dim=d_s)
    flat_act = act.reshape(n_ep * steps, -1)
    every = np.arange(n_ep * steps)

    with mp.get_context("fork").Pool(args.workers) as pool:
        nxt = replay_all(pool, [(k, every, flat_act) for k in range(len(names))])
        nxt = np.stack(nxt).reshape(len(names), n_ep, steps, d_s)  # s'_k(a_t)
        print(f"replays elapsed {time.monotonic() - start_time:.0f}s", flush=True)

        ts = np.arange(args.h, steps, args.stride)
        ep_idx, t_idx = np.repeat(np.arange(n_ep), len(ts)), np.tile(ts, n_ep)
        rows = ep_idx * steps + t_idx
        window = t_idx[:, None] - args.h + np.arange(args.h)  # [N, H]
        s_w = torch.as_tensor(obs[ep_idx[:, None], window])
        a_w = torch.as_tensor(act[ep_idx[:, None], window])
        with torch.no_grad():
            c_win = torch.stack([model.encode(s_w, a_w, torch.as_tensor(nxt[k][ep_idx[:, None], window]) - s_w)
                                 for k in range(len(names))])  # [C, N, d_c]
        c_oracle = F.normalize(c_win.mean(1), dim=-1)  # [C, d_c]
        c0 = c_oracle[0:1]
        s_t, a_t = torch.as_tensor(obs[ep_idx, t_idx]), torch.as_tensor(act[ep_idx, t_idx])

        grid = [(k, lr) for k in args.ks for lr in args.lrs]
        shape = (len(names), 2, len(grid), len(rows))
        da, before, after = np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        adapted, jobs = {}, []
        for k in range(len(names)):
            for j, c1 in enumerate([c_win[k], c_oracle[k:k + 1]]):
                for g, (n_steps, lr) in enumerate(grid):
                    a2, ob, oa = adapt(model, s_t, a_t, c1, c0, n_steps, lr)
                    da[k, j, g] = (a2 - a_t).norm(dim=-1).numpy()
                    before[k, j, g], after[k, j, g] = ob.numpy(), oa.numpy()
                    adapted[(k, j, g)] = len(jobs)
                    jobs.append((k, rows, a2.numpy()))
        print(f"adapted elapsed {time.monotonic() - start_time:.0f}s", flush=True)
        replayed = replay_all(pool, jobs)
        print(f"adapted replays elapsed {time.monotonic() - start_time:.0f}s", flush=True)

    def backward(x):
        with torch.no_grad():
            return agent.FB.backward_representation(torch.as_tensor(x, dtype=torch.float32)).numpy()

    stock = nxt[0][ep_idx, t_idx]
    b_stock = backward(stock)
    norm = lambda x: np.linalg.norm(x, axis=-1)  # noqa: E731
    b_none, s_none = np.zeros(shape[:1] + shape[3:], np.float32), np.zeros(shape[:1] + shape[3:], np.float32)
    b_adapt, s_adapt = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    for k in range(len(names)):
        x = nxt[k][ep_idx, t_idx]
        b_none[k], s_none[k] = norm(backward(x) - b_stock), norm(x - stock)
        for j in range(2):
            for g in range(len(grid)):
                x = replayed[adapted[(k, j, g)]]
                b_adapt[k, j, g], s_adapt[k, j, g] = norm(backward(x) - b_stock), norm(x - stock)
    ctx_dist = (c_win - c0).norm(dim=-1).numpy()

    for k, name in enumerate(names):
        best = np.argmin(np.median(b_adapt[k, 0], -1))
        print(f"{name:22s} median |dB| none {np.median(b_none[k]):.4f}  window best {grid[best]} "
              f"{np.median(b_adapt[k, 0, best]):.4f}  oracle same {np.median(b_adapt[k, 1, best]):.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(args.out) as temporary:
        np.savez_compressed(
            temporary, configs=np.array(names), kinds=np.array(["window", "oracle"]),
            ks=np.array([g[0] for g in grid]), lrs=np.array([g[1] for g in grid]),
            task=np.array(task_of)[ep_idx], episode=np.array(episode_of)[ep_idx], t=t_idx,
            b_none=b_none, s_none=s_none, b_adapt=b_adapt, s_adapt=s_adapt, da=da,
            obj_before=before, obj_after=after, ctx_dist=ctx_dist)
    meta = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_sha256=sha256(args.checkpoint), latents_sha256=sha256(args.latents),
        model_sha256=sha256(args.model),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        out_sha256=sha256(args.out))
    with args.out.with_suffix(".json").open("x") as stream:
        json.dump(meta, stream, indent=2)
    print(f"Wrote {args.out} elapsed {time.monotonic() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
