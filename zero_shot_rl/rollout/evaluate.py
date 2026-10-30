"""Roll out the fixed standard FB actor at every latent written by zero_shot_rl.latent.infer.

Each latent runs --episodes deterministic episodes (clipped mean action) of 1000 steps.
Episode e starts from the reset of an environment seeded with e, so all latents share
the same start states. Per episode it records

    return        sum of the latent's task reward
    fall_frac     fraction of steps with torso height < 0.6
    speed         mean horizontal velocity of the centre of mass
    upright       mean torso uprightness
    abs_action    mean |a|; sat_frac: fraction of action entries with |a| > 0.99
    knn_dist      mean distance of visited (s, a) to the nearest training (s, a),
                  per-dimension standardised (every --knn_every steps)
    m7            mean F(s, pi(s, z))^T u_1 over the same steps
    q_corr        (per latent) Pearson correlation of F(s, a)^T z with the discounted
                  return-to-go over steps t <= 750 (U6 for G0)

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl python -m zero_shot_rl.rollout.evaluate \
        --checkpoint <best.pt> --zs <zs.npz> --dataset_path <dataset.npz> --out <rollouts.csv>
"""

import csv
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor
from zero_shot_rl.rollout.env import forward, make_env, policy, render_episode, write_video

DISCOUNT = 0.98  # agents/fb/config.yaml
FALL_HEIGHT = 0.6
RTG_HORIZON = 750  # return-to-go is truncated by at most gamma^250 ~ 0.006 of its scale

G = {}  # per-process state, filled before the pool forks


def rollout(row):
    torch.set_num_threads(1)
    args, agent, data = G["args"], G["agent"], G["data"]
    task, z_np = data["task"][row], data["z"][row]
    reward_fn = G["reward_fns"][task]
    n = args.episodes
    envs = [make_env(e) for e in range(n)]
    obs = np.stack([env.reset().observation["observations"] for env in envs])
    z = torch.as_tensor(z_np, dtype=torch.float32).expand(n, -1)

    rewards = np.zeros((n, 1000))
    heights, speeds, uprights = (np.zeros((n, 1000)) for _ in range(3))
    abs_actions, sats = np.zeros((n, 1000)), np.zeros((n, 1000))
    probe_obs, probe_actions, probe_steps = [], [], []
    for t in range(1000):
        with torch.no_grad():
            actions = policy(agent, torch.as_tensor(obs, dtype=torch.float32), z).numpy()
        if t % args.knn_every == 0:
            probe_obs.append(obs.copy())
            probe_actions.append(actions.copy())
            probe_steps.append(t)
        for i, env in enumerate(envs):
            obs[i] = env.step(actions[i]).observation["observations"]
            p = env.physics
            rewards[i, t] = reward_fn(p)
            heights[i, t] = p.torso_height()
            speeds[i, t] = p.horizontal_velocity()
            uprights[i, t] = p.torso_upright()
        abs_actions[:, t] = np.abs(actions).mean(1)
        sats[:, t] = (np.abs(actions) > 0.99).mean(1)

    # probes: [steps, episodes, dim] -> [episodes * steps, dim]
    s = torch.as_tensor(np.stack(probe_obs, 1).reshape(-1, obs.shape[1]), dtype=torch.float32)
    a = torch.as_tensor(np.stack(probe_actions, 1).reshape(-1, actions.shape[1]))
    query = (torch.cat([s, a], 1) - G["sa_mean"]) / G["sa_std"]
    nearest = torch.full((len(query),), float("inf"))
    for start in range(0, len(G["sa"]), 20000):
        nearest = torch.minimum(nearest, torch.cdist(query, G["sa"][start:start + 20000]).min(1).values)
    with torch.no_grad():
        z_probe = torch.as_tensor(z_np, dtype=torch.float32).expand(len(s), -1)
        m7 = forward(agent, s, policy(agent, s, z_probe), z_probe)
        m7 = (m7 @ torch.as_tensor(G["u_1"])).numpy()
        q = (forward(agent, s, a, z_probe) @ torch.as_tensor(z_np)).numpy()

    # discounted return-to-go at the probe steps
    rtg = np.zeros((n, 1001))
    for t in range(999, -1, -1):
        rtg[:, t] = rewards[:, t] + DISCOUNT * rtg[:, t + 1]
    steps = np.array(probe_steps)
    rtg_probe = rtg[:, steps].reshape(-1)
    keep = np.tile(steps <= RTG_HORIZON, n)
    q_corr = float(np.corrcoef(q[keep], rtg_probe[keep])[0, 1])

    per_episode = len(steps)
    out = []
    for e in range(n):
        sl = slice(e * per_episode, (e + 1) * per_episode)
        out.append(
            {
                "row": row,
                "episode": e,
                "return": rewards[e].sum(),
                "fall_frac": (heights[e] < FALL_HEIGHT).mean(),
                "speed": speeds[e].mean(),
                "upright": uprights[e].mean(),
                "abs_action": abs_actions[e].mean(),
                "sat_frac": sats[e].mean(),
                "knn_dist": nearest[sl].mean().item(),
                "m7": m7[sl].mean(),
                "q_corr": q_corr,
            }
        )
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--transitions", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--knn_every", type=int, default=10)
    parser.add_argument("--families", nargs="+", default=None, help="default: all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--video_families", nargs="+", default=None)
    args = parser.parse_args()
    if args.video_dir is not None and args.video_families is None:
        parser.error("--video_dir requires --video_families")

    torch.set_num_threads(1)
    agent = load_agent(Path(args.checkpoint), torch.device("cpu"))
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    data = dict(np.load(args.zs))
    reward_constructor = RewardFunctionConstructor(
        domain_name="walker", task_names=sorted(set(data["task"])), seed=0, device=torch.device("cpu")
    )
    buffer = FBReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=Path(args.dataset_path),
        discount=DISCOUNT,
        transitions=args.transitions,
        device=torch.device("cpu"),
    )
    sa = torch.cat([buffer.storage["observations"], buffer.storage["actions"]], 1)
    sa_mean, sa_std = sa.mean(0), sa.std(0) + 1e-6

    rows = [
        i for i in range(len(data["task"]))
        if args.families is None or data["family"][i] in args.families
    ]

    def video(row):
        if args.video_dir is None or data["family"][row] not in args.video_families:
            return None
        name = f"{data['task'][row]}_{data['family'][row]}_{data['target_angle'][row]:g}_{row}.mp4"
        return Path(args.video_dir) / name

    G.update(
        args=args, agent=agent, data=data, reward_fns=reward_constructor.reward_functions,
        sa=(sa - sa_mean) / sa_std, sa_mean=sa_mean, sa_std=sa_std, u_1=data["u_1"],
    )

    fields = ["row", "task", "family", "strength", "target_angle", "angle", "episode",
              "return", "fall_frac", "speed", "upright", "abs_action", "sat_frac",
              "knn_dist", "m7", "q_corr"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f, mp.get_context("fork").Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for done, result in enumerate(pool.imap_unordered(rollout, rows), 1):
            for rec in result:
                r = rec["row"]
                rec.update(task=data["task"][r], family=data["family"][r],
                           strength=data["strength"][r], target_angle=data["target_angle"][r],
                           angle=data["angle"][r])
                writer.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                                 for k, v in rec.items()})
            f.flush()
            rets = [rec["return"] for rec in result]
            print(f"[{done}/{len(rows)}] {data['task'][r]} {data['family'][r]} "
                  f"angle {data['angle'][r]:.1f} return median {np.median(rets):.1f} "
                  f"knn {np.mean([rec['knn_dist'] for rec in result]):.3f}", flush=True)

    for row in rows:
        path = video(row)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_video(render_episode(agent, data["z"][row]), path)
            print(f"video {path}", flush=True)


if __name__ == "__main__":
    main()
