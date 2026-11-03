"""Rule C: switch per step between the optimistic and pessimistic actions of a fixed FB.

Neggatu et al. (arXiv 2503.12222) Algorithm 1, with the behaviour-cloning action
replaced by the actor at a pessimistic latent (zero_shot_rl.latent.switch). At every
step s, with z_r and the CQL latent z_cql at --angle degrees from z_r
(tasks/T06-latent-switching/cql_sweep.py) for --task,

    a_opt  = pi(s, z_r),  a_pess = pi(s, z_cql)
    Q(a)   = mean_k F_k(s, a, z_r)^T z_r,  u(a) = std_k F_k(s, a, z_r)^T z_r
    take a_pess if Q(a_opt) - kappa u(a_opt) < Q(a_pess), else a_opt

with F_k the members of a zero_shot_rl.train.ensemble ensemble trained for the same
checkpoint. z_r is the unit latent: the rule compares Q and kappa u, which both scale with |z_hat_r|, so
the choice is the same in reward units, and Q, u and the gap below are in units of
|z_hat_r|. The paper's sigma_D term is left out, since it needs returns.

u is dominated by the members' offsets in the value level, which a_opt and a_pess at the
same state share: on walk u is about 50 while Q(a_opt) - Q(a_pess) has an interquartile
range of about 0.3, so the default kappas span 0 to 0.1, where the share of a_pess moves
from about one half to nearly all steps.

Each condition, "opt" (always a_opt), "pess" (always a_pess), a kappa of --kappas, or
"random" with a p of --random_ps (a_pess with probability p at every step, independent
of the ensemble: the control for the rule's share of a_pess), runs --episodes deterministic episodes of 1000 steps; episode e starts from the reset
of an environment seeded with e, as in zero_shot_rl.eval.latents. Per episode it records

    return     sum of the task reward
    pess_frac  fraction of steps that take a_pess
    switches   number of steps whose choice differs from the previous step's
    speed      mean horizontal velocity; fall_frac: fraction of steps with torso height < 0.6
    u_opt      mean u(a_opt)
    gap        mean Q(a_opt) - Q(a_pess)

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.switch \
        --checkpoint <fb.pt> --ensemble <ensemble.pt> --zs <zs_sweep.npz> --out <switch.csv>
"""

import csv
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from rewards import RewardFunctionConstructor
from zero_shot_rl.env.walker import make_env, policy
from zero_shot_rl.latent.switch import KAPPAS, choose_pess, latents, values
from zero_shot_rl.provenance import sha256
from zero_shot_rl.train.ensemble import load_ensemble

FALL_HEIGHT = 0.6
FIELDS = ["condition", "kappa", "p", "episode", "return", "pess_frac", "switches", "speed",
          "fall_frac", "u_opt", "gap"]

G = {}  # per-process state, filled before the pool forks


def rollout(condition):
    torch.set_num_threads(1)
    args, agent, ensemble, reward_fn = G["args"], G["agent"], G["ensemble"], G["reward_fn"]
    n = args.episodes
    envs = [make_env(e) for e in range(n)]
    obs = np.stack([env.reset().observation["observations"] for env in envs])
    z_r, z_cql = (torch.as_tensor(z, dtype=torch.float32).expand(n, -1) for z in G["z"])
    random_p = condition[1] if isinstance(condition, tuple) else None
    rng = np.random.default_rng([6, round(1e6 * (random_p or 0))])

    rewards, pess, heights, speeds, u_opt, gap = (np.zeros((n, 1000)) for _ in range(6))
    for t in range(1000):
        s = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a_opt, a_pess = policy(agent, s, z_r), policy(agent, s, z_cql)
            q_opt, u = values(ensemble, s, a_opt, z_r)
            q_pess, _ = values(ensemble, s, a_pess, z_r)
        if random_p is None:
            take = choose_pess(q_opt, u, q_pess, condition)
        else:
            take = rng.random(n) < random_p
        actions = np.where(take[:, None], a_pess.numpy(), a_opt.numpy())
        pess[:, t], u_opt[:, t], gap[:, t] = take, u, q_opt - q_pess
        for i, env in enumerate(envs):
            obs[i] = env.step(actions[i]).observation["observations"]
            p = env.physics
            rewards[i, t] = reward_fn(p)
            heights[i, t] = p.torso_height()
            speeds[i, t] = p.horizontal_velocity()

    if random_p is not None:
        name, kappa, p = "random", np.nan, random_p
    elif isinstance(condition, str):
        name, kappa, p = condition, np.nan, np.nan
    else:
        name, kappa, p = "switch", condition, np.nan
    return [
        {
            "condition": name, "kappa": kappa, "p": p, "episode": e, "return": rewards[e].sum(),
            "pess_frac": pess[e].mean(), "switches": int((np.diff(pess[e]) != 0).sum()),
            "speed": speeds[e].mean(), "fall_frac": (heights[e] < FALL_HEIGHT).mean(),
            "u_opt": u_opt[e].mean(), "gap": gap[e].mean(),
        }
        for e in range(n)
    ]


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ensemble", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--angle", type=float, default=7.5)
    parser.add_argument("--kappas", nargs="*", type=float, default=KAPPAS)
    parser.add_argument("--random_ps", nargs="*", type=float, default=[])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cpu")
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    ens_meta = torch.load(args.ensemble, map_location=device, weights_only=True)
    if ens_meta["checkpoint_sha256"] != sha256(Path(args.checkpoint)):
        raise ValueError(f"{args.ensemble} was trained for another checkpoint")
    ensemble = load_ensemble(Path(args.ensemble), device).eval()
    reward_constructor = RewardFunctionConstructor(
        domain_name="walker", task_names=[args.task], seed=0, device=device
    )
    G.update(args=args, agent=agent, ensemble=ensemble,
             reward_fn=reward_constructor.reward_functions[args.task],
             z=latents(np.load(args.zs), args.task, args.angle))

    conditions = ["opt", "pess", *args.kappas, *(("random", p) for p in args.random_ps)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f, mp.get_context("fork").Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for result in pool.imap_unordered(rollout, conditions):
            for rec in result:
                writer.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                                 for k, v in rec.items()})
            f.flush()
            print(f"{result[0]['condition']:6s} kappa {result[0]['kappa']:4g} "
                  f"p {result[0]['p']:4g} "
                  f"return {np.mean([r['return'] for r in result]):6.1f} "
                  f"pess_frac {np.mean([r['pess_frac'] for r in result]):.3f} "
                  f"switches {np.mean([r['switches'] for r in result]):6.1f}", flush=True)


if __name__ == "__main__":
    main()
