"""Online task inference of an FB checkpoint on Walker: OpTI-BFM and its baselines.

Rupf et al., "Optimistic Task Inference for Behavior Foundation Models" (arXiv 2510.20264),
section 5.1. Per task, --trials runs act for --episodes consecutive deterministic 1000-step
episodes. Run i uses one environment seeded with i, reset before every episode, so episodes
of a run start from different states and run i starts from the same states for every method.
After each step the run observes the task reward of the new state; the belief of UCB and TS
carries over between episodes of a run and starts from the prior at the start of a task.

--methods lists rules as METHOD[:PARAM][@PERIOD] (zero_shot_rl.online.selectors):

    fb          z_r of offline task inference (zero_shot_rl.latent.task_z), fixed
    random      w uniform on the sphere
    ucb:BETA    OpTI-BFM
    ts:SIGMA    OpTI-BFM-TS

A new w is chosen every PERIOD steps of an episode (default 1, the per-step rule of the
paper; 1000 is the episodic variant -EP of its section 5.3). phi and psi come from
zero_shot_rl.online.fb, with Cov_D(B) over the --dataset_transitions training subsample.

--out gets one row per rule, run and episode: method, param, period, task, trial, episode,
return (sum of the task reward) and cos, the cosine between the belief mean and z_r at the
end of the episode (UCB and TS only). --z_dir gets one <rule>.npz per rule with, per task,
the least-squares task vector of eq. 4 from the first n transitions of every run for each n
of --snapshots, as an array [trials, len(snapshots), d]
(zero_shot_rl.eval.fixed_z_returns evaluates them).

--predictions_dir gets one <rule>.npz per rule (UCB and TS) with, per task <t>,

    <t>_belief  belief mean at the end of every episode [trials, episodes, d]
    <t>_pred    psi(s_t, w_t)^T mean, from the belief before the step's update, for the
                heads of F mean and min, at the steps of PREDICTION_STEPS
                [trials, episodes, len(PREDICTION_STEPS), 2]
    <t>_actual  the return that prediction estimates, sum_i gamma^i r_{t+i} to the end of
                the episode [trials, episodes, len(PREDICTION_STEPS)]

Logging does not change the run. zero_shot_rl.eval.latent_ranking uses the belief means.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.online_inference \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> \
        --methods fb random ucb:0.1 ts:0.01 ucb:0.1@1000 --out runs.csv --z_dir z/

--dynamics <yaml> runs the episodes in a shifted environment (zero_shot_rl.env.dynamics);
z_r still comes from the dataset.
"""

import csv
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.agent import FB
from zero_shot_rl.env.dynamics import load_dynamics
from zero_shot_rl.env.walker import STEPS, make_env
from zero_shot_rl.latent import task_z
from zero_shot_rl.online.fb import FBModel
from zero_shot_rl.online.selectors import TS, UCB, OfflineZ, Random
from zero_shot_rl.provenance import no_clobber

FIELDS = ["method", "param", "period", "task", "trial", "episode", "return", "cos"]
SNAPSHOTS = [100, 200, 500, 1000, 2000, 5000, 10000]
PREDICTION_STEPS = list(range(0, 500, 50))  # leaves 500 steps, gamma^500 < 1e-4


def parse_rule(text):
    """'ucb:0.1@1000' -> ('ucb', 0.1, 1000)."""
    rule, _, period = text.partition("@")
    method, _, param = rule.partition(":")
    if method not in ("fb", "random", "ucb", "ts"):
        raise ValueError(f"unknown method in {text!r}")
    if (method in ("ucb", "ts")) != bool(param):
        raise ValueError(f"{text!r}: ucb and ts take a parameter, the others none")
    return method, float(param) if param else float("nan"), int(period or 1)


def make_selector(args, rule, model, w_r, generator, device):
    method, param, _ = rule
    n = args.trials
    if method == "fb":
        return OfflineZ(n, w_r)
    if method == "random":
        return Random(n, model.d, generator)
    if method == "ucb":
        return UCB(model, n, param, lam=args.lam, candidates=args.candidates,
                   generator=generator, device=device)
    return TS(model, n, param, lam=args.lam, generator=generator, device=device)


def run_task(args, rule, model, task, task_index, w_r, reward_fn, device):
    """Returns the CSV rows, the snapshot task vectors [trials, len(snapshots), d] and the
    arrays of --predictions_dir."""
    n = args.trials
    generator = torch.Generator().manual_seed(task_index)
    method, param, period = rule
    selector = make_selector(args, rule, model, w_r, generator, device)
    envs = [make_env(i, args.dynamics) for i in range(n)]
    snapshots = [s for s in args.snapshots if s <= args.episodes * STEPS]
    horizon = max(snapshots, default=0)
    phis = torch.zeros(n, horizon, model.d, dtype=torch.float64)
    rewards = torch.zeros(n, horizon, dtype=torch.float64)
    z_n = np.zeros((n, len(snapshots), model.d))
    log = args.predictions_dir and hasattr(selector, "belief")
    predictions = {"belief": np.zeros((n, args.episodes, model.d)),
                   "pred": np.zeros((n, args.episodes, len(PREDICTION_STEPS), 2)),
                   "actual": np.zeros((n, args.episodes, len(PREDICTION_STEPS)))}
    discounts = args.discount ** np.arange(STEPS)
    rows, step = [], 0
    for episode in range(args.episodes):
        obs = np.stack([env.reset().observation["observations"] for env in envs])
        returns = np.zeros(n)
        episode_rewards = np.zeros((n, STEPS))
        for t in range(STEPS):
            s = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if t % period == 0:
                w = selector.choose(s).to(device)
            if log and t in PREDICTION_STEPS:
                sf = model.psi(s, w[:, None, :]).double()[:, 0]  # [n, heads, d]
                value = (sf * selector.belief.mean[:, None, :]).sum(-1).cpu().numpy()
                predictions["pred"][:, episode, PREDICTION_STEPS.index(t)] = np.stack(
                    [value.mean(1), value.min(1)], axis=1)
            a = model.policy(s, w).cpu().numpy()
            r = np.zeros(n)
            for i, env in enumerate(envs):
                obs[i] = env.step(a[i]).observation["observations"]
                r[i] = reward_fn(env.physics)
            episode_rewards[:, t] = r
            phi = model.phi(torch.as_tensor(obs, dtype=torch.float32, device=device))
            selector.observe(phi, torch.as_tensor(r, device=device))
            returns += r
            if step < horizon:
                phis[:, step], rewards[:, step] = phi.cpu(), torch.as_tensor(r)
            step += 1
            for k, size in enumerate(snapshots):
                if step == size:
                    z_n[:, k] = [model.lstsq(phis[i, :size], rewards[i, :size]).numpy()
                                 for i in range(n)]
        if log:
            for k, t in enumerate(PREDICTION_STEPS):
                predictions["actual"][:, episode, k] = (
                    episode_rewards[:, t:] * discounts[: STEPS - t]).sum(1)
            predictions["belief"][:, episode] = selector.belief.mean.cpu().numpy()
        if hasattr(selector, "belief"):
            mean = selector.belief.mean.cpu()
            cos = torch.nn.functional.cosine_similarity(mean, w_r.double().expand_as(mean), dim=1)
        else:
            cos = torch.full((n,), float("nan"))
        rows += [{"method": method, "param": param, "period": period,
                  "task": task, "trial": i, "episode": episode, "return": returns[i],
                  "cos": cos[i].item()} for i in range(n)]
        print(f"{method}:{param}@{period} {task} episode {episode}: "
              f"mean return {returns.mean():.1f}", flush=True)
    for env in envs:
        env.close()
    return rows, z_n, snapshots, predictions


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--methods", nargs="+", required=True,
                        help="rules METHOD[:PARAM][@PERIOD], e.g. fb ucb:0.1 ts:0.01@1000")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--tasks", nargs="+", default=task_z.TASKS)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--snapshots", nargs="*", type=int, default=SNAPSHOTS)
    parser.add_argument("--dynamics", type=str, default=None,
                        help="yaml of zero_shot_rl.env.dynamics; default is the stock environment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--z_dir", type=str, default=None)
    parser.add_argument("--predictions_dir", type=str, default=None)
    parser.add_argument("--discount", type=float, default=0.98)
    args = parser.parse_args()
    rules = {text: parse_rule(text) for text in args.methods}
    args.dynamics = load_dynamics(args.dynamics) if args.dynamics else None

    device = task_z.device()
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    buffer, observations_z, rewards_z = task_z.inference_set(
        Namespace(seed=args.seed, discount=0.98, dataset_path=args.dataset_path,
                  dataset_transitions=args.dataset_transitions,
                  z_inference_steps=args.z_inference_steps),
        torch.device("cpu"),
    )
    model = FBModel(agent, buffer.storage["next_observations"].to(device))
    reward_fns = task_z.reward_functions(args, device)
    del buffer

    w_r = {task: torch.as_tensor(
        agent.infer_z(observations_z.to(device), rewards_z[task].to(device)),
        dtype=torch.float64,
    ) for task in args.tasks}
    rows, z_out, prediction_out = [], {text: {} for text in rules}, {text: {} for text in rules}
    for text, rule in rules.items():
        for task_index, task in enumerate(args.tasks):
            start = time.time()
            task_rows, z_n, snapshots, predictions = run_task(
                args, rule, model, task, task_index, w_r[task], reward_fns[task], device)
            rows += task_rows
            z_out[text][task] = z_n
            prediction_out[text].update({f"{task}_{k}": v for k, v in predictions.items()})
            print(f"{text} {task}: {time.time() - start:.0f} s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(out) as path, open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                             for k, v in row.items()})
    if args.z_dir:
        z_dir = Path(args.z_dir)
        z_dir.mkdir(parents=True, exist_ok=True)
        for text, by_task in z_out.items():
            with no_clobber(z_dir / f"{text}.npz") as path:
                np.savez(path, snapshots=np.array(snapshots), **by_task)
    if args.predictions_dir:
        directory = Path(args.predictions_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for text, arrays in prediction_out.items():
            if arrays["%s_belief" % args.tasks[0]].any():
                with no_clobber(directory / f"{text}.npz") as path:
                    np.savez(path, steps=np.array(PREDICTION_STEPS), **arrays)


if __name__ == "__main__":
    main()
