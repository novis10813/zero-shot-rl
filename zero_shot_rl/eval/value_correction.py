"""Online value correction of an FB checkpoint on Walker: train Q_adapt, then test it.

zero_shot_rl.online.value_correction, with z fixed at z_r of offline task inference
(zero_shot_rl.latent.task_z). Per variant of --variants and task, one Q_adapt is trained for
--episodes consecutive 1000-step episodes in one environment seeded with --train_seed (the
stock Walker or --dynamics), with one critic update per step once the buffer holds --warmup
transitions. Training executes the greedy candidate, or with probability --explore a uniform
one. The residual variant is residual soft Q-learning (zero_shot_rl.online.value_correction):
it executes a sample of softmax(score / --alpha), ignores --explore, and --omega weights the
prior policy.

After each episode count of --eval_every (0, every, 2 every, ... and the last one; 0 is before
training) or of the explicit list --eval_at the critics are held fixed and
the greedy rule runs one episode in each of the --eval_episodes environments seeded
0..eval_episodes-1, with the same candidate noise at every count. Variant `fb` is the frozen FB
actor in the same environments. The test seeds must differ from --train_seed.

--out_dir gets

    eval.csv    variant, task, train_episodes, seed, return, shift, gain, a0_frac
    train.csv   variant, task, episode, return, loss, q_pred, q_actual, shift, gain, a0_frac
    q/<variant>_<task>.pt   the critics after the last episode (QAdapt.save)
    q/<variant>_<task>_ep<n>.pt   with --save_every, the critics after n episodes, for
                            zero_shot_rl.eval.value_correction_checkpoints

With --train_only nothing is tested here (no eval.csv, no frozen FB rows).

`return` is the sum of the task reward, `shift` the mean over the steps of |a - a_0|, the
distance of the executed action to the FB action, and `gain` the mean of
Q_adapt(s, a) - Q_adapt(s, a_0), what Q_adapt claims the executed action gains over a_0 and
`a0_frac` the fraction of steps whose executed action is a_0 (the rows of variant `fb` have
0, 0 and 1). In train.csv the executed action includes the --explore draws. `loss` is the mean TD loss of the episode's updates. q_pred is
the mean Q_adapt of the executed action over the steps t < 700 of the episode and q_actual the
mean discounted return sum_i discount^i r_{t+1+i} to the end of the episode from the same
steps (discount^300 < 0.003).

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.value_correction \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> --dynamics <shift.yaml> \
        --out_dir out/
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
from zero_shot_rl.env.walker import STEPS, make_env, policy
from zero_shot_rl.latent import task_z
from zero_shot_rl.online.value_correction import VARIANTS, Buffer, QAdapt, reward_scale
from zero_shot_rl.provenance import no_clobber
from zero_shot_rl.utils import set_seed_everywhere

EVAL_FIELDS = ["variant", "task", "train_episodes", "seed", "return", "shift", "gain", "a0_frac"]
TRAIN_FIELDS = ["variant", "task", "episode", "return", "loss", "q_pred", "q_actual", "shift", "gain",
                "a0_frac"]
VALUE_STEPS = STEPS - 300  # steps whose discounted return the episode covers


def eval_points(episodes, every):
    """Episode counts to test at: 0, every, 2 every, ... and the last one."""
    return sorted({*range(0, episodes + 1, every), episodes})


def evaluate(args, agent, z, model, reward_fn):
    """One greedy episode per test environment; model None is the frozen FB actor. Returns
    the returns, shifts, gains and a0 fractions [eval_episodes] (see the module docstring)."""
    n = args.eval_episodes
    generator = torch.Generator().manual_seed(args.seed)
    envs = [make_env(i, args.dynamics) for i in range(n)]
    obs = np.stack([env.reset().observation["observations"] for env in envs])
    returns, shift, gain = np.zeros(n), np.zeros(n), np.zeros(n)
    a0 = np.zeros(n)
    for _ in range(STEPS):
        s = torch.as_tensor(obs, dtype=torch.float32, device=z.device)
        if model is None:
            with torch.no_grad():
                a = policy(agent, s, z.expand(n, -1))
        else:
            a, info = model.act(s, generator)
            shift += info["shift"].cpu().numpy()
            gain += info["gain"].cpu().numpy()
            a0 += info["at_fb"].cpu().numpy()
        a = a.cpu().numpy()
        for i, env in enumerate(envs):
            obs[i] = env.step(a[i]).observation["observations"]
            returns[i] += reward_fn(env.physics)
    for env in envs:
        env.close()
    a0 = np.ones(n) if model is None else a0 / STEPS
    return returns, shift / STEPS, gain / STEPS, a0


def frozen_rows(args, agent, z, task, reward_fn):
    """The rows of eval.csv of the frozen FB actor."""
    returns, *_ = evaluate(args, agent, z, None, reward_fn)
    return [{"variant": "fb", "task": task, "train_episodes": 0, "seed": i, "return": r,
             "shift": 0.0, "gain": 0.0, "a0_frac": 1.0} for i, r in enumerate(returns)]


def run(args, agent, z, scale, variant, task, reward_fn):
    """Trains one Q_adapt and tests it at args.eval_at. Returns the rows of eval.csv, the
    rows of train.csv and the trained QAdapt."""
    set_seed_everywhere(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    env = make_env(args.train_seed, args.dynamics)
    observation_length = env.observation_spec()["observations"].shape[0]
    action_length = env.action_spec().shape[0]
    model = QAdapt(agent, z, scale, variant, observation_length, action_length,
                   candidates=args.candidates, sigma=args.sigma, radius=args.radius, hidden=args.hidden,
                   omega=args.omega, alpha=args.alpha,
                   learning_rate=args.learning_rate, tau=args.tau, discount=args.discount)
    buffer = Buffer(args.episodes * STEPS, observation_length, action_length, z.device)
    discounts = args.discount ** np.arange(STEPS)
    eval_rows, train_rows = [], []

    def save(train_episodes):
        if args.save_every and train_episodes % args.save_every == 0:
            with no_clobber(Path(args.out_dir) / "q" / f"{variant}_{task}_ep{train_episodes}.pt") as partial:
                model.save(partial)

    def test(train_episodes):
        returns, shift, gain, a0 = evaluate(args, agent, z, model, reward_fn)
        eval_rows.extend({"variant": variant, "task": task, "train_episodes": train_episodes,
                          "seed": i, "return": returns[i], "shift": shift[i],
                          "gain": gain[i], "a0_frac": a0[i]}
                         for i in range(len(returns)))
        print(f"{variant} {task} after {train_episodes} episodes: "
              f"test return {returns.mean():.1f}, shift {shift.mean():.2f}, "
              f"gain {gain.mean():.2f}, a0 {a0.mean():.2f}", flush=True)

    save(0)
    if 0 in args.eval_at:
        test(0)
    for episode in range(args.episodes):
        obs = torch.as_tensor(env.reset().observation["observations"],
                              dtype=torch.float32, device=z.device)
        rewards, values, losses = np.zeros(STEPS), np.zeros(STEPS), []
        shift, gain, a0 = np.zeros(STEPS), np.zeros(STEPS), np.zeros(STEPS)
        for t in range(STEPS):
            a, info = model.act(obs.unsqueeze(0), generator, args.explore, sample=True)
            next_obs = torch.as_tensor(env.step(a[0].cpu().numpy()).observation["observations"],
                                       dtype=torch.float32, device=z.device)
            rewards[t], values[t] = reward_fn(env.physics), info["value"].item()
            shift[t], gain[t] = info["shift"].item(), info["gain"].item()
            a0[t] = info["at_fb"].item()
            buffer.add(obs, a[0], rewards[t], next_obs)
            obs = next_obs
            if buffer.size >= args.warmup:
                losses.append(model.update(buffer.sample(args.batch_size, generator), generator))
        actual = [(rewards[t:] * discounts[: STEPS - t]).sum() for t in range(VALUE_STEPS)]
        train_rows.append({
            "variant": variant, "task": task, "episode": episode, "return": rewards.sum(),
            "loss": np.mean(losses) if losses else float("nan"),
            "q_pred": values[:VALUE_STEPS].mean(), "q_actual": np.mean(actual),
            "shift": shift.mean(), "gain": gain.mean(), "a0_frac": a0.mean()})
        save(episode + 1)
        if episode + 1 in args.eval_at:
            test(episode + 1)
    env.close()
    return eval_rows, train_rows, model


def write_csv(path, fields, rows):
    with no_clobber(path) as partial, open(partial, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                             for k, v in row.items()})


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--variants", nargs="+", default=["direct", "residual"], choices=VARIANTS,
                        help="`search` is the greedy rule on the FB value alone, with no "
                             "training: use it with --episodes 0")
    parser.add_argument("--tasks", nargs="+", default=task_z.TASKS)
    parser.add_argument("--dynamics", type=str, default=None,
                        help="yaml of zero_shot_rl.env.dynamics; default is the stock environment")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=10,
                        help="test after 0, eval_every, 2 eval_every, ... episodes")
    parser.add_argument("--eval_at", nargs="+", type=int, default=None,
                        help="explicit episode counts to test after; replaces --eval_every")
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=0,
                        help="save the critics after every save_every episodes (0 is before "
                             "training); default 0 saves only the last")
    parser.add_argument("--train_only", action="store_true",
                        help="no tests and no frozen FB rows; test the saved critics with "
                             "zero_shot_rl.eval.value_correction_checkpoints")
    parser.add_argument("--train_seed", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--radius", type=float, default=None,
                        help="L2 bound of the candidate offsets from the FB action; default none")
    parser.add_argument("--explore", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--omega", type=float, default=1.0,
                        help="residual variant: weight of the prior policy log pi (omega' of "
                             "Li et al. 2023); large values keep the FB action")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="residual variant: temperature of the soft Q target and policy")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    args = parser.parse_args()
    if args.train_seed < args.eval_episodes:
        raise ValueError("--train_seed is one of the test seeds 0..eval_episodes-1")
    if args.train_only:
        args.eval_at = []
    elif args.eval_at is None:
        args.eval_at = eval_points(args.episodes, args.eval_every)
    args.dynamics = load_dynamics(args.dynamics) if args.dynamics else None
    out_dir = Path(args.out_dir)
    (out_dir / "q").mkdir(parents=True, exist_ok=True)

    device = task_z.device()
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    _, observations_z, rewards_z = task_z.inference_set(
        Namespace(seed=args.seed, discount=0.98, dataset_path=args.dataset_path,
                  dataset_transitions=args.dataset_transitions,
                  z_inference_steps=args.z_inference_steps),
        torch.device("cpu"),
    )
    observations_z = observations_z.to(device)
    reward_fns = task_z.reward_functions(args, device)

    eval_rows, train_rows = [], []
    for task in args.tasks:
        rewards = rewards_z[task].to(device)
        z = torch.as_tensor(agent.infer_z(observations_z, rewards),
                            dtype=torch.float32, device=device)
        scale = reward_scale(agent, observations_z, rewards)
        if not args.train_only:
            rows = frozen_rows(args, agent, z, task, reward_fns[task])
            eval_rows += rows
            print(f"fb {task}: test return {np.mean([r['return'] for r in rows]):.1f}, "
                  f"scale {scale:.4g}", flush=True)
        for variant in args.variants:
            start = time.time()
            rows, logs, model = run(args, agent, z, scale, variant, task, reward_fns[task])
            eval_rows += rows
            train_rows += logs
            with no_clobber(out_dir / "q" / f"{variant}_{task}.pt") as partial:
                model.save(partial)
            print(f"{variant} {task}: {time.time() - start:.0f} s", flush=True)
    if not args.train_only:
        write_csv(out_dir / "eval.csv", EVAL_FIELDS, eval_rows)
    write_csv(out_dir / "train.csv", TRAIN_FIELDS, train_rows)


if __name__ == "__main__":
    main()
