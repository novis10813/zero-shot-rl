"""Tests the critics saved by zero_shot_rl.eval.value_correction --save_every.

For each q/<variant>_<task>_ep<n>.pt of --q_dir (optionally only the counts n of --at) the
greedy rule of the saved QAdapt runs one episode in each of the --eval_episodes environments
seeded 0..eval_episodes-1, with the same candidate noise at every count, as in
zero_shot_rl.eval.value_correction. With --frozen the frozen FB actor is tested too, at the z
of each task. Tests of different counts are independent, so they can be split over jobs and
hosts with --at.

--out_dir gets eval.csv with the columns of zero_shot_rl.eval.value_correction (variant `fb`
has train_episodes 0).

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.value_correction_checkpoints \
        --checkpoint <fb.pt> --q_dir <out_dir>/q --dynamics <shift.yaml> --out_dir out/ \
        --eval_episodes 5 --at 0 10 20
"""

import re
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from agents.base import load_agent
from agents.fb.agent import FB
from zero_shot_rl.env.dynamics import load_dynamics
from zero_shot_rl.env.walker import make_env
from zero_shot_rl.eval.value_correction import EVAL_FIELDS, evaluate, frozen_rows, write_csv
from zero_shot_rl.latent import task_z
from zero_shot_rl.online.value_correction import QAdapt

NAME = re.compile(r"(?P<variant>[a-z]+)_(?P<task>[a-z]+)_ep(?P<episodes>\d+)\.pt")


def find(q_dir, at=None):
    """[(variant, task, episodes, path)] of the saved critics, sorted, filtered to the counts
    `at` if given."""
    found = []
    for path in Path(q_dir).glob("*_ep*.pt"):
        match = NAME.fullmatch(path.name)
        if match and (at is None or int(match["episodes"]) in at):
            found.append((match["variant"], match["task"], int(match["episodes"]), path))
    return sorted(found, key=lambda f: (f[0], f[1], f[2]))


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="the FB checkpoint")
    parser.add_argument("--q_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dynamics", type=str, default=None)
    parser.add_argument("--at", nargs="+", type=int, default=None,
                        help="only these episode counts; default all")
    parser.add_argument("--frozen", action="store_true", help="also test the frozen FB actor")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.dynamics = load_dynamics(args.dynamics) if args.dynamics else None
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = task_z.device()
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError("Only standard FB is supported")
    agent.eval()
    reward_fns = task_z.reward_functions(args, device)
    probe = make_env(0, args.dynamics)
    observation_length = probe.observation_spec()["observations"].shape[0]
    action_length = probe.action_spec().shape[0]
    probe.close()

    rows, frozen_done = [], set()
    for variant, task, episodes, path in find(args.q_dir, args.at):
        start = time.time()
        model = QAdapt.load(path, agent, observation_length, action_length)
        if args.frozen and task not in frozen_done:
            rows += frozen_rows(args, agent, model.z, task, reward_fns[task])
            frozen_done.add(task)
        returns, shift, gain, a0 = evaluate(args, agent, model.z, model, reward_fns[task])
        rows += [{"variant": variant, "task": task, "train_episodes": episodes, "seed": i,
                  "return": returns[i], "shift": shift[i], "gain": gain[i], "a0_frac": a0[i]}
                 for i in range(len(returns))]
        print(f"{variant} {task} after {episodes} episodes: test return {returns.mean():.1f}, "
              f"shift {np.mean(shift):.2f}, gain {np.mean(gain):.2f}, a0 {np.mean(a0):.2f} "
              f"({time.time() - start:.0f} s)", flush=True)
    write_csv(Path(args.out_dir) / "eval.csv", EVAL_FIELDS, rows)


if __name__ == "__main__":
    main()
