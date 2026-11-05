"""
Reward of the action each checkpoint takes at the states of an episode, against the
distance of that action to the dataset behaviour.

Every checkpoint runs one deterministic 1000-step episode per task (z inferred as in
zero_shot_rl.latent.task_z) and keeps the state (observation and physics) every --stride
steps (the "pool" of states). All episodes use one environment seeded with --env_seed and
start from its next reset, so their start states differ. At each kept state every checkpoint
(the "actor") acts with its own policy, scored with the joint and conditional distances of
zero_shot_rl.latent.behavior. The same action is then executed from that exact physics state
in the simulator and the task reward is summed over --horizon steps (the first step uses the
action above, later steps the actor's own policy). --horizon 1 is the reward r(s') after
taking a at s.

Writes one CSV row per (pool, actor, task, step).

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.behavior_reward \
        --checkpoints FB a.pt b.pt c.pt --checkpoints VC-FB d.pt e.pt f.pt \
        --dataset_path <dataset.npz> --out rewards.csv
"""

from argparse import ArgumentParser

import pandas as pd
import torch

from zero_shot_rl.env.walker import act, make_env, policy, run_episodes, set_physics
from zero_shot_rl.latent import behavior as behavior_module
from zero_shot_rl.latent import task_z
from zero_shot_rl.latent.behavior import METRICS, Behavior
from zero_shot_rl.latent.task_z import TASKS


def reward_from(env, physics, obs, agent, z, reward_fn, horizon):
    """Reward summed over `horizon` steps starting by acting at `obs` from `physics`."""
    set_physics(env, physics)
    total = 0.0
    for _ in range(horizon):
        obs = env.step(act(agent, obs, z)).observation["observations"]
        total += reward_fn(env.physics)
    return total


def main():
    parser = ArgumentParser()
    task_z.add_arguments(parser)
    behavior_module.add_arguments(parser)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    device = task_z.device()
    buffer, observations_z, rewards_z = task_z.inference_set(args, device)
    behavior = Behavior.from_buffer(buffer, args.lam, args.k, device)
    reward_fns = task_z.reward_functions(args, device)
    agents = {
        f"{label} seed {index}": (agent, zs)
        for label, index, agent, zs in task_z.load_checkpoints(args, observations_z, rewards_z, device)
    }

    env = make_env(args.env_seed)
    rows = []
    for pool, (pool_agent, pool_zs) in agents.items():
        for task in TASKS:
            run = run_episodes(pool_agent, pool_zs[task], [env], stride=args.stride, physics=True)
            observations, physics = run["observations"][0], run["physics"][0]
            s = torch.as_tensor(observations, dtype=torch.float32, device=device)
            state_term = behavior.state_term(s)
            for actor, (agent, zs) in agents.items():
                z = zs[task]
                with torch.no_grad():
                    a = policy(agent, s, z.expand(len(s), -1))
                joint, conditional = behavior.distances(state_term, a)
                for i in range(len(s)):
                    rows.append({
                        "pool": pool, "actor": actor, "task": task, "step": i * args.stride,
                        "reward": reward_from(env, physics[i], observations[i], agent, z,
                                              reward_fns[task], args.horizon),
                        "joint": joint[i], "conditional": conditional[i],
                    })
            print(f"{pool} {task} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    own = df[df["pool"] == df["actor"]]
    print(own.groupby(["task", "actor"])[METRICS + ["reward"]].mean().round(3))


if __name__ == "__main__":
    main()
