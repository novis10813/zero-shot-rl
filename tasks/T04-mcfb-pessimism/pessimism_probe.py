"""Probe how pessimistic FB, VC-FB and MC-FB value estimates are on walker rnd-100k.

Calibration (reward units). For each eval task r with inferred z_r, the model's value
    Q_hat(s, a) = c_r * min(F1, F2)(s, a, z_r)^T z_r,    c_r = |E_D[r B]| / sqrt(d)
undoes the sqrt(d) normalisation of z_r, so Q_hat estimates sum_t gamma^t r_{t+1}.
It is compared with a Monte Carlo return G(s, a): set the simulator to s, take a,
then follow the actor at z_r with the training target noise (std 0.2, clip 0.3) for
--horizon steps. Four (s, a) sources:
    data     (s, a) from the 100k training set
    ood      s from the training set, a ~ U(-1, 1)^A (the penalty's OOD actions)
    policy   s from the training set, a = pi(s, z_r)
    visited  s visited by the deterministic eval policy from the env start, a = pi(s, z_r)

Action gap (raw F^T z units, no ground truth). For z ~ prior, z = B(s+) and z = z_r,
the mean of Q(s, a_data), Q(s, a_uniform) and Q(s, pi(s, z)) over training states.

Usage (repo root):
    PYTHONPATH=. python tasks/T04-mcfb-pessimism/pessimism_probe.py \
        --dataset_path <dataset.npz> --out <out.json> --checkpoints mcfb_s0=<path.pt> ...
"""

import json
import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

import custom_dmc_tasks as cdmc
from agents.base import load_agent
from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere

TASKS = ["stand", "walk", "run", "flip"]
TRAIN_STD, TRAIN_CLIP = 0.2, 0.3  # agents/*/config.yaml std_dev_schedule, std_dev_clip

parser = ArgumentParser()
parser.add_argument("--checkpoints", nargs="+", required=True, help="label=path")
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--discount", type=float, default=0.98)
parser.add_argument("--n_states", type=int, default=1000)
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--visited_episodes", type=int, default=10)
parser.add_argument("--visited_every", type=int, default=50)
parser.add_argument("--gap_states", type=int, default=4096)
parser.add_argument("--gap_actions", type=int, default=32)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed_everywhere(args.seed)

reward_constructor = RewardFunctionConstructor(
    domain_name="walker", task_names=TASKS, seed=args.seed, device=device
)
replay_buffer = FBReplayBuffer(
    reward_constructor=reward_constructor,
    dataset_path=Path(args.dataset_path),
    transitions=100000,
    relabel=False,
    task=None,
    device=device,
    discount=args.discount,
)
storage = replay_buffer.storage
n_data = storage["observations"].shape[0]
obs_mean = storage["observations"].mean(0)
obs_std = storage["observations"].std(0) + 1e-6

# shared across checkpoints: z inference samples, probe states, OOD actions, gap z's
z_inf_obs, z_inf_rewards = replay_buffer.sample_task_inference_transitions(10000)
rng = np.random.default_rng(args.seed)
probe_idx = rng.choice(n_data, args.n_states, replace=False)
action_length = storage["actions"].shape[1]
ood_actions = rng.uniform(-1, 1, size=(args.n_states, action_length)).astype(np.float32)
gap_idx = rng.choice(n_data, args.gap_states, replace=False)
goal_idx = rng.choice(n_data, args.gap_states, replace=False)
gap_uniform = torch.empty(
    args.gap_actions, args.gap_states, action_length, device=device
).uniform_(-1, 1, generator=torch.Generator(device).manual_seed(args.seed))
prior_gen = torch.Generator(device).manual_seed(args.seed + 1)


def make_env(seed):
    return cdmc.make(
        domain="walker",
        task="flip",
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": seed},
    )


envs = []
reward_fns = reward_constructor.reward_functions


def get_envs(n):
    while len(envs) < n:
        envs.append(make_env(args.seed + len(envs)))
    return envs[:n]


def set_physics(env, state):
    """reset_context runs mj_forward, so positions-derived quantities match the state."""
    env.reset()
    with env.physics.reset_context():
        env.physics.set_state(state)


def nn_distance(obs, exclude_self_idx=None):
    """Distance to the nearest training state, in per-dimension standardised units."""
    query = (obs - obs_mean) / obs_std
    data = (storage["observations"] - obs_mean) / obs_std
    best = torch.full((query.shape[0],), float("inf"), device=device)
    for start in range(0, n_data, 20000):
        d = torch.cdist(query, data[start : start + 20000])
        if exclude_self_idx is not None:
            own = (exclude_self_idx >= start) & (exclude_self_idx < start + 20000)
            rows = torch.nonzero(own).squeeze(1)
            d[rows, exclude_self_idx[rows] - start] = float("inf")
        best = torch.minimum(best, d.min(1).values)
    return best


def q_values(agent, obs, actions, z):
    """min(F1, F2)^T z, plus each head."""
    F1, F2 = agent.FB.forward_representation(observation=obs, z=z, action=actions)
    q1 = (F1 * z).sum(-1)
    q2 = (F2 * z).sum(-1)
    return torch.minimum(q1, q2), q1, q2


def policy_action(agent, obs, z, noisy):
    if noisy:
        action, _ = agent.actor(obs, z, TRAIN_STD, sample=True)
    else:
        action, _ = agent.actor(obs, z, 0.05, sample=False)  # std_dev_eval, mean action
    return action


def mc_returns(agent, z, task, physics_states, first_actions):
    """Set each env to a physics state, take the first action, then follow the noisy
    actor at z. Returns sum_{t < horizon} gamma^t r_t, where r_t follows action t."""
    n = len(physics_states)
    get_envs(n)
    returns = np.zeros(n)
    obs = np.zeros((n, storage["observations"].shape[1]), dtype=np.float32)
    for i in range(n):
        set_physics(envs[i], physics_states[i])
        ts = envs[i].step(first_actions[i])
        obs[i] = ts.observation["observations"]
        returns[i] += reward_fns[task](envs[i].physics)
    zs = z.unsqueeze(0).repeat(n, 1)
    for t in range(1, args.horizon):
        with torch.no_grad():
            acts = policy_action(
                agent, torch.as_tensor(obs, device=device), zs, noisy=True
            ).cpu().numpy()
        for i in range(n):
            ts = envs[i].step(acts[i])
            obs[i] = ts.observation["observations"]
            returns[i] += args.discount**t * reward_fns[task](envs[i].physics)
    return returns


def visited_states(agent, z):
    """Physics states and observations along deterministic eval rollouts from reset."""
    physics, observations = [], []
    zs = z.unsqueeze(0).repeat(args.visited_episodes, 1)
    get_envs(args.visited_episodes)
    obs = np.stack(
        [envs[i].reset().observation["observations"] for i in range(args.visited_episodes)]
    ).astype(np.float32)
    for t in range(1, 1000 - args.horizon + 1):
        with torch.no_grad():
            acts = policy_action(
                agent, torch.as_tensor(obs, device=device), zs, noisy=False
            ).cpu().numpy()
        for i in range(args.visited_episodes):
            obs[i] = envs[i].step(acts[i]).observation["observations"]
            if t % args.visited_every == 0:
                physics.append(envs[i].physics.get_state().copy())
                observations.append(obs[i].copy())
    return physics, np.stack(observations)


def rounded(x):
    x = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return [round(float(v), 4) for v in x]


def calibration(agent, z, scale, task):
    probe_obs = storage["observations"][probe_idx]
    probe_physics = storage["physics"][probe_idx]
    zs = z.unsqueeze(0).repeat(args.n_states, 1)
    with torch.no_grad():
        pi_actions = policy_action(agent, probe_obs, zs, noisy=False)
    sources = {
        "data": (probe_obs, probe_physics, storage["actions"][probe_idx]),
        "ood": (probe_obs, probe_physics, torch.as_tensor(ood_actions, device=device)),
        "policy": (probe_obs, probe_physics, pi_actions),
    }
    v_physics, v_obs = visited_states(agent, z)
    v_obs = torch.as_tensor(v_obs, device=device)
    with torch.no_grad():
        v_actions = policy_action(agent, v_obs, z.unsqueeze(0).repeat(len(v_obs), 1), False)
    sources["visited"] = (v_obs, np.stack(v_physics), v_actions)

    result = {}
    for name, (obs, physics, actions) in sources.items():
        zs = z.unsqueeze(0).repeat(len(obs), 1)
        with torch.no_grad():
            q, q1, q2 = q_values(agent, obs, actions, zs)
        g = mc_returns(agent, z, task, physics, actions.cpu().numpy())
        result[name] = {
            "q_hat": rounded(scale * q),
            "q1_hat_mean": (scale * q1).mean().item(),
            "q2_hat_mean": (scale * q2).mean().item(),
            "mc_return": rounded(g),
        }
    result["visited"]["nn_distance"] = rounded(nn_distance(v_obs))
    return result


def action_gaps(agent, task_zs):
    obs = storage["observations"][gap_idx]
    data_actions = storage["actions"][gap_idx]
    with torch.no_grad():
        prior = torch.randn(args.gap_states, agent._z_dimension, device=device, generator=prior_gen)
        goal = agent.FB.backward_representation(storage["next_observations"][goal_idx])
    d = agent._z_dimension
    z_sets = {
        "prior": math.sqrt(d) * torch.nn.functional.normalize(prior, dim=1),
        "backward": math.sqrt(d) * torch.nn.functional.normalize(goal, dim=1),
        **{f"task_{t}": z.unsqueeze(0).repeat(args.gap_states, 1) for t, z in task_zs.items()},
    }
    gaps = {}
    for name, zs in z_sets.items():
        with torch.no_grad():
            q_data = q_values(agent, obs, data_actions, zs)[0]
            q_pi = q_values(agent, obs, policy_action(agent, obs, zs, noisy=False), zs)[0]
            q_unif = torch.stack(
                [q_values(agent, obs, gap_uniform[k], zs)[0] for k in range(args.gap_actions)]
            )
        gaps[name] = {
            "q_data": q_data.mean().item(),
            "q_uniform": q_unif.mean().item(),
            "q_policy": q_pi.mean().item(),
            "frac_data_above_uniform": (q_data.unsqueeze(0) > q_unif).float().mean().item(),
            "q_data_std": q_data.std().item(),
        }
    return gaps


# sanity: dataset observation t is the observation of dataset physics t
check_env = get_envs(1)[0]
max_obs_err = 0.0
for i in probe_idx[:20]:
    set_physics(check_env, storage["physics"][i])
    obs = check_env._task.get_observation(check_env.physics)  # pylint: disable=protected-access
    flat = np.concatenate([np.asarray(v).ravel() for v in obs.values()])
    max_obs_err = max(max_obs_err, float(np.abs(flat - storage["observations"][i].cpu().numpy()).max()))
print(f"max |obs(physics) - dataset obs| on 20 probe states: {max_obs_err:.2e}")

output = {
    "args": vars(args),
    "max_obs_err": max_obs_err,
    "data_nn_distance": rounded(nn_distance(
        storage["observations"][probe_idx], torch.as_tensor(probe_idx, device=device)
    )),
    "models": {},
}
for spec in args.checkpoints:
    label, path = spec.split("=", 1)
    agent = load_agent(Path(path), device)
    agent.eval()
    with torch.no_grad():
        B = agent.FB.backward_representation(z_inf_obs)
    task_zs, scales = {}, {}
    for task in TASKS:
        z_raw = (z_inf_rewards[task] * B).mean(0)
        task_zs[task] = math.sqrt(agent._z_dimension) * z_raw / z_raw.norm()
        scales[task] = (z_raw.norm() / math.sqrt(agent._z_dimension)).item()
    model = {"checkpoint": path, "scale": scales, "calibration": {}}
    for task in TASKS:
        model["calibration"][task] = calibration(agent, task_zs[task], scales[task], task)
        cal = model["calibration"][task]
        print(
            f"{label} {task}: "
            + ", ".join(
                f"{k} q_hat={np.mean(v['q_hat']):.2f} G={np.mean(v['mc_return']):.2f}"
                for k, v in cal.items()
            ),
            flush=True,
        )
    model["action_gaps"] = action_gaps(agent, task_zs)
    print(f"{label} gaps: {json.dumps(model['action_gaps'])}", flush=True)
    output["models"][label] = model

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    json.dump(output, f)
print(f"wrote {args.out}")
