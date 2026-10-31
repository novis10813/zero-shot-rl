"""Stage 0 probe: trust signals of a fixed standard FB against Monte Carlo returns.

For every task, z_r and the CQL latent z_cql at --target_angle degrees are read from a
T05 latent NPZ (zero_shot_rl.latent.infer); the policies use these directions. Their
unnormalised forms z_hat_r = E_D[r B] and z_hat_cql are solved again with
zero_shot_rl.latent.infer.build_solver and give the norms only: z_hat = |z_hat_solved|
z / |z|, so F^T z_hat is in reward units. The inference set depends on the episode
order of the dataset file, so a re-solve on another copy of the dataset differs by a
few degrees; it must stay within --max_angle_error degrees of the stored latents. For every probed (s, a), with F_k = F_k(s, a, z_r):

    q_k  = F_k^T z_hat_r                  value of r, head k = 1, 2
    p_k  = F_k^T (z_hat_cql - z_hat_r)    value of the implicit CQL reward
    c_k  = F_k^T u_1                      value of the constant reward 1 (exact: 1/(1-gamma))
    nn   nearest training (s, a), per-dimension standardised
    G    Monte Carlo return: set the simulator to s, take a, then follow the actor at z_r
         with the training noise (std 0.2) for --horizon steps, gamma = 0.98

Sources (state, first action):

    data          training state, dataset action
    ood           training state, a ~ U(-1, 1)^A
    policy        training state, pi(s, z_r)
    pess          training state, pi(s, z_cql)
    visited       state of a deterministic pi(., z_r) episode from reset, pi(s, z_r)
    visited_pess  the same visited state, pi(s, z_cql)

Rows of policy/pess and visited/visited_pess share the column `state`.

Usage (repo root):
    PYTHONPATH=.:tasks/T06-latent-switching MUJOCO_GL=egl uv run python -m probe \
        --checkpoint <fb.pt> --zs <zs.npz> --dataset_path <dataset.npz> --out <probe.npz>
"""

import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from rewards import RewardFunctionConstructor
from zero_shot_rl.eval.env import make_env, policy
from zero_shot_rl.latent.archive import matched_rows, validate
from zero_shot_rl.latent.implicit import DISCOUNT, Simulator
from zero_shot_rl.latent.infer import TASKS, build_solver, solver_arguments
from zero_shot_rl.latent.solver import angle, normalized
from zero_shot_rl.provenance import sha256

TRAIN_STD = 0.2  # agents/fb/config.yaml std_dev_schedule; the actor clips at 0.3
EPISODE = 1000

W = {}  # state of a rollout worker


def _init_worker(checkpoint):
    torch.set_num_threads(1)
    W["agent"] = load_agent(Path(checkpoint), torch.device("cpu"))
    W["agent"].eval()
    W["rewards"] = RewardFunctionConstructor(
        domain_name="walker", task_names=TASKS, seed=0, device=torch.device("cpu")
    ).reward_functions


def set_physics(env, state):
    """reset_context runs mj_forward, so position-derived quantities match the state."""
    env.reset()
    with env.physics.reset_context():
        env.physics.set_state(state)


def _mc_returns(job):
    """Discounted return of each (physics state, first action), then the noisy actor."""
    task, z, physics, first_actions, horizon, seed = job
    torch.manual_seed(seed)
    agent, reward_fn = W["agent"], W["rewards"][task]
    envs = [make_env(seed + i) for i in range(len(physics))]
    returns = np.zeros(len(physics))
    obs = []
    for env, state, action in zip(envs, physics, first_actions):
        set_physics(env, state)
        obs.append(env.step(action).observation["observations"])
        returns[len(obs) - 1] = reward_fn(env.physics)
    obs = np.stack(obs).astype(np.float32)
    zs = torch.as_tensor(z, dtype=torch.float32).expand(len(envs), -1)
    for t in range(1, horizon):
        with torch.no_grad():
            actions, _ = agent.actor(torch.as_tensor(obs), zs, TRAIN_STD, sample=True)
        for i, env in enumerate(envs):
            obs[i] = env.step(actions[i].numpy()).observation["observations"]
            returns[i] += DISCOUNT**t * reward_fn(env.physics)
    for env in envs:
        env.close()
    return returns


def _visited(job):
    """Physics states every `every` steps of a deterministic episode at z, up to the
    last step that still has `horizon` steps left."""
    z, episode, horizon, every = job
    agent, env = W["agent"], make_env(episode)
    obs = env.reset().observation["observations"]
    zs = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)
    physics = []
    for t in range(1, EPISODE - horizon + 1):
        with torch.no_grad():
            action = policy(agent, torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), zs)
        obs = env.step(action.numpy()[0]).observation["observations"]
        if t % every == 0:
            physics.append(env.physics.get_state().copy())
    env.close()
    return physics


class Rollouts:
    """Simulator rollouts of the actor on CPU workers."""

    def __init__(self, checkpoint, workers):
        self.pool = mp.get_context("spawn").Pool(
            workers, initializer=_init_worker, initargs=(checkpoint,)
        )
        self.workers = workers

    def mc_returns(self, task, z, physics, actions, horizon, seed):
        bounds = np.linspace(0, len(physics), self.workers * 2 + 1).astype(int)
        jobs = [(task, z, physics[a:b], actions[a:b], horizon, seed + a)
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        return np.concatenate(self.pool.map(_mc_returns, jobs))

    def visited(self, z, episodes, horizon, every):
        jobs = [(z, e, horizon, every) for e in range(episodes)]
        return np.stack([p for ep in self.pool.map(_visited, jobs) for p in ep])


def observations(env, physics):
    """Observations of physics states, as the dataset stores them."""
    obs = []
    for state in physics:
        set_physics(env, state)
        values = env._task.get_observation(env.physics)  # pylint: disable=protected-access
        obs.append(np.concatenate([np.asarray(v).ravel() for v in values.values()]))
    return np.stack(obs).astype(np.float32)


def main():
    parser = ArgumentParser()
    solver_arguments(parser)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--target_angle", type=float, default=10.0)
    parser.add_argument("--max_angle_error", type=float, default=10.0)
    parser.add_argument("--n_states", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--visited_episodes", type=int, default=10)
    parser.add_argument("--visited_every", type=int, default=10)
    parser.add_argument("--rollout_workers", type=int, default=16)
    args = parser.parse_args()
    args.tasks = TASKS

    # spawned before CUDA is initialised
    sim = Simulator(args.workers)
    rollouts = Rollouts(args.checkpoint, args.rollout_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, buffer, _, _, iz = build_solver(args, sim, device)
    agent.eval()
    d = iz.d

    stored = np.load(args.zs)
    validate(stored, args.zs, TASKS, d)
    rows = dict(zip(TASKS, np.reshape(
        matched_rows(stored, TASKS, ["z_r", "CQL"], args.target_angle), (len(TASKS), 2)
    )))

    storage = buffer.storage
    data_obs, data_actions = storage["observations"], storage["actions"]
    sa = torch.cat([data_obs, data_actions], 1)
    sa_mean, sa_std = sa.mean(0), sa.std(0) + 1e-6
    sa_std_data = (sa - sa_mean) / sa_std

    def nn_distance(obs, actions):
        query = (torch.cat([obs, actions], 1) - sa_mean) / sa_std
        best = torch.full((len(query),), float("inf"), device=device)
        for start in range(0, len(sa_std_data), 20000):
            d_sa = torch.cdist(query, sa_std_data[start : start + 20000])
            best = torch.minimum(best, d_sa.min(1).values)
        return best

    rng = np.random.default_rng([args.seed, 6])
    probe_idx = rng.choice(len(data_obs), args.n_states, replace=False)
    t_idx = torch.as_tensor(probe_idx, device=device)
    probe_obs, probe_physics = data_obs[t_idx], storage["physics"][probe_idx]
    ood = torch.as_tensor(
        rng.uniform(-1, 1, size=(args.n_states, data_actions.shape[1])), dtype=torch.float32,
        device=device,
    )
    env = make_env(0)
    obs_err = np.abs(observations(env, probe_physics[:20]) - probe_obs[:20].cpu().numpy()).max()
    print(f"max |obs(physics) - dataset obs| on 20 probe states: {obs_err:.2e}", flush=True)

    def act(obs, z):
        with torch.no_grad():
            return policy(agent, obs, z.expand(len(obs), -1))

    out = {key: [] for key in ("task", "source", "state", "q1", "q2", "p1", "p2", "c1", "c2",
                               "nn", "G")}
    latents = {"task": [], "z_hat_r": [], "z_hat_cql": [], "angle_r": [], "angle_cql": []}
    for task_i, task in enumerate(TASKS):
        row_r, row_cql = rows[task]
        solved_r = iz.z_hat_r(task)
        solved_cql, converged, _ = iz.solve("CQL", float(stored["strength"][row_cql]), solved_r)
        z_r = normalized(torch.as_tensor(stored["z"][row_r], device=device), d)
        z_cql = normalized(torch.as_tensor(stored["z"][row_cql], device=device), d)
        errors = (angle(solved_r, z_r), angle(solved_cql, z_cql))
        print(f"{task}: angle of re-solved to stored z_r {errors[0]:.3f}, z_cql "
              f"{errors[1]:.3f}; stored z_cql to z_r {angle(z_cql, z_r):.2f}, "
              f"converged {converged}", flush=True)
        if not converged or max(errors) > args.max_angle_error:
            raise ValueError(f"{task}: re-solved latents differ from {args.zs}")
        z_hat_r = z_r * solved_r.norm() / z_r.norm()
        z_hat_cql = z_cql * solved_cql.norm() / z_cql.norm()
        for key, value in zip(latents, (task, z_hat_r, z_hat_cql, *errors)):
            latents[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)

        v_physics = rollouts.visited(z_r.cpu().numpy(), args.visited_episodes, args.horizon,
                                     args.visited_every)
        v_obs = torch.as_tensor(observations(env, v_physics), device=device)
        sources = {
            "data": (probe_obs, probe_physics, data_actions[t_idx]),
            "ood": (probe_obs, probe_physics, ood),
            "policy": (probe_obs, probe_physics, act(probe_obs, z_r)),
            "pess": (probe_obs, probe_physics, act(probe_obs, z_cql)),
            "visited": (v_obs, v_physics, act(v_obs, z_r)),
            "visited_pess": (v_obs, v_physics, act(v_obs, z_cql)),
        }
        for source_i, (source, (obs, physics, actions)) in enumerate(sources.items()):
            with torch.no_grad():
                F1, F2 = agent.FB.forward_representation(
                    observation=obs, action=actions, z=z_r.expand(len(obs), -1)
                )
            values = {}
            for k, F in (("1", F1), ("2", F2)):
                values["q" + k] = F @ z_hat_r
                values["p" + k] = F @ (z_hat_cql - z_hat_r)
                values["c" + k] = F @ iz.u_1
            values["nn"] = nn_distance(obs, actions)
            values["G"] = rollouts.mc_returns(
                task, z_r.cpu().numpy(), physics, actions.cpu().numpy(), args.horizon,
                seed=args.seed + 100000 * (task_i * len(sources) + source_i),
            )
            n = len(obs)
            out["task"].append(np.full(n, task))
            out["source"].append(np.full(n, source))
            out["state"].append(np.arange(n))
            for key, value in values.items():
                out[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)
            q = np.minimum(out["q1"][-1], out["q2"][-1])
            print(f"{task} {source:12s} n {n:4d} q_hat {q.mean():7.2f} "
                  f"G {out['G'][-1].mean():7.2f}", flush=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{key: np.concatenate(value) for key, value in out.items()},
        **{"latent_" + key: np.stack(value) for key, value in latents.items()},
        u_1=iz.u_1.cpu().numpy(),
        max_obs_err=obs_err,
        checkpoint=args.checkpoint,
        zs=args.zs,
        dataset_sha256=sha256(Path(args.dataset_path)),
    )
    print(f"wrote {path}")
    sim.pool.close()
    rollouts.pool.close()


if __name__ == "__main__":
    main()
