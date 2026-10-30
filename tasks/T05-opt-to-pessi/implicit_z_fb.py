"""Implicit-reward latents for a fixed standard FB agent (agents/fb).

Standard FB has a state-only B and attributes the reward of step t to the next state,
Q(s, a) = F(s, a, z)^T z_hat with z_hat = E_D[r(s') B(s')]. A pseudo-reward r_imp(s, a)
of a dataset transition (s, a, s') therefore shifts the latent by

    dz = E_D[r_imp(s, a) B(s')]

and for density-ratio rewards E_D[(rho / pi_beta)(a|s) B(s')] = E_{s~D, a~rho}[B(s'_a)],
where s'_a is the next state of action a at the dataset state s. s'_a comes from one
simulator step from the stored physics, as for the dataset action in z inference
(rewards.RewardFunctionConstructor.process_timesteps).

Families (strength = coefficient, matched by latent_solver angle search):

    z_r        dz = 0
    CQL        r_imp = -alpha (mu/pi_beta - 1), mu ~ exp(Q/tau) over --n_mu_actions
               uniform actions:  dz = -alpha (E_mu[B(s'_a)] - E_D[B(s')])
    ReBRAC     r_imp = -gamma beta p(s'), p(s') = ||pi_z(s') - a'_D||^2:
               dz = -gamma beta E_D[p(s') B(s')]
    E2E        r_imp = +c on the data:  dz = c E_D[B(s')]
    ReBRAC+E2E ReBRAC plus the E2E constant, the constant scaled to the same norm as the
               ReBRAC shift at z_r
    E2E-bi     bilevel E2E, r_imp = +beta pi/pi_beta:  dz = beta E_{s~D}[B(s'_{pi_z(s)})]
    AX         common axis: z_r rotated by +-angle towards d = normalise(t_E2E-bi - mean(
               t_CQL, t_ReBRAC)), t_f the unit tangent of family f at 10 degrees
               (negative angle = pessimistic side)
    C1, C2     random rotations and disjoint-subset noise controls

CQL, ReBRAC, ReBRAC+E2E and E2E-bi depend on z and are solved by the damped fixed point
of latent_solver.py.

Usage (repo root):
    PYTHONPATH=. python tasks/T05-opt-to-pessi/implicit_z_fb.py \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> --out <zs.npz>
"""

import math
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.inference_buffer import FBInferenceReplayBuffer
from latent_solver import LatentSolver, angle, normalized
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere

TASKS = ["stand", "walk", "run", "flip"]
ANGLES = [5, 10, 20, 35, 50, 70]  # target angles to z_r, degrees
DISCOUNT = 0.98  # agents/fb/config.yaml
AXIS_ANGLES = [-50, -35, -20, -10, 10, 20, 35, 50]
Z_FAMILIES = ("CQL", "ReBRAC", "ReBRAC+E2E", "E2E-bi")

W = {}  # simulator of a pool worker


def _init_worker():
    torch.set_num_threads(1)
    W["rc"] = RewardFunctionConstructor(
        domain_name="walker", task_names=["stand"], seed=0, device=torch.device("cpu")
    )


def _step(chunk):
    physics, actions = chunk
    obs, _ = W["rc"].process_timesteps(batch_physics=physics, batch_actions=actions)
    return obs.numpy()


class Simulator:
    """Next observations of (physics, action) pairs, one simulator step each."""

    def __init__(self, workers):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker)
        self.workers = workers

    def next_obs(self, physics, actions):
        chunks = [
            (p, a)
            for p, a in zip(np.array_split(physics, self.workers * 4),
                            np.array_split(actions, self.workers * 4))
            if len(p)
        ]
        return np.concatenate(self.pool.map(_step, chunks))


class FBImplicitZ(LatentSolver):
    def __init__(self, agent, data, sim, args):
        self.agent, self.args, self.sim = agent, args, sim
        self.d = agent._z_dimension
        self.gamma = DISCOUNT
        self.device = data["next_obs"].device
        self.B = self.backward(data["next_obs"])
        self.u_1 = self.B.mean(0)
        self.next_obs, self.next_actions = data["next_obs"], data["next_actions"]
        self.rewards = data["rewards"]

        # CQL candidates: n states, k uniform actions, their simulated next states
        n, k = args.n_mu_states, args.n_mu_actions
        self.mu_obs, self.mu_physics = data["obs"][:n], data["physics"][:n]
        gen = torch.Generator().manual_seed(args.seed)
        mu_actions = torch.rand((n, k, data["actions"].shape[1]), generator=gen) * 2 - 1
        self.mu_flat_obs = self.mu_obs.unsqueeze(1).expand(n, k, -1).reshape(n * k, -1)
        self.mu_flat_actions = mu_actions.reshape(n * k, -1).to(self.device)
        next_obs = sim.next_obs(np.repeat(self.mu_physics, k, axis=0),
                                mu_actions.reshape(n * k, -1).numpy())
        self.mu_B = self.backward(torch.as_tensor(next_obs, device=self.device))
        self.u_1_mu = self.B[:n].mean(0)  # E_D[B(s')] on the same states
        self.mu_shape = (n, k)
        self.bi_obs = data["obs"][: args.n_bi_states]
        self.bi_physics = data["physics"][: args.n_bi_states]

    def backward(self, obs):
        with torch.no_grad():
            return torch.cat([self.agent.FB.backward_representation(o) for o in obs.split(20000)])

    def policy(self, obs, z_hat):
        z = normalized(z_hat, self.d).expand(obs.shape[0], -1)
        with torch.no_grad():
            action, _ = self.agent.actor(obs, z, 0.2, sample=False)
        return action.clamp(-1.0, 1.0)

    def set_task(self, z_hat_r):
        u_reb = (self.rebrac_penalty(z_hat_r) * self.B).mean(0)
        self.e2e_k = float(self.gamma * u_reb.norm() / self.u_1.norm())

    def u_mu(self, z_hat):
        z = normalized(z_hat, self.d).expand(self.mu_flat_obs.shape[0], -1)
        with torch.no_grad():
            F1, F2 = self.agent.FB.forward_representation(
                observation=self.mu_flat_obs, action=self.mu_flat_actions, z=z
            )
        q = torch.minimum(F1 @ z_hat, F2 @ z_hat).reshape(self.mu_shape)
        w = torch.softmax(q / self.args.cql_tau, dim=1).reshape(-1, 1)
        return (w * self.mu_B).sum(0) / self.mu_shape[0]

    def u_pi(self, z_hat):
        actions = self.policy(self.bi_obs, z_hat).cpu().numpy()
        next_obs = self.sim.next_obs(self.bi_physics, actions)
        return self.backward(torch.as_tensor(next_obs, device=self.device)).mean(0)

    def rebrac_penalty(self, z_hat):
        a = self.policy(self.next_obs, z_hat)
        return ((a - self.next_actions) ** 2).sum(1, keepdim=True)

    def dz(self, family, strength, z_hat):
        if family == "CQL":
            return -strength * (self.u_mu(z_hat) - self.u_1_mu)
        if family in ("ReBRAC", "ReBRAC+E2E"):
            dz = -self.gamma * strength * (self.rebrac_penalty(z_hat) * self.B).mean(0)
            if family == "ReBRAC+E2E":
                dz = dz + strength * self.e2e_k * self.u_1
            return dz
        if family == "E2E":
            return strength * self.u_1
        if family == "E2E-bi":
            return strength * self.u_pi(z_hat)
        raise ValueError(family)

    def solve(self, family, strength, z_hat_r, init=None):
        if family == "E2E" or strength == 0:
            return z_hat_r + self.dz(family, strength, z_hat_r), True, 0
        return super().solve(family, strength, z_hat_r, init)


def tangent(z, z_r):
    """Unit direction of z away from z_r, orthogonal to z_r."""
    u = z_r / z_r.norm()
    t = z - torch.dot(z, u) * u
    return t / t.norm()


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--transitions", type=int, default=100000)
    parser.add_argument("--n_inference", type=int, default=100000)
    parser.add_argument("--n_mu_states", type=int, default=2048)
    parser.add_argument("--n_mu_actions", type=int, default=32)
    parser.add_argument("--n_bi_states", type=int, default=1024)
    parser.add_argument("--cql_tau", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--bisect_iters", type=int, default=10)
    parser.add_argument("--n_random", type=int, default=3)
    parser.add_argument("--n_subsets", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sim = Simulator(args.workers)  # spawned before CUDA is initialised
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(args.seed)
    agent = load_agent(Path(args.checkpoint), device)
    reward_constructor = RewardFunctionConstructor(
        domain_name="walker", task_names=args.tasks, seed=args.seed, device=device
    )
    buffer = FBInferenceReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=Path(args.dataset_path),
        discount=DISCOUNT,
        transitions=args.transitions,
        device=device,
    )

    # inference set: transitions with a next dataset action; s' and r(s') from the
    # simulator as in the FB workspace
    rng = np.random.default_rng(args.seed)
    valid = buffer.valid_indices.cpu().numpy()
    idx = rng.choice(valid, min(args.n_inference, len(valid)), replace=False)
    storage = buffer.storage
    next_obs, rewards = reward_constructor.process_timesteps(
        batch_physics=storage["physics"][idx], batch_actions=storage["actions"][idx].cpu().numpy()
    )
    t_idx = torch.as_tensor(idx, device=device)
    data = {
        "obs": storage["observations"][t_idx],
        "actions": storage["actions"][t_idx],
        "physics": storage["physics"][idx],
        "next_obs": next_obs,
        "next_actions": storage["next_actions"][t_idx],
        "rewards": rewards,
    }
    iz = FBImplicitZ(agent, data, sim, args)
    d = iz.d

    rows = []

    def add(task, family, z_hat, z_hat_r, strength=0.0, target=np.nan, converged=True, iters=0,
            multi=False):
        z = normalized(z_hat, d)
        rows.append({
            "task": task, "family": family, "strength": float(strength),
            "target_angle": float(target), "angle": angle(z, normalized(z_hat_r, d)),
            "converged": converged, "iters": iters, "multi": multi, "z": z.cpu().numpy(),
        })
        print(f"{task:5s} {family:10s} target {target:5.1f} angle {rows[-1]['angle']:6.2f} "
              f"strength {strength:+.4g} iters {iters} converged {converged} multi {multi}",
              flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    for task in args.tasks:
        z_hat_r = iz.z_hat_r(task)
        iz.set_task(z_hat_r)
        add(task, "z_r", z_hat_r, z_hat_r)
        # zero strength reproduces z_r exactly
        for family in Z_FAMILIES + ("E2E",):
            z0, _, _ = iz.solve(family, 0.0, z_hat_r)
            assert torch.equal(normalized(z0, d), normalized(z_hat_r, d)), family

        tangents = {}
        for family in Z_FAMILIES + ("E2E",):
            grid = iz.angle_grid(family, 1, z_hat_r)
            for target in ANGLES:
                match = iz.match_angle(family, 1, target, z_hat_r, grid)
                if match is None or isinstance(match, str):
                    reason = match or f"unreachable (max angle {max(g[1] for g in grid):.1f})"
                    print(f"{task:5s} {family:10s} target {target:5.1f} {reason}", flush=True)
                    continue
                s, init = match
                z_hat, converged, iters = iz.solve(family, s, z_hat_r)
                multi = abs(angle(normalized(z_hat, d), normalized(z_hat_r, d)) - target) > 1.0
                if multi:
                    z_hat, converged, _ = iz.solve(family, s, z_hat_r, init=init)
                    if abs(angle(normalized(z_hat, d), normalized(z_hat_r, d)) - target) > 1.0:
                        print(f"{task:5s} {family:10s} target {target:5.1f} discontinuous",
                              flush=True)
                        continue
                add(task, family, z_hat, z_hat_r, s, target, converged, iters, multi)
                if target <= 10:  # 10 degrees, or 5 if 10 is not reached
                    tangents[family] = tangent(normalized(z_hat, d), normalized(z_hat_r, d))

        # AX: common pessimistic-to-optimistic axis
        unit_r = z_hat_r / z_hat_r.norm()
        pess = (tangents["CQL"] + tangents["ReBRAC"]) / 2
        axis = tangents["E2E-bi"] - pess
        axis = axis - torch.dot(axis, unit_r) * unit_r
        axis = axis / axis.norm()
        for target in AXIS_ANGLES:
            t = math.radians(target)
            add(task, "AX", math.cos(t) * unit_r + math.sin(t) * axis, z_hat_r, 0.0, target)

        # C1: random orthogonal rotations
        for _ in range(args.n_random):
            v = torch.randn(d, generator=gen).to(device)
            v = v - torch.dot(v, unit_r) * unit_r
            v = v / v.norm()
            for target in ANGLES:
                t = math.radians(target)
                add(task, "C1", math.cos(t) * unit_r + math.sin(t) * v, z_hat_r, 0.0, target)

        # C2: disjoint subsets of the inference set
        perm = torch.as_tensor(rng.permutation(len(idx)), device=device)
        for part in torch.chunk(perm, args.n_subsets):
            add(task, "C2", iz.z_hat_r(task, part), z_hat_r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        task=np.array([r["task"] for r in rows]),
        family=np.array([r["family"] for r in rows]),
        strength=np.array([r["strength"] for r in rows]),
        target_angle=np.array([r["target_angle"] for r in rows]),
        angle=np.array([r["angle"] for r in rows]),
        converged=np.array([r["converged"] for r in rows]),
        iters=np.array([r["iters"] for r in rows]),
        multi=np.array([r["multi"] for r in rows]),
        z=np.stack([r["z"] for r in rows]),
        u_1=iz.u_1.cpu().numpy(),
        checkpoint=args.checkpoint,
    )
    print(f"wrote {len(rows)} latents to {out}")
    sim.pool.close()


if __name__ == "__main__":
    main()
