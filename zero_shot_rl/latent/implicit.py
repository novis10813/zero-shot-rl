"""Implicit-reward latent shifts for a fixed standard FB agent (agents/fb).

Standard FB has a state-only B and attributes the reward of step t to the next state,
Q(s, a) = F(s, a, z)^T z_hat with z_hat = E_D[r(s') B(s')]. A pseudo-reward r_imp(s, a)
of a dataset transition (s, a, s') therefore shifts the latent by

    dz = E_D[r_imp(s, a) B(s')]

and for density-ratio rewards E_D[(rho / pi_beta)(a|s) B(s')] = E_{s~D, a~rho}[B(s'_a)],
where s'_a is the next state of action a at the dataset state s. s'_a comes from one
simulator step from the stored physics, as for the dataset action in z inference
(rewards.RewardFunctionConstructor.process_timesteps).

Families (strength = coefficient, matched by the solver angle search):

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
of zero_shot_rl.latent.solver. The entry module is zero_shot_rl.latent.infer.
"""

import multiprocessing as mp

import numpy as np
import torch

from rewards import RewardFunctionConstructor
from zero_shot_rl.latent.solver import LatentSolver, normalized

DISCOUNT = 0.98  # agents/fb/config.yaml

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
