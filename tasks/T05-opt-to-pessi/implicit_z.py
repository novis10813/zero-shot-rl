"""Task latents of a fixed one-step FB for rewards with an implicit offline-RL term.

For a reward r' = r + r_imp, the one-step FB latent is the linear
    z_hat(r') = E_D[r'(s, a) B(s, a)] = z_hat_r + dz,      z = sqrt(d) normalize(z_hat),
so every family below only changes dz; F, B and the actor stay fixed.

    G0   linear z_hat_r = E_D[r B]                                  (neutral)
    G0p  the paper's softmax(10 r)-weighted latent (agent.infer_z)
    G1   CQL(H):  r_imp = -alpha (mu/pi_beta - 1),  dz = -alpha (u_mu(z) - u_1)
                  mu(a|s) ~ exp(Q(s, a) / tau), Q = F^T z_hat / (1 - gamma); alpha < 0 is G1neg
    G3   ReBRAC:  r_imp = -gamma beta ||pi_z(s') - a'_data||^2,  dz = -gamma beta u_reb(z)
                  G3c uses the centred penalty (its dataset mean removed)
    G5   E2E practical: r_imp = c, dz = c u_1; G5neg is c < 0
    G6   ReBRAC + E2E practical: dz = beta (-gamma u_reb(z) + k u_1), k = gamma |u_reb(z_r)| / |u_1|
         so both terms have the same size at z_r
    C1   random directions orthogonal to z_hat_r at the same angles (control)
    C2   z_hat_r recomputed on disjoint subsets of the inference set (noise floor)

u_1 = E_D[B(s, a)], u_mu = E_{s~D, a~mu}[B(s, a)] (64 uniform actions per state,
self-normalised weights), u_reb = E_D[||pi_z(s') - a'||^2 B(s, a)].
G1 and G3 depend on z through mu and pi_z, so z_hat solves a damped fixed point
    z_hat <- (1 - eta) z_hat + eta (z_hat_r + dz(z_hat)).
Strengths are chosen so the angle between z and z_r hits each target angle.

Usage (repo root):
    PYTHONPATH=. python tasks/T05-opt-to-pessi/implicit_z.py --checkpoint <best.pt> \
        --dataset_path <dataset.npz> --out tasks/T05-opt-to-pessi/data/zs_<job>.npz
"""

import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.onestep_fb.replay_buffer import OneStepFBReplayBuffer
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere

TASKS = ["stand", "walk", "run", "flip"]
ANGLES = [5, 10, 20, 35, 50, 70]  # target angles to z_r, degrees
E2E_ANGLES = [5, 20, 50]  # G5 per sign
G6_ANGLES = [5, 10, 20, 35]


def normalized(z_hat, d):
    return math.sqrt(d) * z_hat / z_hat.norm()


def angle(a, b):
    cos = torch.dot(a, b) / (a.norm() * b.norm())
    return math.degrees(math.acos(float(cos.clamp(-1.0, 1.0))))


class ImplicitZ:
    """Holds the inference set and the B-statistics shared by all families."""

    def __init__(self, agent, data, args):
        self.agent, self.args = agent, args
        self.d = agent._z_dimension
        self.gamma = agent._discount
        with torch.no_grad():
            self.B = agent.backward(data["obs"], data["actions"])
            self.u_1 = self.B.mean(0)
        self.next_obs, self.next_actions = data["next_obs"], data["next_actions"]
        self.rewards = data["rewards"]

        # u_mu: a subset of states, each with uniform candidate actions
        n, k = args.n_mu_states, args.n_mu_actions
        mu_obs = data["obs"][:n]
        gen = torch.Generator().manual_seed(args.seed)
        mu_actions = torch.rand((n, k, data["actions"].shape[1]), generator=gen) * 2 - 1
        mu_actions = mu_actions.to(mu_obs.device)
        flat_obs = mu_obs.unsqueeze(1).expand(n, k, -1).reshape(n * k, -1)
        flat_actions = mu_actions.reshape(n * k, -1)
        with torch.no_grad():
            self.mu_F = agent.forward_representation(flat_obs, flat_actions).mean(0)
            self.mu_B = agent.backward(flat_obs, flat_actions)
        self.mu_shape = (n, k)

    def set_task(self, z_hat_r):
        u_reb = (self.rebrac_penalty(z_hat_r) * self.B).mean(0)
        self.g6_k = float(self.gamma * u_reb.norm() / self.u_1.norm())

    def z_hat_r(self, task, idx=None):
        r = self.rewards[task] if idx is None else self.rewards[task][idx]
        B = self.B if idx is None else self.B[idx]
        return (r * B).mean(0)

    def u_mu(self, z_hat):
        q = (self.mu_F @ z_hat / (1 - self.gamma)).reshape(self.mu_shape)
        w = torch.softmax(q / self.args.cql_tau, dim=1).reshape(-1, 1)
        return (w * self.mu_B).sum(0) / self.mu_shape[0]

    def rebrac_penalty(self, z_hat):
        z = normalized(z_hat, self.d).expand(self.next_obs.shape[0], -1)
        with torch.no_grad():
            a = self.agent.policy(self.next_obs, z)
        return ((a - self.next_actions) ** 2).sum(1, keepdim=True)

    def dz(self, family, strength, z_hat):
        if family == "G1":
            return -strength * (self.u_mu(z_hat) - self.u_1)
        if family in ("G3", "G3c", "G6"):
            pen = self.rebrac_penalty(z_hat)
            if family == "G3c":
                pen = pen - pen.mean()
            dz = -self.gamma * strength * (pen * self.B).mean(0)
            if family == "G6":
                dz = dz + strength * self.g6_k * self.u_1
            return dz
        if family == "G5":
            return strength * self.u_1
        raise ValueError(family)

    def solve(self, family, strength, z_hat_r, init=None):
        """Damped fixed point for z-dependent families (started from `init`, default
        z_hat_r); one step otherwise."""
        if family == "G5" or strength == 0:
            return z_hat_r + self.dz(family, strength, z_hat_r), True, 0
        z_hat = (z_hat_r if init is None else init).clone()
        for it in range(1, self.args.max_iters + 1):
            target = z_hat_r + self.dz(family, strength, z_hat)
            new = (1 - self.args.eta) * z_hat + self.args.eta * target
            residual = float((new - z_hat).norm() / z_hat_r.norm())
            z_hat = new
            if residual < self.args.tol:
                return z_hat, True, it
        return z_hat, False, self.args.max_iters

    def angle_grid(self, family, sign, z_hat_r):
        """Angle to z_r on a log grid of strengths, walked upwards with warm starts;
        stops at the first strength whose fixed point does not converge."""
        z_r = normalized(z_hat_r, self.d)
        scale = float(z_hat_r.norm() / self.dz(family, 1.0, z_hat_r).norm())
        grid, z_hat = [(0.0, 0.0, z_hat_r)], z_hat_r
        for s in scale * np.logspace(-3, 3, 25):
            z_hat, converged, _ = self.solve(family, sign * s, z_hat_r, init=z_hat)
            if not converged:
                break
            grid.append((s, angle(normalized(z_hat, self.d), z_r), z_hat))
        return grid

    def match_angle(self, family, sign, target, z_hat_r, grid):
        """Strength (with the given sign) whose solved z is `target` degrees from z_r:
        bisection inside the first grid interval that crosses it. None if unreachable."""
        z_r = normalized(z_hat_r, self.d)
        for (lo, th_lo, z_lo), (hi, th_hi, _) in zip(grid, grid[1:]):
            if th_lo < target <= th_hi:
                break
        else:
            return None
        for _ in range(self.args.bisect_iters):
            mid = 0.5 * (lo + hi)
            z_hat, converged, _ = self.solve(family, sign * mid, z_hat_r, init=z_lo)
            if not converged:
                return "no fixed point"
            if angle(normalized(z_hat, self.d), z_r) < target:
                lo, z_lo = mid, z_hat
            else:
                hi = mid
        return sign * 0.5 * (lo + hi), z_lo


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--transitions", type=int, default=100000)
    parser.add_argument("--n_inference", type=int, default=100000)
    parser.add_argument("--n_mu_states", type=int, default=8192)
    parser.add_argument("--n_mu_actions", type=int, default=64)
    parser.add_argument("--cql_tau", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--bisect_iters", type=int, default=12)
    parser.add_argument("--n_random", type=int, default=3)
    parser.add_argument("--n_subsets", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(args.seed)
    agent = load_agent(Path(args.checkpoint), device)
    reward_constructor = RewardFunctionConstructor(
        domain_name="walker", task_names=args.tasks, seed=args.seed, device=device
    )
    buffer = OneStepFBReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=Path(args.dataset_path),
        discount=agent._discount,
        transitions=args.transitions,
        device=device,
    )

    # inference set: transitions with a next dataset action, rewards from the simulator
    rng = np.random.default_rng(args.seed)
    valid = buffer.valid_indices.cpu().numpy()
    idx = torch.as_tensor(
        rng.choice(valid, min(args.n_inference, len(valid)), replace=False), device=device
    )
    storage = buffer.storage
    _, rewards = reward_constructor.process_timesteps(
        batch_physics=storage["physics"][idx.cpu().numpy()],
        batch_actions=storage["actions"][idx].cpu().numpy(),
    )
    data = {
        "obs": storage["observations"][idx],
        "actions": storage["actions"][idx],
        "next_obs": storage["next_observations"][idx],
        "next_actions": storage["next_actions"][idx],
        "rewards": rewards,
    }
    iz = ImplicitZ(agent, data, args)
    d = iz.d

    rows = []

    def add(task, family, z_hat, z_hat_r, strength=0.0, target=np.nan, converged=True, iters=0,
            multi=False):
        z = normalized(z_hat, d)
        rows.append(
            {
                "task": task,
                "family": family,
                "strength": float(strength),
                "target_angle": float(target),
                "angle": angle(z, normalized(z_hat_r, d)),
                "converged": converged,
                "iters": iters,
                "multi": multi,
                "z": z.cpu().numpy(),
            }
        )
        print(f"{task:5s} {family:6s} target {target:5.1f} angle {rows[-1]['angle']:6.2f} "
              f"strength {strength:+.4g} iters {iters} converged {converged} multi {multi}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    for task in args.tasks:
        z_hat_r = iz.z_hat_r(task)
        iz.set_task(z_hat_r)
        add(task, "G0", z_hat_r, z_hat_r)
        z_paper = torch.as_tensor(
            agent.infer_z(data["obs"], data["actions"], data["rewards"][task]), device=device
        )
        add(task, "G0p", z_paper, z_hat_r)

        # U1: zero strength reproduces G0 exactly
        for family in ("G1", "G3", "G3c", "G5", "G6"):
            z0, _, _ = iz.solve(family, 0.0, z_hat_r)
            assert torch.equal(normalized(z0, d), normalized(z_hat_r, d)), family

        specs = [("G1", 1, ANGLES), ("G1neg", -1, ANGLES), ("G3", 1, ANGLES),
                 ("G3c", 1, ANGLES), ("G5", 1, E2E_ANGLES), ("G5neg", -1, E2E_ANGLES),
                 ("G6", 1, G6_ANGLES)]
        for name, sign, targets in specs:
            family = name.replace("neg", "")
            grid = iz.angle_grid(family, sign, z_hat_r)
            for target in targets:
                match = iz.match_angle(family, sign, target, z_hat_r, grid)
                if match is None or isinstance(match, str):
                    reason = match or f"unreachable (max angle {max(g[1] for g in grid):.1f})"
                    print(f"{task:5s} {name:6s} target {target:5.1f} {reason}", flush=True)
                    continue
                # U4: the cold-start fixed point should land on the matched angle; if it
                # does not, the fixed point is not unique and the warm-start one is kept
                s, init = match
                z_hat, converged, iters = iz.solve(family, s, z_hat_r)
                multi = abs(angle(normalized(z_hat, d), normalized(z_hat_r, d)) - target) > 1.0
                if multi:
                    z_hat, converged, _ = iz.solve(family, s, z_hat_r, init=init)
                    if abs(angle(normalized(z_hat, d), normalized(z_hat_r, d)) - target) > 1.0:
                        print(f"{task:5s} {name:6s} target {target:5.1f} discontinuous", flush=True)
                        continue
                add(task, name, z_hat, z_hat_r, s, target, converged, iters, multi)

        # C1: rotate z_r towards random orthogonal directions
        unit_r = z_hat_r / z_hat_r.norm()
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


if __name__ == "__main__":
    main()
