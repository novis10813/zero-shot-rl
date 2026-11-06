"""Dynamics-shift score of a fixed FB from the one-step forward-backward residual.

For a transition (s, a, s') and a latent z, with F the mean of the two forward heads,

    e = F(s, a, z) - B(s') - gamma F(s', pi_z(s'), z),    pi_z(s') = clipped actor mean

If s' follows the dynamics that trained the FB, e is small and has a regular distribution.
The detector fits on training transitions

    mu_0(s, a, z)   = E[e | s, a, z]    an MLP regressed on e
    Sigma_0         = Cov(e - mu_0)     one covariance pooled over (s, a, z), estimated on
                                        transitions the MLP was not trained on

and scores a transition by the Mahalanobis distance

    D = (e - mu_0)^T (Sigma_0 + eps I)^{-1} (e - mu_0)

On in-distribution transitions D has mean close to z_dimension. A shifted simulator moves
s' away from what F predicts and raises D. z is drawn as in FB training (uniform on the sphere
of radius sqrt(d), mixed with B(s) at the agent's z_mix_ratio); --zs adds the stored task
latents (zero_shot_rl.latent.infer) for a fraction of the draws.

Usage (repo root), fit on the dataset the FB was trained on and print D on held-out
transitions:
    PYTHONPATH=. uv run python -m zero_shot_rl.latent.shift_detector \
        --checkpoint <fb.pt> --dataset_path <dir>/walker/rnd/dataset.npz \
        --zs <zs.npz> --out <detector.pt>

Score transitions of a rollout with load_detector(path).score(agent, s, a, s_next, z).
"""

import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import OfflineReplayBuffer, load_agent
from zero_shot_rl.env.walker import policy
from zero_shot_rl.provenance import sha256

DISCOUNT = 0.98  # agents/fb/config.yaml


@torch.no_grad()
def residual(agent, s, a, s_next, z, discount=DISCOUNT):
    """e = F(s, a, z) - B(s') - discount F(s', pi_z(s'), z), shape [N, z_dimension]."""
    fb = agent.FB

    def f(o, act):
        return sum(fb.forward_representation(o, act, z)) / 2

    return f(s, a) - fb.backward_representation(s_next) - discount * f(s_next, policy(agent, s_next, z))


def sample_z(agent, s, task_z=None, task_fraction=0.0):
    """Latents for the states `s`: FB training draws, with a fraction replaced by task latents."""
    n, d = len(s), agent._z_dimension  # pylint: disable=protected-access
    z = agent.sample_z(n)
    mix = torch.rand(n, device=s.device) < agent._z_mix_ratio  # pylint: disable=protected-access
    if mix.any():
        with torch.no_grad():
            b = agent.FB.backward_representation(s[torch.randperm(n, device=s.device)][mix])
        z[mix] = math.sqrt(d) * torch.nn.functional.normalize(b, dim=1)
    if task_z is not None and task_fraction > 0:
        pick = torch.rand(n, device=s.device) < task_fraction
        z[pick] = task_z.to(s.device)[torch.randint(len(task_z), (int(pick.sum()),), device=s.device)]
    return z


class ShiftDetector(torch.nn.Module):
    """mu_0 network, residual normalisation and Sigma_0 of one FB checkpoint."""

    def __init__(self, obs_dim, action_dim, z_dim, hidden=1024, layers=3, eps=1e-6):
        super().__init__()
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, z_dim=z_dim, hidden=hidden,
                           layers=layers, eps=eps)
        dims = [obs_dim + action_dim + z_dim] + [hidden] * layers
        body = []
        for i, o in zip(dims[:-1], dims[1:]):
            body += [torch.nn.Linear(i, o), torch.nn.LayerNorm(o), torch.nn.ReLU()]
        self.net = torch.nn.Sequential(*body, torch.nn.Linear(hidden, z_dim))
        # Scale of e, so that the regression loss weights the dimensions equally.
        self.register_buffer("e_mean", torch.zeros(z_dim))
        self.register_buffer("e_std", torch.ones(z_dim))
        self.register_buffer("precision_chol", torch.eye(z_dim))  # W with W W^T = (Sigma_0 + eps I)^{-1}
        self.register_buffer("sigma", torch.eye(z_dim))
        self.eps = eps

    def mu(self, s, a, z):
        return self.net(torch.cat([s, a, z], -1)) * self.e_std + self.e_mean

    def set_covariance(self, sigma):
        """Store Sigma_0 (float64 [d, d]) and the factor of (Sigma_0 + eps I)^{-1}."""
        d = sigma.shape[0]
        reg = sigma + self.eps * torch.eye(d, dtype=sigma.dtype)
        chol = torch.linalg.cholesky(reg)
        self.sigma.copy_(sigma.float())
        self.precision_chol.copy_(torch.linalg.solve_triangular(chol, torch.eye(d, dtype=sigma.dtype), upper=False).T.float())

    @torch.no_grad()
    def distance(self, s, a, z, e):
        """D for given residuals e: |W^T (e - mu_0)|^2."""
        r = e - self.mu(s, a, z)
        return ((r.double() @ self.precision_chol.double()) ** 2).sum(-1)

    @torch.no_grad()
    def score(self, agent, s, a, s_next, z):
        """D of transitions (s, a, s') under latent z; tensors with the batch on axis 0."""
        return self.distance(s, a, z, residual(agent, s, a, s_next, z))

    def save(self, path, **meta):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)


def load_detector(path, device=torch.device("cpu")):
    blob = torch.load(path, map_location=device)
    detector = ShiftDetector(**blob["config"]).to(device)
    detector.load_state_dict(blob["state_dict"])
    return detector.eval()


def covariance(r):
    """Sample covariance of rows of r (float64)."""
    r = r.double()
    r = r - r.mean(0)
    return r.T @ r / (len(r) - 1)


def make_draw(agent, storage, args, device):
    """draw(idx, n): n transitions of indices `idx` with latents z and residuals e."""
    s_all, a_all, n_all = (storage[k].float().to(device) for k in ("observations", "actions", "next_observations"))

    def draw(idx, n):
        pick = idx[torch.randint(len(idx), (n,), device=device)]
        s, a, s2 = s_all[pick], a_all[pick], n_all[pick]
        z = sample_z(agent, s, args.task_z, args.task_fraction)
        return s, a, z, residual(agent, s, a, s2, z)

    return draw, s_all.shape[1], a_all.shape[1]


def fit(agent, draw, dims, train_idx, args, device):
    """Train mu_0 on transitions `train_idx`."""
    z_dim = agent._z_dimension  # pylint: disable=protected-access
    detector = ShiftDetector(*dims, z_dim, args.hidden, args.layers, args.eps).to(device)
    e = draw(train_idx, 100_000)[-1]
    detector.e_mean.copy_(e.mean(0))
    detector.e_std.copy_(e.std(0).clamp_min(1e-8))
    opt = torch.optim.Adam(detector.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for step in range(1, args.steps + 1):
        s, a, z, e = draw(train_idx, args.batch_size)
        loss = (((detector.mu(s, a, z) - e) / detector.e_std) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)
    return detector.eval()


@torch.no_grad()
def centered(detector, draw, idx, n_batches, batch=10_000):
    """e - mu_0 on `n_batches` batches of (s, a, z) drawn from the transitions `idx`."""
    out = []
    for _ in range(n_batches):
        s, a, z, e = draw(idx, batch)
        out.append((e - detector.mu(s, a, z)).cpu())
    return torch.cat(out)


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--zs", type=str, default=None, help="latents NPZ of zero_shot_rl.latent.infer")
    parser.add_argument("--tasks", nargs="*", default=["walk"], help="tasks whose z_r joins the z draws")
    parser.add_argument("--task_fraction", type=float, default=0.2)
    parser.add_argument("--transitions", type=int, default=100000)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--cov_batches", type=int, default=20, help="10k (s, a, z) draws each for Sigma_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(Path(args.checkpoint), device).eval()
    buffer = OfflineReplayBuffer(
        reward_constructor=None, dataset_path=Path(args.dataset_path), discount=DISCOUNT,
        device=torch.device("cpu"), task="walk", relabel=False, transitions=args.transitions,
    )
    storage = buffer.storage
    n = len(storage["observations"])
    perm = torch.as_tensor(np.random.default_rng(args.seed).permutation(n))
    splits = {"train": perm[: int(0.8 * n)], "cov": perm[int(0.8 * n): int(0.9 * n)], "val": perm[int(0.9 * n):]}

    args.task_z = None
    if args.zs:
        zs = np.load(args.zs)
        rows = [i for i in range(len(zs["task"])) if zs["family"][i] == "z_r" and zs["task"][i] in args.tasks]
        args.task_z = torch.as_tensor(zs["z"][rows], dtype=torch.float32)

    draw, *dims = make_draw(agent, storage, args, device)
    detector = fit(agent, draw, dims, splits["train"].to(device), args, device)
    detector.set_covariance(covariance(centered(detector, draw, splits["cov"].to(device), args.cov_batches)).cpu())
    detector.to(device)

    for name in ("train", "cov", "val"):
        r = centered(detector, draw, splits[name].to(device), 5)
        d = ((r.double() @ detector.precision_chol.double().cpu()) ** 2).sum(-1)
        print(f"{name:5s}  D mean {d.mean():.2f}  median {d.median():.2f}  p99 {d.quantile(0.99):.2f}  (z_dim {r.shape[1]})")
    detector.save(args.out, checkpoint_sha256=sha256(Path(args.checkpoint)),
                  args={k: v for k, v in vars(args).items() if k != "task_z"})


if __name__ == "__main__":
    main()
