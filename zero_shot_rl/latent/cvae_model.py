"""Conditional VAE ensemble of the transition P(s' | s, a), after Prashant et al. 2025 (arXiv 2503.05238).

Each member is a CVAE whose encoder q(z | s', s, a) and decoder p(s' | z, s, a) are ReLU MLPs; the
condition is the state-action pair (the paper conditions on s only, as its policy is deterministic).
A member is trained on the ELBO of the standardised target y (s' as in the paper, or Delta s = s' - s),

    |y - dec(z, s, a)|^2 + beta KL(q(z | y, s, a) || N(0, I)),   z ~ q,

on baseline-environment transitions only, with the episode split, the validation early stopping
and the ExORL standardisation of zero_shot_rl.latent.dynamics_model. N members differ in their
initialisation and batch order (the paper uses N = 5).

A transition is scored by the reconstruction error of each member, with z the mean of q,

    e_i = s' - s'_i = (s' - s) - f_i(s, a),     r_i = mean(e_i^2)    (the units of the observation squared)

The paper scores the mean squared error of the reconstruction of s', r_i = mean(e_i^2) over the
observation dimensions, which is what the detector below uses (r_i is |e_i|^2 / obs_dim). The
detector is the inductive conformal prediction of the paper: the calibration set holds the r_i
of all members on baseline validation transitions (M x N values). With delta in (0, 1) the
threshold is its element number floor(delta M N) in ascending order, and a transition is OOD when
no member has r_i below the threshold, i.e. min_i r_i >= threshold.

Usage (repo root):
    PYTHONPATH=. uv run python -m zero_shot_rl.latent.cvae_model \
        --rollouts <train_s0.npz> <train_s1.npz> <train_s2.npz> --stats_dataset <walker/rnd/dataset.npz> \
        --out <model.pt>

Score paired transitions with zero_shot_rl.latent.cvae_score.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.latent.dynamics_model import (
    TARGETS, exorl_statistics, flatten, load_rollouts, split_episodes,
)
from zero_shot_rl.provenance import sha256


def mlp(in_dim, hidden, layers, out_dim):
    dims = [in_dim] + [hidden] * layers
    body = []
    for i, o in zip(dims[:-1], dims[1:]):
        body += [torch.nn.Linear(i, o), torch.nn.ReLU()]
    return torch.nn.Sequential(*body, torch.nn.Linear(hidden, out_dim))


class CVAE(torch.nn.Module):
    """One member: encoder (x, y) -> (mu, log var) of z, decoder (x, z) -> y, on standardised x = [s, a] and y."""

    def __init__(self, obs_dim, action_dim, latent, hidden, layers):
        super().__init__()
        self.latent = latent
        self.encoder = mlp(obs_dim + action_dim + obs_dim, hidden, layers, 2 * latent)
        self.decoder = mlp(obs_dim + action_dim + latent, hidden, layers, obs_dim)

    def encode(self, x, y):
        mu, log_var = self.encoder(torch.cat([x, y], -1)).chunk(2, -1)
        return mu, log_var.clamp(-10.0, 10.0)

    def decode(self, x, z):
        return self.decoder(torch.cat([x, z], -1))

    def loss(self, x, y, beta, sample=True):
        """Mean over the batch of (squared reconstruction error + beta KL), summed over dimensions.

        z is sampled from q if `sample`, else it is the mean of q (the quantity scored and used for early stopping).
        """
        mu, log_var = self.encode(x, y)
        z = mu + (0.5 * log_var).exp() * torch.randn_like(mu) if sample else mu
        rec = ((self.decode(x, z) - y) ** 2).sum(-1)
        kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(-1)
        return (rec + beta * kl).mean(), rec.mean(), kl.mean()


class CVAEEnsemble(torch.nn.Module):
    """N CVAEs of f(s, a) ~ Delta s with the standardisation of [s, a] and of the target as buffers.

    target "next_state": y = s' (standardised with the statistics of s) and f(s, a) = s' - s.
    target "delta": y = Delta s (standardised). The error e = s' - s'_reconstructed is the same quantity in both.
    """

    def __init__(self, obs_dim, action_dim, members=5, latent=8, hidden=512, layers=3, beta=1.0, target="next_state"):
        super().__init__()
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, members=members, latent=latent,
                           hidden=hidden, layers=layers, beta=beta, target=target)
        self.target = target
        self.beta = beta
        self.members = torch.nn.ModuleList(CVAE(obs_dim, action_dim, latent, hidden, layers) for _ in range(members))
        self.register_buffer("x_mean", torch.zeros(obs_dim + action_dim))
        self.register_buffer("x_std", torch.ones(obs_dim + action_dim))
        self.register_buffer("y_mean", torch.zeros(obs_dim))
        self.register_buffer("y_std", torch.ones(obs_dim))

    def inputs(self, s, a):
        return (torch.cat([s, a], -1) - self.x_mean) / self.x_std

    def target_of(self, s, s_next):
        """Standardised regression target; the difference s' - s is taken in float64 before the cast."""
        s64, s2_64 = torch.as_tensor(s, dtype=torch.float64), torch.as_tensor(s_next, dtype=torch.float64)
        y = s2_64 if self.target == "next_state" else s2_64 - s64
        return (y.float().to(self.y_mean.device) - self.y_mean) / self.y_std

    @torch.no_grad()
    def displacement(self, s, a, s_next):
        """f_i(s, a) of every member [N, B, obs_dim], reconstructing s' with z the mean of q(z | s', s, a)."""
        s = torch.as_tensor(s, dtype=torch.float64)
        s32 = s.float()
        x, y = self.inputs(s32, a), self.target_of(s, s_next)
        out = []
        for member in self.members:
            mu, _ = member.encode(x, y)
            f = member.decode(x, mu) * self.y_std + self.y_mean
            out.append(f - s32 if self.target == "next_state" else f)
        return torch.stack(out)

    @torch.no_grad()
    def error(self, s, a, s_next):
        """e_i = (s' - s) - f_i(s, a) of every member [N, B, obs_dim]."""
        s64 = torch.as_tensor(s, dtype=torch.float64)
        delta = (torch.as_tensor(s_next, dtype=torch.float64) - s64).float()
        return delta - self.displacement(s, a, s_next)

    def save(self, path, **meta):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)


def load_cvae_ensemble(path, device=torch.device("cpu")):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = CVAEEnsemble(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob["meta"]


def icp_threshold(calibration, delta):
    """Element number floor(delta M N) (1-based, ascending) of the pooled calibration errors."""
    pooled = np.sort(np.asarray(calibration).ravel())
    return pooled[min(max(int(np.floor(delta * len(pooled))), 1), len(pooled)) - 1]


def is_ood(member_errors, threshold):
    """Algorithm 3 of the paper: OOD when no member error [N, ...] is below the threshold."""
    return np.asarray(member_errors).min(0) >= threshold


def fit(model, train, val, args, device, stats=None):
    """Train every member on `train`, early stopping each on the validation loss (mean-z reconstruction + beta KL).

    Standardisation as in zero_shot_rl.latent.dynamics_model.fit. Returns the model and the best
    validation loss of each member; the best epoch, reconstruction and KL (in total and per dimension
    of z) of each member are in `model.fit_info`.
    """
    nxt = model.target == "next_state"
    s, a = (torch.as_tensor(train[k], dtype=torch.float32, device=device) for k in ("s", "a"))
    delta = torch.as_tensor(train["s2"] - train["s"], dtype=torch.float64).float().to(device)
    target = torch.as_tensor(train["s2"], dtype=torch.float32, device=device) if nxt else delta
    x = torch.cat([s, a], -1)
    if stats is None:
        stats = ((x.mean(0), x.std(0)), (target.mean(0), target.std(0)))
    elif nxt:
        stats = (stats[0], (stats[0][0][:s.shape[1]], stats[0][1][:s.shape[1]]))
    (x_mean, x_std), (y_mean, y_std) = ((torch.as_tensor(v, dtype=torch.float32) for v in pair) for pair in stats)
    model.x_mean.copy_(x_mean)
    model.x_std.copy_(x_std.clamp_min(1e-6))
    model.y_mean.copy_(y_mean)
    model.y_std.copy_(y_std.clamp_min(1e-6))
    xs, y = model.inputs(s, a), model.target_of(train["s"], train["s2"]).to(device)
    vx = model.inputs(*(torch.as_tensor(val[k], dtype=torch.float32, device=device) for k in ("s", "a")))
    vy = model.target_of(val["s"], val["s2"]).to(device)

    bests, info = [], []
    for i, member in enumerate(model.members):
        opt = torch.optim.Adam(member.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, best_state, best_epoch, stale = float("inf"), None, 0, 0
        for epoch in range(1, args.epochs + 1):
            member.train()
            perm = torch.randperm(len(xs), device=device)
            for j in range(0, len(perm), args.batch_size):
                idx = perm[j:j + args.batch_size]
                loss, _, _ = member.loss(xs[idx], y[idx], model.beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            member.eval()
            with torch.no_grad():
                val_loss, val_rec, val_kl = (v.item() for v in member.loss(vx, vy, model.beta, sample=False))
            if val_loss < best:
                best, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in member.state_dict().items()}
                best_rec, best_kl = val_rec, val_kl
            else:
                stale += 1
            if epoch % args.log_every == 0:
                print(f"member {i} epoch {epoch:4d}  train {loss.item():.4f}  val {val_loss:.4f} "
                      f"(rec {val_rec:.4f}, kl {val_kl:.4f})  best {best:.4f}", flush=True)
            if stale >= args.patience:
                print(f"member {i} stop at epoch {epoch}, best validation {best:.4f}", flush=True)
                break
        member.load_state_dict(best_state)
        member.eval()
        with torch.no_grad():
            mu, log_var = member.encode(vx, vy)
            kl_dims = (-0.5 * (1 + log_var - mu ** 2 - log_var.exp())).mean(0).cpu().numpy()
        bests.append(best)
        info.append({"best_epoch": best_epoch, "val_loss": best, "val_rec": best_rec, "val_kl": best_kl,
                     "val_kl_per_dim": kl_dims})
    model.fit_info = info
    return model.eval(), bests


def main():
    parser = ArgumentParser()
    parser.add_argument("--rollouts", nargs="+", required=True,
                        help="baseline rollouts NPZ (zero_shot_rl.eval.paired_step), pooled")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stats_dataset", type=str, default=None,
                        help="ExORL dataset NPZ whose mean and std of [s, a] and of Delta s standardise "
                             "the inputs and the target (default: those of the training transitions)")
    parser.add_argument("--val_episodes", type=int, default=4, help="validation episodes per rollout file")
    parser.add_argument("--target", choices=TARGETS, default="next_state",
                        help="reconstruct s' (default, as in the paper) or Delta s")
    parser.add_argument("--members", type=int, default=5, help="ensemble size N")
    parser.add_argument("--latent", type=int, default=8, help="dimension of z")
    parser.add_argument("--beta", type=float, default=1.0, help="weight of the KL term")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = load_rollouts(args.rollouts)
    masks = split_episodes([len(f["s"]) for f in files], args.val_episodes, args.seed)
    train, val = flatten(files, masks, False), flatten(files, masks, True)
    print(f"train {len(train['s'])} transitions, validation {len(val['s'])}", flush=True)

    model = CVAEEnsemble(train["s"].shape[1], train["a"].shape[1], args.members, args.latent, args.hidden,
                         args.layers, args.beta, args.target).to(device)
    stats = exorl_statistics(args.stats_dataset) if args.stats_dataset else None
    model, bests = fit(model, train, val, args, device, stats)
    e_val = model.error(torch.as_tensor(val["s"], device=device),
                        torch.as_tensor(val["a"], dtype=torch.float32, device=device),
                        torch.as_tensor(val["s2"], device=device))
    val_member = (e_val ** 2).mean(-1).cpu().numpy()
    print(f"validation  member r median {np.median(val_member, axis=1)}  "
          f"threshold p95 {icp_threshold(val_member, 0.95):.3g}  p99 {icp_threshold(val_member, 0.99):.3g}")
    meta = {
        "val_member": val_member, "val_loss": bests, "fit_info": model.fit_info,
        "val_episodes": [f["episode"][m].tolist() for f, m in zip(files, masks)],
        "x_mean": model.x_mean.cpu().numpy(), "x_std": model.x_std.cpu().numpy(),
        "y_mean": model.y_mean.cpu().numpy(), "y_std": model.y_std.cpu().numpy(),
        "rollouts_sha256": [sha256(Path(p)) for p in args.rollouts], "args": vars(args),
    }
    model.cpu().save(args.out, **meta)


if __name__ == "__main__":
    main()
