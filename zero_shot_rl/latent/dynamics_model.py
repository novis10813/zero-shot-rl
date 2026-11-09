"""Action-conditioned dynamics model for detecting dynamics shifts.

A ReLU MLP f predicts the state displacement of one step from the state and the action,

    Delta s_t = s_{t+1} - s_t,    e_t = Delta s_t - f(s_t, a_t)

It is trained on baseline-environment transitions only: the stock rollouts (episodes of
zero_shot_rl.eval.paired_step, `observations`, `actions`, `next_rollout`) of one or several FB
checkpoints, pooled. Episodes, not transitions, are split into training and validation, because
neighbouring steps of an episode are almost identical. The validation set stops the training
(best validation MSE of the standardised Delta s) and gives Sigma_0 = Cov(e) of the baseline.
A transition is scored by

    |e|_2                           the error in the units of the observation
    D_dyn = e^T Sigma_0^{-1} e      the Mahalanobis distance, as in zero_shot_rl.latent.shift_detector

The inputs [s, a] and the target Delta s are standardised with the mean and std of a full ExORL
dataset (--stats_dataset).
The scores of the validation episodes are stored in the model file, from which the threshold of
a detector is taken (a percentile of the baseline scores).

Usage (repo root):
    PYTHONPATH=. uv run python -m zero_shot_rl.latent.dynamics_model \
        --rollouts <train_s0.npz> <train_s1.npz> <train_s2.npz> --stats_dataset <walker/rnd/dataset.npz> \
        --out <model.pt>

Score paired transitions with zero_shot_rl.latent.dynamics_score.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.provenance import sha256


class DynamicsModel(torch.nn.Module):
    """f(s, a) -> Delta s, a ReLU MLP, with the standardisation and Sigma_0 of the baseline stored as buffers."""

    def __init__(self, obs_dim, action_dim, hidden=64, layers=2, eps=1e-6):
        super().__init__()
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, layers=layers, eps=eps)
        dims = [obs_dim + action_dim] + [hidden] * layers
        body = []
        for i, o in zip(dims[:-1], dims[1:]):
            body += [torch.nn.Linear(i, o), torch.nn.ReLU()]
        self.net = torch.nn.Sequential(*body, torch.nn.Linear(hidden, obs_dim))
        self.register_buffer("x_mean", torch.zeros(obs_dim + action_dim))
        self.register_buffer("x_std", torch.ones(obs_dim + action_dim))
        self.register_buffer("y_mean", torch.zeros(obs_dim))
        self.register_buffer("y_std", torch.ones(obs_dim))
        self.register_buffer("precision_chol", torch.eye(obs_dim))  # W with W W^T = (Sigma_0 + eps I)^{-1}
        self.eps = eps

    def standardised(self, s, a):
        """Network output, Delta s in units of its standard deviation."""
        return self.net((torch.cat([s, a], -1) - self.x_mean) / self.x_std)

    def forward(self, s, a):
        return self.standardised(s, a) * self.y_std + self.y_mean

    def set_covariance(self, e):
        """Store the factor of (Cov(e) + eps I)^{-1} for the errors e [N, obs_dim] of baseline transitions."""
        e = e.double() - e.double().mean(0)
        reg = e.T @ e / (len(e) - 1) + self.eps * torch.eye(e.shape[1], dtype=torch.float64)
        chol = torch.linalg.cholesky(reg)
        eye = torch.eye(e.shape[1], dtype=torch.float64)
        self.precision_chol.copy_(torch.linalg.solve_triangular(chol, eye, upper=False).T.float())

    @torch.no_grad()
    def error(self, s, a, s_next):
        """e = (s' - s) - f(s, a); the difference is taken in float64 before the cast."""
        s64 = torch.as_tensor(s, dtype=torch.float64)
        delta = torch.as_tensor(s_next, dtype=torch.float64) - s64
        return delta.float() - self(s64.float(), a)

    def distance(self, e):
        """D_dyn = |W^T e|^2."""
        return ((e.double() @ self.precision_chol.double()) ** 2).sum(-1)

    def save(self, path, **meta):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)


def load_dynamics_model(path, device=torch.device("cpu")):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = DynamicsModel(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob["meta"]


def split_episodes(sizes, val_per_file, seed):
    """Boolean validation masks, one [episodes] array per file; `val_per_file` random episodes each."""
    rng = np.random.default_rng(seed)
    masks = []
    for n in sizes:
        if not 0 < val_per_file < n:
            raise ValueError(f"cannot take {val_per_file} validation episodes from {n}")
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, val_per_file, replace=False)] = True
        masks.append(mask)
    return masks


def load_rollouts(paths):
    """Per file: observations, actions and next observations as float64 [episodes, steps, dim]."""
    out = []
    for path in paths:
        data = np.load(path)
        out.append({"s": data["observations"], "a": data["actions"], "s2": data["next_rollout"],
                    "episode": data["episode"]})
    return out


def flatten(files, masks, want_val):
    """Transitions of the training (want_val False) or validation episodes of all files, pooled."""
    parts = {k: [] for k in ("s", "a", "s2")}
    for f, mask in zip(files, masks):
        keep = mask if want_val else ~mask
        for k in parts:
            parts[k].append(f[k][keep].reshape(-1, f[k].shape[-1]))
    return {k: np.concatenate(v) for k, v in parts.items()}


def exorl_statistics(path):
    """Mean and standard deviation of [s, a] and of Delta s over all transitions of an ExORL dataset NPZ.

    Pairs observation[t] with action[t + 1], as agents.base.OfflineReplayBuffer does.
    Returns ((x_mean, x_std), (y_mean, y_std)).
    """
    n, total, square = 0, 0.0, 0.0
    for episode in np.load(path, allow_pickle=True).values():
        episode = episode.item()
        obs = episode["observation"].astype(np.float64)
        x = np.concatenate([obs[:-1], episode["action"][1:]], axis=1)
        y = obs[1:] - obs[:-1]
        both = np.concatenate([x, y], axis=1)
        n, total, square = n + len(both), total + both.sum(0), square + (both ** 2).sum(0)
    mean = total / n
    std = np.sqrt(square / n - mean ** 2)
    d = y.shape[1]
    return (mean[:-d], std[:-d]), (mean[-d:], std[-d:])


def fit(model, train, val, args, device, stats=None):
    """Train on `train` with early stopping on the validation MSE of the standardised Delta s.

    [s, a] and Delta s are standardised with `stats` = ((x_mean, x_std), (y_mean, y_std)) if given,
    else with the training transitions.
    """
    s, a = (torch.as_tensor(train[k], dtype=torch.float32, device=device) for k in ("s", "a"))
    delta = torch.as_tensor(train["s2"] - train["s"], dtype=torch.float64).float().to(device)
    x = torch.cat([s, a], -1)
    if stats is None:
        stats = ((x.mean(0), x.std(0)), (delta.mean(0), delta.std(0)))
    (x_mean, x_std), (y_mean, y_std) = ((torch.as_tensor(v, dtype=torch.float32) for v in pair) for pair in stats)
    model.x_mean.copy_(x_mean)
    model.x_std.copy_(x_std.clamp_min(1e-6))
    model.y_mean.copy_(y_mean)
    model.y_std.copy_(y_std.clamp_min(1e-6))
    y = (delta - model.y_mean) / model.y_std

    vs, va = (torch.as_tensor(val[k], dtype=torch.float32, device=device) for k in ("s", "a"))
    vy = (torch.as_tensor(val["s2"] - val["s"], dtype=torch.float64).float().to(device) - model.y_mean) / model.y_std

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(s), device=device)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = ((model.standardised(s[idx], a[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = ((model.standardised(vs, va) - vy) ** 2).mean().item()
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch % args.log_every == 0:
            print(f"epoch {epoch:4d}  train {loss.item():.4f}  val {val_loss:.4f}  best {best:.4f}", flush=True)
        if stale >= args.patience:
            print(f"stop at epoch {epoch}, best validation {best:.4f}", flush=True)
            break
    model.load_state_dict(best_state)
    return model.eval(), best


def main():
    parser = ArgumentParser()
    parser.add_argument("--rollouts", nargs="+", required=True,
                        help="baseline rollouts NPZ (zero_shot_rl.eval.paired_step), pooled")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stats_dataset", type=str, default=None,
                        help="ExORL dataset NPZ whose mean and std of [s, a] and of Delta s standardise "
                             "the inputs and the target (default: those of the training transitions)")
    parser.add_argument("--val_episodes", type=int, default=4, help="validation episodes per rollout file")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = load_rollouts(args.rollouts)
    masks = split_episodes([len(f["s"]) for f in files], args.val_episodes, args.seed)
    train, val = flatten(files, masks, False), flatten(files, masks, True)
    print(f"train {len(train['s'])} transitions, validation {len(val['s'])}", flush=True)

    model = DynamicsModel(train["s"].shape[1], train["a"].shape[1], args.hidden, args.layers, args.eps).to(device)
    stats = exorl_statistics(args.stats_dataset) if args.stats_dataset else None
    model, best = fit(model, train, val, args, device, stats)
    e_val = model.error(torch.as_tensor(val["s"], device=device), torch.as_tensor(val["a"], dtype=torch.float32, device=device),
                        torch.as_tensor(val["s2"], device=device))
    model.set_covariance(e_val.cpu())
    d_val = model.distance(e_val).cpu()
    print(f"validation  |e| median {e_val.norm(dim=1).median():.4f}  D_dyn mean {d_val.mean():.2f} "
          f"(obs_dim {e_val.shape[1]})")
    meta = {
        "val_norm": e_val.norm(dim=1).cpu().numpy(), "val_maha": d_val.numpy(), "val_loss": best,
        "val_episodes": [f["episode"][m].tolist() for f, m in zip(files, masks)],
        "x_mean": model.x_mean.cpu().numpy(), "x_std": model.x_std.cpu().numpy(),
        "y_mean": model.y_mean.cpu().numpy(), "y_std": model.y_std.cpu().numpy(),
        "rollouts_sha256": [sha256(Path(p)) for p in args.rollouts], "args": vars(args),
    }
    model.cpu().save(args.out, **meta)


if __name__ == "__main__":
    main()
