"""Two-predictor Action Transformation (FB-AA) without modifying the FB backbone.

Given a frozen pretrained FB agent with backward representation B_0:
    1. g_0(s, a) ~ E_{P_0}[B_0(s') | s, a] is trained offline on stock data (e.g. ExORL rnd).
    2. At test time in shifted dynamics P_1, g_1(s, a) is fine-tuned from g_0 on test-time
       trajectories collected in P_1 (using separate finetune seeds).
    3. Action transformation T_theta(s, a) = tanh(MLP(s, a)) in (-1, 1)^{d_a} is trained to
       minimize:
           |g_1(s, T_theta(s, a)) - g_0(s, a)|^2 + lam * |T_theta(s, a) - a|^2

Usage (train g_0 offline from ExORL dataset):
    PYTHONPATH=. uv run python -m zero_shot_rl.latent.action_transform \
        --dataset datasets/walker/rnd/dataset.npz \
        --checkpoint <fb.pt> --out-dir <dir>
"""

import copy
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from zero_shot_rl.provenance import sha256


def mlp(dims, out):
    layers = []
    for i, o in zip(dims[:-1], dims[1:]):
        layers += [torch.nn.Linear(i, o), torch.nn.ReLU()]
    return torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], out))


class FeaturePredictor(torch.nn.Module):
    """One-step backward-feature predictor g(s, a) -> B_0(s')."""

    def __init__(self, obs_dim, action_dim, feature_dim, hidden=512, layers=3):
        super().__init__()
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, feature_dim=feature_dim,
                           hidden=hidden, layers=layers)
        self.net = mlp([obs_dim + action_dim] + [hidden] * layers, feature_dim)
        self.register_buffer("x_mean", torch.zeros(obs_dim + action_dim))
        self.register_buffer("x_std", torch.ones(obs_dim + action_dim))

    def forward(self, s, a):
        x = (torch.cat([s, a], dim=-1) - self.x_mean) / self.x_std
        return self.net(x)

    def save(self, path, **meta):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)


def load_feature_predictor(path, device=torch.device("cpu")):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = FeaturePredictor(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob["meta"]


class ActionTransformer(torch.nn.Module):
    """Residual action transformation T_theta(s, a) = tanh(atanh(clip(a)) + delta(s, a)) in (-1, 1)."""

    def __init__(self, obs_dim, action_dim, hidden=256, layers=3):
        super().__init__()
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, layers=layers)
        self.net = mlp([obs_dim + action_dim] + [hidden] * layers, action_dim)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("x_mean", torch.zeros(obs_dim + action_dim))
        self.register_buffer("x_std", torch.ones(obs_dim + action_dim))

    def forward(self, s, a):
        x = (torch.cat([s, a], dim=-1) - self.x_mean) / self.x_std
        base = torch.atanh(a.clamp(-0.999, 0.999))
        return torch.tanh(base + self.net(x))

    def save(self, path, **meta):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)


def load_action_transformer(path, device=torch.device("cpu")):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = ActionTransformer(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob["meta"]


def load_exorl_transitions(dataset_path):
    """Load (s, a, s') from an ExORL dataset.npz, pairing observation[t] with action[t + 1].

    Returns dict with float32 arrays s, a, s2 [N, d] and episode_id [N].
    """
    s, a, s2, ep_id = [], [], [], []
    with np.load(dataset_path, allow_pickle=True) as archive:
        for idx, ep in enumerate(archive.values()):
            ep = ep.item()
            obs = ep["observation"].astype(np.float32)
            act = ep["action"].astype(np.float32)
            if len(obs) < 2:
                continue
            s.append(obs[:-1])
            a.append(act[1:])
            s2.append(obs[1:])
            ep_id.append(np.full(len(obs) - 1, idx, dtype=np.int64))
    return dict(
        s=np.concatenate(s),
        a=np.concatenate(a),
        s2=np.concatenate(s2),
        episode_id=np.concatenate(ep_id),
    )


def load_stock_transitions(stock_paths, val_seed=40):
    """Load (s, a, s') from T07 fb_trajectories_seed*.npz files.

    Returns dict with float32 arrays s, a, s2 [N, d] and boolean val_mask [N].
    """
    s, a, s2, val = [], [], [], []
    for path in stock_paths:
        with np.load(path, allow_pickle=False) as f:
            obs, act, episodes = f["observations"], f["actions"], f["episodes"]
            for t in range(obs.shape[0]):
                for e in range(obs.shape[1]):
                    o = obs[t, e].astype(np.float32)
                    ac = act[t, e].astype(np.float32)
                    n = len(o) - 1
                    if n < 1:
                        continue
                    s.append(o[:-1])
                    a.append(ac[:n])
                    s2.append(o[1:])
                    val.append(np.full(n, int(episodes[e]) >= val_seed, dtype=bool))
    return dict(
        s=np.concatenate(s),
        a=np.concatenate(a),
        s2=np.concatenate(s2),
        val_mask=np.concatenate(val),
    )


@torch.no_grad()
def compute_backward_features(agent, s_next, device, chunk=65536):
    """Compute frozen B_0(s') in chunks."""
    out = []
    for i in range(0, len(s_next), chunk):
        x = torch.as_tensor(s_next[i:i + chunk], dtype=torch.float32, device=device)
        out.append(agent.FB.backward_representation(x).float())
    return torch.cat(out)


def train_g0_model(s, a, b_next, train_mask, val_mask, hidden=512, layers=3,
                   steps=50000, batch_size=1024, lr=3e-4, val_every=500, val_samples=8192,
                   seed=0, device=torch.device("cpu")):
    """Train g_0(s, a) -> B_0(s') on stock transitions. Returns (best_model, final_model, history)."""
    torch.manual_seed(seed)
    obs_dim, action_dim, feature_dim = s.shape[1], a.shape[1], b_next.shape[1]
    model = FeaturePredictor(obs_dim, action_dim, feature_dim, hidden=hidden, layers=layers).to(device)

    s_tr = torch.as_tensor(s[train_mask], dtype=torch.float32, device=device)
    a_tr = torch.as_tensor(a[train_mask], dtype=torch.float32, device=device)
    b_tr = torch.as_tensor(b_next[train_mask], dtype=torch.float32, device=device)

    x_tr = torch.cat([s_tr, a_tr], dim=-1)
    model.x_mean.copy_(x_tr.mean(0))
    model.x_std.copy_(x_tr.std(0).clamp_min(1e-6))

    s_va = torch.as_tensor(s[val_mask], dtype=torch.float32, device=device)
    a_va = torch.as_tensor(a[val_mask], dtype=torch.float32, device=device)
    b_va = torch.as_tensor(b_next[val_mask], dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_val = None, float("inf")
    history = []
    n_tr = len(s_tr)

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n_tr, (batch_size,), device=device)
        pred = model(s_tr[idx], a_tr[idx])
        loss = ((pred - b_tr[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % val_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                n_va = min(val_samples, len(s_va))
                v_idx = torch.arange(n_va, device=device)
                v_loss = ((model(s_va[v_idx], a_va[v_idx]) - b_va[v_idx]) ** 2).mean().item()
            history.append({"step": step, "train_mse": float(loss.item()), "val_mse": float(v_loss)})
            if v_loss < best_val:
                best_val = v_loss
                best_state = copy.deepcopy(model.state_dict())

    best_model = FeaturePredictor(**model.config).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()
    model.eval()
    return best_model, model, history


def _model_device(model):
    param = next(model.parameters(), None)
    if param is not None:
        return param.device
    buf = next(model.buffers(), None)
    return buf.device if buf is not None else torch.device("cpu")


def finetune_g1(g0, s, a, b_next, steps=2000, batch_size=256, lr=1e-4):
    """Fine-tune g_1 from g_0 on test-time transitions from P_1 without mutating g_0."""
    device = _model_device(g0)
    g1 = copy.deepcopy(g0).to(device)
    g1.requires_grad_(True)
    g1.train()

    s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
    b_t = torch.as_tensor(b_next, dtype=torch.float32, device=device)
    n = len(s_t)
    bs = min(batch_size, n)

    opt = torch.optim.Adam(g1.parameters(), lr=lr)
    last_loss = 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        loss = ((g1(s_t[idx], a_t[idx]) - b_t[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    g1.eval()
    return g1, last_loss


def pretrain_identity(transformer, s, a, steps=500, batch_size=256, lr=1e-3):
    """Initialize T_theta(s, a) ~ a before optimizing the dynamics matching objective."""
    device = next(transformer.parameters()).device
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
    x_t = torch.cat([s_t, a_t], dim=-1)
    transformer.x_mean.copy_(x_t.mean(0))
    transformer.x_std.copy_(x_t.std(0).clamp_min(1e-6))

    transformer.train()
    n = len(s_t)
    bs = min(batch_size, n)
    opt = torch.optim.Adam(transformer.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        pred = transformer(s_t[idx], a_t[idx])
        loss = ((pred - a_t[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    transformer.eval()


def train_action_transformer(g0, g1, s, a, lam=0.01, steps=2000, batch_size=256, lr=3e-4,
                             hidden=256, layers=3, pretrain_steps=0, seed=0):
    """Train T_theta(s, a) to minimize |g_1(s, T(s, a)) - g_0(s, a)|^2 + lam |T(s, a) - a|^2."""
    torch.manual_seed(seed)
    device = _model_device(g0)
    obs_dim, action_dim = s.shape[1], a.shape[1]
    transformer = ActionTransformer(obs_dim, action_dim, hidden=hidden, layers=layers).to(device)
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
    x_t = torch.cat([s_t, a_t], dim=-1)
    transformer.x_mean.copy_(x_t.mean(0))
    transformer.x_std.copy_(x_t.std(0).clamp_min(1e-6))
    if pretrain_steps > 0:
        pretrain_identity(transformer, s, a, steps=pretrain_steps, batch_size=batch_size, lr=1e-3)

    g0_frozen = copy.deepcopy(g0).eval()
    g1_frozen = copy.deepcopy(g1).eval()
    g0_frozen.requires_grad_(False)
    g1_frozen.requires_grad_(False)

    with torch.no_grad():
        target_b = g0_frozen(s_t, a_t)

    n = len(s_t)
    bs = min(batch_size, n)
    opt = torch.optim.Adam(transformer.parameters(), lr=lr)
    transformer.train()
    last_loss, last_dyn, last_reg = 0.0, 0.0, 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        a_prime = transformer(s_t[idx], a_t[idx])
        pred_b = g1_frozen(s_t[idx], a_prime)
        dyn_loss = ((pred_b - target_b[idx]) ** 2).sum(-1).mean()
        reg_loss = ((a_prime - a_t[idx]) ** 2).sum(-1).mean()
        loss = dyn_loss + lam * reg_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        last_dyn = float(dyn_loss.item())
        last_reg = float(reg_loss.item())
    transformer.eval()
    return transformer, {"loss": last_loss, "dyn_loss": last_dyn, "reg_loss": last_reg}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="ExORL dataset.npz")
    parser.add_argument("--stock", type=Path, nargs="+", default=None,
                        help="T07 fb_trajectories_seed*.npz files")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained FB checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.dataset is None and args.stock is None:
        parser.error("Provide --stock and/or --dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.monotonic()

    s_parts, a_parts, s2_parts, val_parts, data_hashes = [], [], [], [], {}
    if args.stock is not None:
        stock_data = load_stock_transitions(args.stock)
        s_parts.append(stock_data["s"])
        a_parts.append(stock_data["a"])
        s2_parts.append(stock_data["s2"])
        val_parts.append(stock_data["val_mask"])
        for p in args.stock:
            data_hashes[str(p)] = sha256(p)
    if args.dataset is not None:
        ex_data = load_exorl_transitions(args.dataset)
        n_eps = int(ex_data["episode_id"].max()) + 1
        rng = np.random.default_rng(args.seed)
        val_eps = set(rng.choice(n_eps, size=max(1, int(n_eps * args.val_frac)), replace=False).tolist())
        val_parts.append(np.isin(ex_data["episode_id"], list(val_eps)))
        s_parts.append(ex_data["s"])
        a_parts.append(ex_data["a"])
        s2_parts.append(ex_data["s2"])
        data_hashes[str(args.dataset)] = sha256(args.dataset)

    s = np.concatenate(s_parts)
    a = np.concatenate(a_parts)
    s2 = np.concatenate(s2_parts)
    val_mask = np.concatenate(val_parts)
    train_mask = ~val_mask

    agent = load_agent(args.checkpoint, device)
    agent.eval()
    b_next = compute_backward_features(agent, s2, device).cpu().numpy()

    best_model, final_model, history = train_g0_model(
        s, a, b_next, train_mask, val_mask,
        hidden=args.hidden, layers=args.layers, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(
        args={k: [str(x) for x in v] if isinstance(v, list) else (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()},
        checkpoint_sha256=sha256(args.checkpoint),
        data_sha256=data_hashes,
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        history=history,
    )
    best_model.save(args.out_dir / "g0_best.pt", **meta)
    final_model.save(args.out_dir / "g0_final.pt", **meta)
    with (args.out_dir / "g0_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.out_dir} best val_mse={min(h['val_mse'] for h in history):.6f} "
          f"elapsed {time.monotonic() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
