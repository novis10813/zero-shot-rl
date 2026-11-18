"""Two-predictor Action Transformation (FB-AA) without modifying the FB backbone.

Given a frozen pretrained FB agent with backward representation B_0:
    1. g_0(s, a) ~ E_{P_0}[B_0(s') | s, a] is trained offline on stock data (e.g. ExORL rnd).
    2. At test time in shifted dynamics P_1, g_1(s, a) is fine-tuned from g_0 on test-time
       trajectories collected in P_1 (using separate finetune seeds).
    3. Action transformation T_theta(s, a) = tanh(MLP(s, a)) in (-1, 1)^{d_a} is trained to
       minimize:
           |g_1(s, T_theta(s, a)) - g_0(s, a)|^2 + lam * |T_theta(s, a) - a|^2

Usage (train g_0 offline from ExORL dataset via zero_shot_rl.train.g0):
    PYTHONPATH=. uv run python -m zero_shot_rl.train.g0 \
        --dataset datasets/walker/rnd/dataset.npz \
        --checkpoint <fb.pt> --out-dir <dir>
"""

import copy

import numpy as np
import torch


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
    from zero_shot_rl.train.g0 import main as train_g0_main

    train_g0_main()


if __name__ == "__main__":
    main()
