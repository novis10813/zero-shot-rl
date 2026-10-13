"""Toy check: project the CQL implicit reward onto a state-action FB model at test time.

Gridworld with a lava strip. The behaviour policy never steps into lava, so the
out-of-distribution (OOD) actions are exactly the lava-entering ones.

For each alpha, solve  z* = z_r + dz(z*)  with
    dz(z) = -alpha * (E_{s~D, a~softmax(Q_z(s,.))}[B(s,a)] - E_{(s,a)~D}[B(s,a)])
and roll out the greedy policy of F(s, a, z*)^T z*. Tabular CQL with the same
alpha is the reference.

Usage: python tasks/T03-cql-z-projection/toy_cql_fb.py --out tasks/T03-cql-z-projection/data/toy.json
"""

import argparse
import json
from collections import deque

import numpy as np
import torch
import torch.nn as nn

ROWS, COLS = 5, 7
LAVA = {(1, c) for c in range(1, 6)}
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # stay up down left right
NA = len(MOVES)
SAFE = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in LAVA]
SIDX = {s: i for i, s in enumerate(SAFE)}
GOALS = [(2, 3), (0, 3), (3, 1), (4, 5)]
HORIZON = 40


def step(s, a):
    if s in LAVA:
        return s
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return s
    return (r, c)


def encode(states):
    s = np.asarray(states, dtype=np.float32)
    return np.stack([2 * s[:, 0] / (ROWS - 1) - 1, 2 * s[:, 1] / (COLS - 1) - 1], 1)


def collect(rng, episodes):
    data = []
    for _ in range(episodes):
        s = SAFE[rng.integers(len(SAFE))]
        for _ in range(HORIZON):
            safe_actions = [a for a in range(NA) if step(s, a) not in LAVA]
            a = safe_actions[rng.integers(len(safe_actions))]
            s2 = step(s, a)
            data.append((s, a, s2))
            s = s2
    return data


def mlp(i, o, h=256):
    return nn.Sequential(
        nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)
    )


class FB(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.F = mlp(2 + NA + d, d)
        self.B = mlp(2 + NA, d)

    def f(self, s, a1h, z):
        return self.F(torch.cat([s, a1h, z], -1))

    def b(self, s, a1h):
        return self.B(torch.cat([s, a1h], -1))

    def f_all(self, s, z):
        """F for every action. s: (n,2), z: (n,d) -> (n,NA,d)."""
        n = s.shape[0]
        eye = torch.eye(NA).expand(n, NA, NA)
        return self.f(
            s[:, None].expand(n, NA, 2), eye, z[:, None].expand(n, NA, self.d)
        )


def unit(z, d):
    return np.sqrt(d) * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def train_fb(data, seed, d, steps, batch, gamma, lr, tau):
    torch.manual_seed(seed)
    s = torch.tensor(encode([x[0] for x in data]))
    a1h = torch.eye(NA)[[x[1] for x in data]]
    s2 = torch.tensor(encode([x[2] for x in data]))
    model, target = FB(d), FB(d)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    off = 1 - torch.eye(batch)
    for it in range(steps):
        idx = torch.randint(len(data), (batch,))
        bs, ba, bs2 = s[idx], a1h[idx], s2[idx]
        with torch.no_grad():
            z = unit(torch.randn(batch, d), d)
            jdx = torch.randint(len(data), (batch,))
            zb = unit(model.b(s[jdx], a1h[jdx]), d)
            mix = torch.rand(batch, 1) < 0.5
            z = torch.where(mix, zb, z)
            fn = target.f_all(bs2, z)  # (b,NA,d)
            an = torch.einsum("bad,bd->ba", fn, z).argmax(1)
            fn = fn[torch.arange(batch), an]
            target_m = gamma * fn @ target.b(bs, ba).T
        f = model.f(bs, ba, z)
        b = model.b(bs, ba)
        m = f @ b.T
        fb_loss = 0.5 * ((m - target_m) ** 2).mean() - m.diag().mean()
        cov = b @ b.T
        orth = 0.5 * ((cov * off) ** 2).mean() - cov.diag().mean()
        loss = fb_loss + orth
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, tp in zip(model.parameters(), target.parameters()):
                tp.lerp_(p, tau)
        if it % 5000 == 0:
            print(f"  seed {seed} step {it} fb {fb_loss.item():.3f} orth {orth.item():.3f}")
    return model


def rollout_stats(policy, goal, counts, oracle):
    """policy: (len(SAFE),) greedy actions. Returns score, lava rate, OOD action rate."""
    rets, lava, ood, steps = [], 0, 0, 0
    starts = [s for s in SAFE if s != goal]
    for s0 in starts:
        s, ret = s0, 0
        for _ in range(HORIZON):
            if s in LAVA:
                break
            ret += s == goal
            a = policy[SIDX[s]]
            ood += counts[SIDX[s], a] == 0
            steps += 1
            s = step(s, a)
        lava += s in LAVA
        rets.append(ret)
    return {
        "score": float(np.mean(rets) / np.mean([oracle[x] for x in starts])),
        "lava_rate": lava / len(starts),
        "ood_action_rate": ood / max(steps, 1),
    }


def oracle_returns(goal):
    dist = {goal: 0}
    q = deque([goal])
    while q:
        x = q.popleft()
        for a in range(NA):
            for y in SAFE:
                if y not in dist and step(y, a) == x:
                    dist[y] = dist[x] + 1
                    q.append(y)
    return {s: max(HORIZON - dist[s], 0) for s in SAFE}


def solve_z(model, z_r, alpha, s_data, b_data, s_all, b_all, sidx_data, iters, eta):
    """Damped fixed-point iteration z <- (1-eta) z + eta (z_r + dz(z))."""
    d = model.d
    b_bar = b_data.mean(0)
    z = z_r.clone()
    resid = 0.0
    for _ in range(iters):
        q = torch.einsum("sad,d->sa", model.f_all(s_all, unit(z, d).expand(len(s_all), d)), z)
        pi = torch.softmax(q, 1)  # (S,NA)
        eb = torch.einsum("sa,sad->sd", pi, b_all)[sidx_data].mean(0)
        new = z_r - alpha * (eb - b_bar)
        resid = ((new - z).norm() / z_r.norm()).item()
        z = (1 - eta) * z + eta * new
    return z, resid


def tabular_cql(data, goal, alpha, gamma, outer, inner, lr):
    counts = np.zeros((len(SAFE), NA))
    for s, a, _ in data:
        counts[SIDX[s], a] += 1
    n_s = counts.sum(1, keepdims=True)
    beta = torch.tensor(counts / np.maximum(n_s, 1), dtype=torch.float32)
    dstate = torch.tensor(n_s[:, 0] / n_s.sum(), dtype=torch.float32)
    nxt = np.array(
        [[SIDX.get(step(s, a), 0) for a in range(NA)] for s in SAFE]
    )  # lava next states only occur for OOD pairs, which have zero weight
    r = torch.tensor([[float(s == goal)] * NA for s in SAFE])
    q = torch.zeros(len(SAFE), NA, requires_grad=True)
    opt = torch.optim.Adam([q], lr=lr)
    for _ in range(outer):
        with torch.no_grad():
            y = r + gamma * q.max(1).values[torch.tensor(nxt)]
        for _ in range(inner):
            td = 0.5 * (dstate[:, None] * beta * (q - y) ** 2).sum()
            reg = (dstate * (torch.logsumexp(q, 1) - (beta * q).sum(1))).sum()
            loss = td + alpha * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
    return q.detach().argmax(1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 0.3, 1, 3, 10, 30])
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--fp_iters", type=int, default=400)
    p.add_argument("--fp_eta", type=float, default=0.1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    s_all = torch.tensor(encode(SAFE))
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        data = collect(rng, args.episodes)
        counts = np.zeros((len(SAFE), NA))
        for s, a, _ in data:
            counts[SIDX[s], a] += 1
        ood_mask = torch.tensor(counts == 0)
        model = train_fb(
            data, seed, args.d, args.steps, args.batch, args.gamma, args.lr, args.tau
        )
        model.eval()
        with torch.no_grad():
            s_data = torch.tensor(encode([x[0] for x in data]))
            a_data = torch.eye(NA)[[x[1] for x in data]]
            b_data = model.b(s_data, a_data)
            b_all = model.b(
                s_all[:, None].expand(-1, NA, 2), torch.eye(NA).expand(len(SAFE), NA, NA)
            )
            sidx_data = torch.tensor([SIDX[x[0]] for x in data])
            cov = b_data.T @ b_data / len(data)
            orth_err = ((cov - torch.eye(args.d)).norm() / np.sqrt(args.d)).item()
            for goal in GOALS:
                oracle = oracle_returns(goal)
                r_data = torch.tensor([float(x[0] == goal) for x in data])
                z_r = (b_data * r_data[:, None]).mean(0)
                for alpha in args.alphas:
                    z, resid = solve_z(
                        model, z_r, alpha, s_data, b_data, s_all, b_all, sidx_data,
                        args.fp_iters, args.fp_eta,
                    )
                    q = torch.einsum(
                        "sad,d->sa", model.f_all(s_all, unit(z, args.d).expand(len(SAFE), args.d)), z
                    )
                    pen = torch.einsum("sad,d->sa", b_all, z - z_r)
                    fb = rollout_stats(q.argmax(1).numpy(), goal, counts, oracle)
                    rows.append({
                        "method": "fb_proj", "seed": seed, "goal": list(goal), "alpha": alpha,
                        **fb,
                        "fp_residual": resid,
                        "orth_err": orth_err,
                        "cos_z_zr": torch.cosine_similarity(z, z_r, 0).item(),
                        "pen_ood": pen[ood_mask].mean().item(),
                        "pen_id": pen[~ood_mask].mean().item(),
                    })
                    with torch.enable_grad():
                        pol = tabular_cql(data, goal, alpha, args.gamma, 150, 30, 0.1)
                    rows.append({
                        "method": "tab_cql", "seed": seed, "goal": list(goal), "alpha": alpha,
                        **rollout_stats(pol, goal, counts, oracle),
                    })
                    print(f"seed {seed} goal {goal} a {alpha}: fb {fb} resid {resid:.3g} | cql {rows[-1]['score']:.2f}")
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
