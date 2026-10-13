"""One-step FB (Zheng et al., arXiv:2602.11399, Eq. 7-9) on the lava gridworld of toy_cql_fb.py.

F(s,a) has no z input and the TD target uses the next action from the dataset, so F^T z_r
should match (1 - gamma) * Q^beta, the behaviour policy's Q, which is computed exactly here.
The policy is greedy in F(s,a)^T z (the tau -> inf limit of the paper's discrete softmax policy).

Usage: python tasks/T03-cql-z-projection/onestep_fb.py --out tasks/T03-cql-z-projection/data/onestep.json
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

import toy_cql_fb
from toy_cql_fb import (
    GOALS, HORIZON, LAVA, NA, SAFE, SIDX, mlp, oracle_returns, rollout_stats, step,
)

STATE_ENC = "xy"


def encode(states):
    if STATE_ENC == "xy":
        return toy_cql_fb.encode(states)
    return np.eye(len(SAFE), dtype=np.float32)[[SIDX[tuple(s)] for s in states]]


def state_dim():
    return 2 if STATE_ENC == "xy" else len(SAFE)


def collect(rng, episodes):
    """Episodes of (s, a, s2, a2) from the uniform policy over lava-free actions."""
    data = []
    for _ in range(episodes):
        s = SAFE[rng.integers(len(SAFE))]
        traj = []
        for _ in range(HORIZON + 1):
            safe_actions = [a for a in range(NA) if step(s, a) not in LAVA]
            a = safe_actions[rng.integers(len(safe_actions))]
            traj.append((s, a))
            s = step(s, a)
        data += [(s, a, s2, a2) for (s, a), (s2, a2) in zip(traj[:-1], traj[1:])]
    return data


def behaviour_q(goal, gamma):
    """Exact Q^beta(s,a) for reward 1[s == goal]. Lava is absorbing with value 0."""
    n = len(SAFE)
    safe_actions = [[a for a in range(NA) if step(s, a) not in LAVA] for s in SAFE]
    p = np.zeros((n, n))  # state transition matrix of the behaviour policy
    for i, s in enumerate(SAFE):
        for a in safe_actions[i]:
            p[i, SIDX[step(s, a)]] += 1 / len(safe_actions[i])
    r = np.array([float(s == goal) for s in SAFE])
    v = np.linalg.solve(np.eye(n) - gamma * p, r)
    q = np.zeros((n, NA))
    for i, s in enumerate(SAFE):
        for a in range(NA):
            s2 = step(s, a)
            q[i, a] = r[i] + (0.0 if s2 in LAVA else gamma * v[SIDX[s2]])
    return q


class OneStepFB(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.F = mlp(state_dim() + NA, d)
        self.B = mlp(state_dim() + NA, d)

    def f(self, s, a1h):
        return self.F(torch.cat([s, a1h], -1))

    def b(self, s, a1h):
        return self.B(torch.cat([s, a1h], -1))


def evaluate(model, data, counts, gamma, d):
    s_all = torch.tensor(encode(SAFE))
    eye = torch.eye(NA).expand(len(SAFE), NA, NA)
    s_rep = s_all[:, None].expand(-1, NA, state_dim())
    s_data = torch.tensor(encode([x[0] for x in data]))
    a_data = torch.eye(NA)[[x[1] for x in data]]
    in_data = torch.tensor(counts > 0)
    with torch.no_grad():
        b_data = model.b(s_data, a_data)
        f_all = model.f(s_rep, eye)  # (S,NA,d)
        cov = b_data.T @ b_data / len(data)
        out = {"orth_err": ((cov - torch.eye(d)).norm() / np.sqrt(d)).item(), "goals": []}
        for goal in GOALS:
            r_data = torch.tensor([float(x[0] == goal) for x in data])
            z = (b_data * r_data[:, None]).mean(0)
            q_hat = (f_all @ z).numpy()
            q_true = (1 - gamma) * behaviour_q(goal, gamma)
            m = in_data.numpy()
            err = q_hat[m] - q_true[m]
            oracle = oracle_returns(goal)
            fb = rollout_stats(q_hat.argmax(1), goal, counts, oracle)
            exact = rollout_stats(q_true.argmax(1), goal, counts, oracle)
            out["goals"].append({
                "goal": list(goal),
                "q_rel_rmse": float(np.sqrt((err**2).mean()) / q_true[m].std()),
                "q_corr": float(np.corrcoef(q_hat[m], q_true[m])[0, 1]),
                "greedy_agree": float((q_hat.argmax(1) == q_true.argmax(1)).mean()),
                "ood_q_minus_true": float((q_hat[~m] - q_true[~m]).mean()),
                **{f"fb_{k}": v for k, v in fb.items()},
                **{f"exact_{k}": v for k, v in exact.items()},
            })
    return out


def train(data, counts, seed, d, steps, batch, gamma, lr, tau, ortho, eval_every):
    torch.manual_seed(seed)
    s = torch.tensor(encode([x[0] for x in data]))
    a = torch.eye(NA)[[x[1] for x in data]]
    s2 = torch.tensor(encode([x[2] for x in data]))
    a2 = torch.eye(NA)[[x[3] for x in data]]
    model, target = OneStepFB(d), OneStepFB(d)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    off = 1 - torch.eye(batch)
    curve = []
    for it in range(steps + 1):
        if it % eval_every == 0:
            ev = evaluate(model, data, counts, gamma, d)
            g = ev["goals"]
            curve.append({
                "step": it,
                "orth_err": ev["orth_err"],
                "q_rel_rmse": float(np.mean([x["q_rel_rmse"] for x in g])),
                "q_corr": float(np.mean([x["q_corr"] for x in g])),
                "fb_score": float(np.mean([x["fb_score"] for x in g])),
            })
            print(f"  seed {seed} {curve[-1]}", flush=True)
        if it == steps:
            break
        idx = torch.randint(len(data), (batch,))
        bs, ba, bs2, ba2 = s[idx], a[idx], s2[idx], a2[idx]
        with torch.no_grad():
            target_m = gamma * target.f(bs2, ba2) @ target.b(bs, ba).T
        f = model.f(bs, ba)
        b = model.b(bs, ba)
        m = f @ b.T
        td = 0.5 * ((m - target_m) ** 2).mean() - (1 - gamma) * m.diag().mean()
        cov = b @ b.T
        orth = (cov * off).pow(2).mean() - 2 * cov.diag().mean()  # Eq. 8
        loss = td + ortho * orth
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, tp in zip(model.parameters(), target.parameters()):
                tp.lerp_(p, tau)
    return model, curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--ortho", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--state_enc", choices=["xy", "onehot"], default="xy")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    global STATE_ENC
    STATE_ENC = args.state_enc

    runs = []
    for seed in args.seeds:
        data = collect(np.random.default_rng(seed), args.episodes)
        counts = np.zeros((len(SAFE), NA))
        for s, a, _, _ in data:
            counts[SIDX[s], a] += 1
        model, curve = train(
            data, counts, seed, args.d, args.steps, args.batch, args.gamma,
            args.lr, args.tau, args.ortho, args.eval_every,
        )
        final = evaluate(model, data, counts, args.gamma, args.d)
        runs.append({"seed": seed, "curve": curve, "final": final})
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
