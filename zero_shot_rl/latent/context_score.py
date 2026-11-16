"""Score test episodes with a context model of zero_shot_rl.latent.context_encoder.

Windows of H transitions start at steps 0, 50, ..., 850 of every episode, for every H in --hs.
For each window the context c = E(window) is compared with the stock context

    c0_H = normalise(mean of E(window) over the same windows of the --stock episodes with
           environment seed < 40, the stock training episodes of the model)

and the predictor is tested on the `queries` transitions that follow the window (steps
start + H .. start + H + queries - 1): the mean over them of |g(s, a, c) - B_0(s')|^2 (with the
window's context) and of |g(s, a, c0_H) - B_0(s')|^2 (with the stock context). Writes --out:

    hs [nH], starts [W]
    context  [E, W, nH, d_c]   float32
    dist     [E, W, nH]        |c - c0_H|_2
    err_ctx, err_stock [E, W, nH]
    c0 [nH, d_c]
    set, task, env_seed, policy_seed [E]   from the test files (policy_seed: index of the file)

Usage (repo root):
    PYTHONPATH=. uv run python -m zero_shot_rl.latent.context_score --model <context.pt> \
        --checkpoint <fb.pt> --test test_seed0.npz test_seed1.npz test_seed2.npz \
        --stock fb_trajectories_seed0.npz ... --out <scores.npz>
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.base import load_agent
from zero_shot_rl.latent.context_encoder import VAL_STOCK_SEED, load_context_model
from zero_shot_rl.provenance import no_clobber

STARTS = np.arange(0, 851, 50)


@torch.no_grad()
def contexts(model, obs, act, starts, h, device, chunk=4096):
    """E(window) for every episode and start: [E, W, d_c]."""
    idx = starts[:, None] + np.arange(h)  # [W, H]
    out, n = [], max(1, chunk // len(starts))
    for i in range(0, len(obs), n):
        o = torch.as_tensor(obs[i:i + n], device=device)
        a = torch.as_tensor(act[i:i + n], device=device)
        s, s2, aa = o[:, idx], o[:, idx + 1], a[:, idx]  # [e, W, H, .]
        e, w = s.shape[:2]
        delta = (s2.double() - s.double()).float()
        c = model.encode(s.reshape(e * w, h, -1), aa.reshape(e * w, h, -1), delta.reshape(e * w, h, -1))
        out.append(c.reshape(e, w, -1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def prediction_errors(model, agent, obs, act, starts, h, c, c0, queries, device, chunk=64):
    """Mean squared error of g over the queries after each window, with c [E, W, d_c] and with c0."""
    idx = starts[:, None] + h + np.arange(queries)  # [W, Q]
    err_ctx, err_stock = [], []
    for i in range(0, len(obs), chunk):
        o = torch.as_tensor(obs[i:i + chunk], device=device)
        a = torch.as_tensor(act[i:i + chunk], device=device)
        s, s2, aa = o[:, idx], o[:, idx + 1], a[:, idx]  # [e, W, Q, .]
        target = agent.FB.backward_representation(s2.reshape(-1, s2.shape[-1])).float().reshape(*s2.shape[:3], -1)
        cc = torch.as_tensor(c[i:i + chunk], device=device)[:, :, None]
        err_ctx.append(((model.predict(s, aa, cc) - target) ** 2).sum(-1).mean(-1).cpu())
        c0t = torch.as_tensor(c0, device=device).expand(*s.shape[:3], -1)
        err_stock.append(((model.predict(s, aa, c0t) - target) ** 2).sum(-1).mean(-1).cpu())
    return torch.cat(err_ctx).numpy(), torch.cat(err_stock).numpy()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", nargs="+", type=Path, required=True)
    parser.add_argument("--stock", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hs", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    if STARTS[-1] + max(args.hs) + args.queries > 1000:
        parser.error("windows and queries must end within the episode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_context_model(args.model, device)
    agent = load_agent(args.checkpoint, device)
    agent.eval()

    tests = [np.load(p) for p in args.test]
    obs = np.concatenate([f["observations"] for f in tests])
    act = np.concatenate([f["actions"] for f in tests])
    meta = {k: np.concatenate([f[k] for f in tests]) for k in ("set", "task", "env_seed")}
    meta["policy_seed"] = np.concatenate([np.full(len(f["set"]), i) for i, f in enumerate(tests)])
    stock_obs, stock_act = [], []
    for p in args.stock:
        with np.load(p) as f:
            keep = f["episodes"] < VAL_STOCK_SEED
            stock_obs.append(f["observations"][:, keep].reshape(-1, *f["observations"].shape[2:]))
            stock_act.append(f["actions"][:, keep].reshape(-1, *f["actions"].shape[2:]))
    stock_obs, stock_act = np.concatenate(stock_obs), np.concatenate(stock_act)

    ctx, dist, err_ctx, err_stock, c0s = [], [], [], [], []
    for h in args.hs:
        c0 = F.normalize(torch.as_tensor(contexts(model, stock_obs, stock_act, STARTS, h, device)).reshape(-1, model.config["context_dim"]).mean(0), dim=0).numpy()
        c = contexts(model, obs, act, STARTS, h, device)
        e_ctx, e_stock = prediction_errors(model, agent, obs, act, STARTS, h, c, c0, args.queries, device)
        ctx.append(c)
        dist.append(np.linalg.norm(c - c0, axis=-1))
        err_ctx.append(e_ctx)
        err_stock.append(e_stock)
        c0s.append(c0)
        print(f"H {h}: median dist {np.median(dist[-1]):.4f}, mean err ctx {e_ctx.mean():.4f}, stock {e_stock.mean():.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(args.out) as temporary:
        np.savez_compressed(temporary, hs=np.array(args.hs), starts=STARTS, context=np.stack(ctx, 2).astype(np.float32),
                            dist=np.stack(dist, 2), err_ctx=np.stack(err_ctx, 2), err_stock=np.stack(err_stock, 2),
                            c0=np.stack(c0s), **meta)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
