"""Dynamics context encoder and context-conditioned backward-feature predictor.

An encoder E maps a segment of H transitions (s, a, s' - s) of one environment to a context
c = E(segment) on the unit sphere, and a predictor g regresses the backward feature of the next
state of a frozen FB model under the dynamics the context came from:

    g(s, a, c) ~ E_P[B_0(s') | s, a]

Training data are episodes under many dynamics (zero_shot_rl.data.multi_dynamics) and stock
episodes (zero_shot_rl.eval.fb_trajectories, the stock dynamics as one more set). Each step draws
`batch` distinct dynamics sets; for each, two different episodes e1, e2, one segment of each
(length H, the same for the whole batch, uniform in 1..max_h) and `queries` transitions of each.
The loss is

    |g(s_q2, a_q2, c1) - B_0(s'_q2)|^2 / 2 + |g(s_q1, a_q1, c2) - B_0(s'_q1)|^2 / 2
        + lam * InfoNCE(c1, c2)

c1 predicts the transitions of the other episode, so it has to carry the dynamics, not the
segment. InfoNCE has logits c1_i . c2_j / tau, the positive pair is the same set, and is taken
in both directions. --encoder transformer: tokens are the standardised transitions, a learned
CLS token, no positional encoding (the order of transitions carries no information), and c is the
CLS output; --encoder mean: a per-transition MLP and the mean over the segment.

Every set whose index is congruent to 9 mod 10 and the stock episodes with environment seed >= 40
are validation data. The best (lowest validation loss) and the final model are saved with the
history of the losses.

Usage (repo root):
    PYTHONPATH=. uv run python -m zero_shot_rl.latent.context_encoder \
        --data multi_dynamics_seed0.npz multi_dynamics_seed1.npz multi_dynamics_seed2.npz \
        --stock fb_trajectories_seed0.npz fb_trajectories_seed1.npz fb_trajectories_seed2.npz \
        --checkpoint <fb.pt> --encoder transformer --out-dir <dir> [--steps 100000]
"""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.base import load_agent
from zero_shot_rl.provenance import sha256

VAL_SET_MOD = 10
VAL_STOCK_SEED = 40


def mlp(dims, out):
    layers = []
    for i, o in zip(dims[:-1], dims[1:]):
        layers += [torch.nn.Linear(i, o), torch.nn.ReLU()]
    return torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], out))


class ContextModel(torch.nn.Module):
    """Encoder E (segment -> unit context) and predictor g (s, a, c -> B_0(s'))."""

    def __init__(self, obs_dim, action_dim, feature_dim, context_dim=16, encoder="transformer",
                 width=128, layers=2, heads=4, hidden=512, g_layers=3):
        super().__init__()
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, feature_dim=feature_dim,
                           context_dim=context_dim, encoder=encoder, width=width, layers=layers,
                           heads=heads, hidden=hidden, g_layers=g_layers)
        token = 2 * obs_dim + action_dim
        if encoder == "transformer":
            self.embed = torch.nn.Linear(token, width)
            self.cls = torch.nn.Parameter(torch.zeros(1, 1, width))
            layer = torch.nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True)
            self.body = torch.nn.TransformerEncoder(layer, layers)
            self.head = torch.nn.Linear(width, context_dim)
        elif encoder == "mean":
            self.embed = mlp([token, 2 * width], 2 * width)
            self.head = torch.nn.Linear(2 * width, context_dim)
        else:
            raise ValueError("encoder must be transformer or mean")
        self.encoder = encoder
        self.g = mlp([obs_dim + action_dim + context_dim] + [hidden] * g_layers, feature_dim)
        self.register_buffer("x_mean", torch.zeros(token))
        self.register_buffer("x_std", torch.ones(token))

    def encode(self, s, a, delta):
        """Segments [B, H, .] of (s, a, s' - s) -> contexts [B, context_dim], unit norm."""
        x = (torch.cat([s, a, delta], -1) - self.x_mean) / self.x_std
        if self.encoder == "transformer":
            tokens = torch.cat([self.cls.expand(len(x), -1, -1), self.embed(x)], 1)
            h = self.body(tokens)[:, 0]
        else:
            h = torch.relu(self.embed(x)).mean(1)
        return F.normalize(self.head(h), dim=-1)

    def predict(self, s, a, c):
        """g(s, a, c); c broadcasts over the leading axes of s and a."""
        d = s.shape[-1] + a.shape[-1]
        x = (torch.cat([s, a], -1) - self.x_mean[:d]) / self.x_std[:d]
        return self.g(torch.cat([x, c.expand(*x.shape[:-1], -1)], -1))

    def save(self, path, **meta):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)


def load_context_model(path, device=torch.device("cpu")):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = ContextModel(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob["meta"]


def load_episodes(data_paths, stock_paths):
    """Transitions of all episodes, flat, and per episode: start, length, set, validation flag.

    Sets of the multi-dynamics files keep their index; the stock dynamics get index n_sets.
    """
    s, a, s2, start, length, sets, val = [], [], [], [], [], [], []
    n, n_sets = 0, None

    def add(obs, act, set_index, is_val):
        nonlocal n
        s.append(obs[:-1])
        a.append(act[:len(obs) - 1])
        s2.append(obs[1:])
        start.append(n)
        length.append(len(obs) - 1)
        sets.append(set_index)
        val.append(is_val)
        n += len(obs) - 1

    for path in data_paths:
        with np.load(path) as f:
            n_sets = len(f["friction"])
            for obs, act, k in zip(f["observations"], f["actions"], f["set"]):
                add(obs, act, int(k), int(k) % VAL_SET_MOD == VAL_SET_MOD - 1)
    if n_sets is None:
        raise ValueError("need at least one multi-dynamics file")
    for path in stock_paths:
        with np.load(path) as f:
            obs, act, episodes = f["observations"], f["actions"], f["episodes"]
            for t in range(obs.shape[0]):
                for e in range(obs.shape[1]):
                    add(obs[t, e], act[t, e], n_sets, int(episodes[e]) >= VAL_STOCK_SEED)
    cat = lambda x: np.concatenate(x).astype(np.float32)  # noqa: E731
    return dict(s=cat(s), a=cat(a), s2=cat(s2), start=np.array(start), length=np.array(length),
                set=np.array(sets), val=np.array(val), n_sets=n_sets + 1)


@torch.no_grad()
def backward_features(agent, s2, device, chunk=65536):
    out = []
    for i in range(0, len(s2), chunk):
        x = torch.as_tensor(s2[i:i + chunk], device=device)
        out.append(agent.FB.backward_representation(x).float())
    return torch.cat(out)


class Sampler:
    """Batches of (segment 1, segment 2, queries 1, queries 2) indices for distinct sets."""

    def __init__(self, episodes, mask, device):
        self.start = torch.as_tensor(episodes["start"][mask], device=device)
        self.length = torch.as_tensor(episodes["length"][mask], device=device)
        sets = episodes["set"][mask]
        self.sets = np.unique(sets)
        groups = [np.flatnonzero(sets == k) for k in self.sets]
        if min(len(g) for g in groups) < 2:
            raise ValueError("every set needs two episodes")
        width = max(len(g) for g in groups)
        table = np.zeros((len(groups), width), dtype=np.int64)
        for i, g in enumerate(groups):
            table[i, :len(g)] = g
        self.table = torch.as_tensor(table, device=device)
        self.count = torch.as_tensor([len(g) for g in groups], device=device)
        self.device = device

    def draw(self, batch, h, queries, generator):
        dev = self.device
        rand = lambda *shape: torch.rand(*shape, device=dev, generator=generator)  # noqa: E731
        k = torch.randperm(len(self.sets), device=dev, generator=generator)[:batch]
        count = self.count[k]
        r1 = (rand(len(k)) * count).long()
        r2 = (r1 + 1 + (rand(len(k)) * (count - 1)).long()) % count
        e1, e2 = self.table[k, r1], self.table[k, r2]
        out = []
        for e in (e1, e2):
            start, length = self.start[e], self.length[e]
            first = (rand(len(e)) * (length - h + 1)).long()
            segment = start[:, None] + first[:, None] + torch.arange(h, device=dev)
            query = start[:, None] + (rand(len(e), queries) * length[:, None]).long()
            out.append((segment, query))
        return out


def losses(model, data, draw, tau):
    """Prediction MSE (mean over feature dims), InfoNCE and its top-1 accuracy for one draw."""
    (seg1, q1), (seg2, q2) = draw
    s, a, delta, b = data
    c1 = model.encode(s[seg1], a[seg1], delta[seg1])
    c2 = model.encode(s[seg2], a[seg2], delta[seg2])
    pred = 0.5 * (((model.predict(s[q2], a[q2], c1[:, None]) - b[q2]) ** 2).mean()
                  + ((model.predict(s[q1], a[q1], c2[:, None]) - b[q1]) ** 2).mean())
    logits = c1 @ c2.T / tau
    target = torch.arange(len(c1), device=c1.device)
    nce = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    acc = (logits.argmax(1) == target).float().mean()
    return pred, nce, acc


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--stock", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=["transformer", "mean"], default="transformer")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--max-h", type=int, default=32)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    name = args.name or f"context_{args.encoder}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: args.out_dir / f"{name}_{k}.pt" for k in ("best", "final")}
    history_path = args.out_dir / f"{name}_history.json"
    for p in [*paths.values(), history_path]:
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")

    episodes = load_episodes(args.data, args.stock)
    agent = load_agent(args.checkpoint, device)
    agent.eval()
    t = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
    s, a = t(episodes["s"]), t(episodes["a"])
    delta = t((episodes["s2"].astype(np.float64) - episodes["s"]).astype(np.float32))
    b = backward_features(agent, episodes["s2"], device)
    data = (s, a, delta, b)
    obs_dim, action_dim, feature_dim = s.shape[1], a.shape[1], b.shape[1]

    val_ep = episodes["val"]
    train_rows = torch.cat([torch.arange(st, st + ln) for st, ln, v in
                            zip(episodes["start"], episodes["length"], val_ep) if not v]).to(device)
    x = torch.cat([s, a, delta], -1)[train_rows]
    model = ContextModel(obs_dim, action_dim, feature_dim, args.context_dim, args.encoder).to(device)
    model.x_mean.copy_(x.mean(0))
    model.x_std.copy_(x.std(0).clamp_min(1e-6))
    del x, train_rows
    train, val = Sampler(episodes, ~val_ep, device), Sampler(episodes, val_ep, device)
    print(f"transitions {len(s)}, train sets {len(train.sets)}, val sets {len(val.sets)}, "
          f"B dim {feature_dim}, mean |B(s')| {b.norm(dim=1).mean():.3f}", flush=True)

    # Fixed validation draws: H in {1, 4, 16, 32} (capped at max_h), five draws each.
    g_val = torch.Generator(device=device).manual_seed(12345)
    val_hs = sorted({min(h, args.max_h) for h in (1, 4, 16, 32)})
    val_draws = [(h, val.draw(args.batch, h, args.queries, g_val)) for h in val_hs for _ in range(5)]
    g_train = torch.Generator(device=device).manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best = [], float("inf")
    meta = dict(vars(args), checkpoint_sha256=sha256(args.checkpoint),
                data_sha256=[sha256(p) for p in args.data], stock_sha256=[sha256(p) for p in args.stock])
    meta = {k: [str(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], Path)
            else str(v) if isinstance(v, Path) else v for k, v in meta.items()}
    start = time.monotonic()
    run = dict(pred=0.0, nce=0.0, acc=0.0, n=0)
    for step in range(1, args.steps + 1):
        model.train()
        h = int(torch.randint(1, args.max_h + 1, (1,), generator=torch.Generator().manual_seed(args.seed * 10**7 + step)))
        pred, nce, acc = losses(model, data, train.draw(args.batch, h, args.queries, g_train), args.tau)
        loss = pred + args.lam * nce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run.update(pred=run["pred"] + pred.item(), nce=run["nce"] + nce.item(), acc=run["acc"] + acc.item(), n=run["n"] + 1)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            record = dict(step=step, train_pred=run["pred"] / run["n"], train_nce=run["nce"] / run["n"],
                          train_acc=run["acc"] / run["n"])
            with torch.no_grad():
                for vh in val_hs:
                    vals = [losses(model, data, d, args.tau) for hh, d in val_draws if hh == vh]
                    record[f"val_pred_h{vh}"] = float(np.mean([v[0].item() for v in vals]))
                    record[f"val_nce_h{vh}"] = float(np.mean([v[1].item() for v in vals]))
                    record[f"val_acc_h{vh}"] = float(np.mean([v[2].item() for v in vals]))
            val_loss = np.mean([record[f"val_pred_h{vh}"] + args.lam * record[f"val_nce_h{vh}"] for vh in val_hs])
            record["val_loss"] = float(val_loss)
            history.append(record)
            run = dict(pred=0.0, nce=0.0, acc=0.0, n=0)
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()})
                  + f" elapsed {time.monotonic() - start:.0f}s", flush=True)
            if val_loss < best:
                best = val_loss
                model.save(paths["best"], step=step, **meta)
    model.save(paths["final"], step=args.steps, **meta)
    history_path.write_text(json.dumps(history, indent=1))
    for p in paths.values():
        print(f"{p} sha256 {sha256(p)}", flush=True)


if __name__ == "__main__":
    main()
