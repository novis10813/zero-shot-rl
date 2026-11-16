"""Single-step separability of each dynamics shift: a classifier of stock vs shifted transitions.

For each shifted config of the paired one-step data of zero_shot_rl.eval.paired_step (the same
(s, a) replayed in the stock and in a shifted simulator), a ReLU MLP is trained to tell the stock
next state from the shifted one. Its input is [s, a, s' - s], standardised with the statistics of
the stock training transitions, and its label is 0 for `next_baseline`, 1 for `next_<config>`.
The classifier sees shifted transitions in training, so its AUROC is an upper reference for a
single-step detector that is fitted on stock data only, not a deployable detector.

Episodes, not transitions, are split: --test_episodes random episodes of the pooled files
(seed --seed) are held out. One classifier per config, trained for --epochs epochs (no early
stopping). Writes <out> (NPZ) with

    test_episodes            [E]                   indices of the held-out episodes in the pooled files
    logit_base_<config>      [E, steps]            classifier logits of the stock test transitions
    logit_shift_<config>     [E, steps]            logits of the shifted test transitions
    gap_<config>             [E, steps]            |s'_config - s'_stock|_2 of the test transitions
    curve_<config>           [epochs/10, 3]        epoch, train AUROC (first 200k samples), test AUROC

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.classifier \
        --paired <paired_s0.npz> <paired_s1.npz> <paired_s2.npz> --out <classifier.npz>
"""

from argparse import ArgumentParser

import numpy as np
import torch

from analysis.detection import auroc

CONFIGS = [
    "friction_mild", "friction_strong", "actuator_mild", "actuator_strong",
    "friction_high_mild", "friction_high_strong", "actuator_high_mild", "actuator_high_strong",
]


def features(s, a, s_next):
    """[s, a, s' - s], float64 [..., d_s + d_a + d_s]; the difference is taken before any cast."""
    return np.concatenate([s, a.astype(np.float64), s_next - s], -1)


def mlp(d_in, hidden, layers):
    dims = [d_in] + [hidden] * layers
    body = []
    for i, o in zip(dims[:-1], dims[1:]):
        body += [torch.nn.Linear(i, o), torch.nn.ReLU()]
    return torch.nn.Sequential(*body, torch.nn.Linear(hidden, 1))


@torch.no_grad()
def logits(net, x, device, batch=65536):
    return np.concatenate([net(x[i:i + batch].to(device)).squeeze(-1).cpu().numpy() for i in range(0, len(x), batch)])


def train_config(xb, xs, train, args, device):
    """Train one classifier; returns test logits (stock, shifted) and the learning curve."""
    d = xb.shape[-1]
    mean = xb[train].reshape(-1, d).mean(0)
    std = xb[train].reshape(-1, d).std(0) + 1e-6
    to_t = lambda x: torch.as_tensor(((x.reshape(-1, d) - mean) / std), dtype=torch.float32)  # noqa: E731
    x_tr = torch.cat([to_t(xb[train]), to_t(xs[train])])
    n = len(x_tr) // 2
    y_tr = torch.cat([torch.zeros(n), torch.ones(n)])
    xb_te, xs_te = to_t(xb[~train]), to_t(xs[~train])

    torch.manual_seed(args.seed)
    net = mlp(d, args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    sub = torch.randperm(len(x_tr), generator=gen)[:200000]
    curve = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(x_tr), args.batch):
            j = perm[i:i + args.batch]
            out = net(x_tr[j].to(device)).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y_tr[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            net.eval()
            p_tr = logits(net, x_tr[sub], device)
            y_sub = y_tr[sub].numpy()
            lb, ls = logits(net, xb_te, device), logits(net, xs_te, device)
            curve.append([epoch, auroc(p_tr[y_sub == 0], p_tr[y_sub == 1]), auroc(lb, ls)])
            print(f"  epoch {epoch}: train AUROC {curve[-1][1]:.4f}, test AUROC {curve[-1][2]:.4f}", flush=True)
    return lb, ls, np.array(curve)


def main():
    parser = ArgumentParser()
    parser.add_argument("--paired", nargs="+", required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    parser.add_argument("--test_episodes", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    files = [np.load(p) for p in args.paired]
    cat = lambda k: np.concatenate([f[k] for f in files])  # noqa: E731
    s, a, base = cat("observations"), cat("actions"), cat("next_baseline")
    episodes, steps = s.shape[:2]
    test = np.random.default_rng(args.seed).choice(episodes, args.test_episodes, replace=False)
    train = np.ones(episodes, bool)
    train[test] = False
    xb = features(s, a, base)

    out = {"test_episodes": np.sort(test)}
    for name in args.configs:
        print(f"config {name}", flush=True)
        shifted = cat("next_" + name)
        lb, ls, curve = train_config(xb, features(s, a, shifted), train, args, device)
        out["logit_base_" + name] = lb.reshape(-1, steps)
        out["logit_shift_" + name] = ls.reshape(-1, steps)
        out["gap_" + name] = np.linalg.norm(shifted[~train] - base[~train], axis=-1)
        out["curve_" + name] = curve
    np.savez_compressed(args.out, **out)


if __name__ == "__main__":
    main()
