"""Train offline g_0(s, a) -> B_0(s') backward-feature predictor on stock data.

Usage (repo root):
    PYTHONPATH=. uv run python -m zero_shot_rl.train.g0 \
        --dataset datasets/walker/rnd/dataset.npz \
        --checkpoint <fb.pt> --out-dir <dir>
"""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from zero_shot_rl.latent.action_transform import (
    compute_backward_features,
    load_exorl_transitions,
    load_stock_transitions,
    train_g0_model,
)
from zero_shot_rl.provenance import sha256


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
