"""Pack the output of run_cvae.sh into data/inputs (the files the analysis reads).

For every `cvae_scored_<run>.npz` in the given directories it writes data/inputs/cvae_<run>.npz
(float32, compressed: min_<config>, mean_<config>, mse_<config>, val_member, var_delta, episode,
seed) and collects the training diagnostics of `cvae_<run>.pt` into data/inputs/fit_info.json.
The state gap |s'_config - s'_stock|, the same for all runs, goes to data/inputs/shift_gap.npz.
The per-member errors `member_*` are not kept: the analysis needs their mean and minimum only.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.2.2-conditioned-vae uv run python -m analyze.pack <dir> [<dir> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
KEEP = ("min_", "mean_", "mse_")
SINGLE = ("val_member", "var_delta", "episode", "seed")


def main():
    out = TASK / "data/inputs"
    out.mkdir(parents=True, exist_ok=True)
    fit_path = out / "fit_info.json"
    fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
    gap = None
    for directory in map(Path, sys.argv[1:]):
        for scored in sorted(directory.glob("cvae_scored_*.npz")):
            run = scored.stem[len("cvae_scored_"):]
            data = np.load(scored)
            keep = {k: data[k].astype(np.float32) if data[k].dtype.kind == "f" else data[k]
                    for k in data.files if k.startswith(KEEP) or k in SINGLE}
            np.savez_compressed(out / f"cvae_{run}.npz", **keep)
            current = {k: data[k].astype(np.float32) for k in data.files if k.startswith("gap_")}
            if gap is None:
                gap = current
            elif any(not np.allclose(gap[k], current[k]) for k in gap):
                raise ValueError(f"{run}: state gaps differ from the other runs")
            meta = torch.load(directory / f"cvae_{run}.pt", map_location="cpu", weights_only=False)["meta"]
            fit[run] = [{"best_epoch": f["best_epoch"], "val_rec": f["val_rec"], "val_kl": f["val_kl"],
                         "sigma": f.get("sigma"), "active_dims": int((f["val_kl_per_dim"] > 0.01).sum())}
                        for f in meta["fit_info"]]
            print(run)
    np.savez_compressed(out / "shift_gap.npz", **gap)
    fit_path.write_text(json.dumps(fit, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
