"""Forward-ensemble analysis of the probe NPZs written by probe.py --ensemble.

The members F_k of ensemble.py give, at every probed (s, a) and z_r, the values
q_ens_k = F_k^T z_hat_r and c_ens_k = F_k^T u_1 (exact: 1/(1-gamma)). With
std_ens = std_k q_ens_k, q_ens = mean_k q_ens_k and over the corrected overestimation of
the FB checkpoint (analyze.stage0), it writes per seed, task and source, and over all
sources pooled ("all"),

    ensemble.csv   c_ens        mean c_ens_k
                   q_ens, G     means
                   ens_over     mean of q_ens - G, the ensemble's own overestimation
                   fb_over      mean of over
                   std_ens      mean std_ens
                   rho_q_ens_G, rho_fb_q_G    Spearman correlation of q_ens and of the
                                checkpoint's uncorrected min(q1, q2) with G
                   rho_std_fb_over, rho_std_ens_over   Spearman correlation of std_ens
                                with over and with q_ens - G

Run from the repository root:
    PYTHONPATH=.:tasks/T06-latent-switching uv run python -m analyze.ensemble \
        --probes <probe_seed0.npz> <probe_seed1.npz> <probe_seed2.npz> --out <dir>
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analyze.stage0 import overestimation, q_hat, write_csv


def ensemble_rows(seed, probe):
    over, q_fb, g = overestimation(probe), q_hat(probe, corrected=False), probe["G"]
    q_ens, std_ens = probe["q_ens"].mean(1), probe["q_ens"].std(1)
    rows = []
    for task in dict.fromkeys(probe["task"]):
        in_task = probe["task"] == task
        for source in [*dict.fromkeys(probe["source"]), "all"]:
            m = in_task if source == "all" else in_task & (probe["source"] == source)
            rows.append({
                "seed": seed, "task": task, "source": source, "n": int(m.sum()),
                "c_ens": probe["c_ens"][m].mean(), "q_ens": q_ens[m].mean(),
                "G": g[m].mean(), "ens_over": (q_ens - g)[m].mean(),
                "fb_over": over[m].mean(), "std_ens": std_ens[m].mean(),
                "rho_q_ens_G": spearmanr(q_ens[m], g[m]).statistic,
                "rho_fb_q_G": spearmanr(q_fb[m], g[m]).statistic,
                "rho_std_fb_over": spearmanr(std_ens[m], over[m]).statistic,
                "rho_std_ens_over": spearmanr(std_ens[m], (q_ens - g)[m]).statistic,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", nargs="+", required=True, help="one NPZ per seed, in order")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, path in enumerate(args.probes):
        rows += ensemble_rows(seed, dict(np.load(path)))
    write_csv(out / "ensemble.csv", rows)
    print(f"wrote {out / 'ensemble.csv'}")


if __name__ == "__main__":
    main()
