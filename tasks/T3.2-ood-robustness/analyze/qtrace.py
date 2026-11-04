"""Q ranking and ensemble spread in the five dynamics configs.

Reads the NPZ traces of zero_shot_rl.eval.qtrace (run_qtrace.sh), one per training seed and
config, and writes data/qtrace.csv with one row per config, pooled over the training seeds:

    points         branch points (episodes x branch steps x seeds)
    mean_abs_dg    mean |g_opt - g_pess|, the realized 100-step return difference of the two actions
    agree          fraction of points where sign(q_opt - q_pess) = sign(g_opt - g_pess)
    agree_top_dq   the same on the quarter of points with the largest |q_opt - q_pess|
    corr           Pearson correlation of q_opt - q_pess with g_opt - g_pess
    corr_seed0..2  the same within each training seed
    q_choice_gain  mean of (g of the action Q prefers) - g_opt over the points
    oracle_gain    mean of max(g_opt, g_pess) - g_opt over the points
    u_opt          mean ensemble std of q_opt over the 1000 steps of the always-opt episodes
    q_opt          mean over the 1000 steps of the ensemble mean Q of a_opt; q_minus_clean and its
                   standard error are paired with the clean config like u_minus_clean
    u_minus_clean  mean over (training seed, episode) of the episode's u_opt minus that of the same
                   episode in the clean config (same start state), with its standard error

Usage (repo root):
    PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.qtrace
    PYTHONPATH=.:tasks/T3.2-ood-robustness uv run python -m analyze.qtrace --check

--check recomputes the table and fails if data/qtrace.csv differs byte for byte.
"""

import csv
import io
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
CONFIGS = ["baseline", "friction_mild", "friction_strong", "actuator_mild", "actuator_strong"]
SEEDS = [0, 1, 2]
FIELDS = (["config", "points", "mean_abs_dg", "agree", "agree_top_dq", "corr"]
          + [f"corr_seed{s}" for s in SEEDS]
          + ["q_choice_gain", "oracle_gain", "u_opt", "u_minus_clean", "u_minus_clean_se", "q_opt", "q_minus_clean",
                                 "q_minus_clean_se"])


def episode_mean(input_dir, config, key):
    """Per (training seed, episode) mean of a per-step trace over the 1000 steps."""
    return np.concatenate([np.load(input_dir / f"qtrace_s{s}_{config}.npz")[key].mean(1) for s in SEEDS])


def table(input_dir):
    rows = []
    clean_u, clean_q = (episode_mean(input_dir, "baseline", k) for k in ("u_opt", "q_opt"))
    for config in CONFIGS:
        dq, dg, go, gp, corr_seed, u = [], [], [], [], [], []
        for seed in SEEDS:
            d = np.load(input_dir / f"qtrace_s{seed}_{config}.npz")
            steps = d["branch_steps"]
            q = (d["q_opt"] - d["q_pess"])[:, steps].ravel().astype(np.float64)
            g = (d["g_opt"] - d["g_pess"]).ravel().astype(np.float64)
            dq.append(q), dg.append(g), go.append(d["g_opt"].ravel()), gp.append(d["g_pess"].ravel())
            corr_seed.append(np.corrcoef(q, g)[0, 1])
            u.append(d["u_opt"])
        q, g, go, gp = (np.concatenate(x) for x in (dq, dg, go, gp))
        u = np.concatenate(u)
        top = np.abs(q) >= np.quantile(np.abs(q), 0.75)
        pick = np.where(q < 0, gp, go)
        row = {
            "config": config, "points": len(q), "mean_abs_dg": np.abs(g).mean(),
            "agree": np.mean(np.sign(q) == np.sign(g)),
            "agree_top_dq": np.mean(np.sign(q[top]) == np.sign(g[top])),
            "corr": np.corrcoef(q, g)[0, 1],
            "q_choice_gain": (pick - go).mean(), "oracle_gain": (np.maximum(go, gp) - go).mean(),
            "u_opt": u.mean(),
        }
        for key, clean in (("u", clean_u), ("q", clean_q)):
            paired = episode_mean(input_dir, config, f"{key}_opt") - clean
            row[f"{key}_minus_clean"] = paired.mean()
            row[f"{key}_minus_clean_se"] = paired.std(ddof=1) / np.sqrt(len(paired)) if config != "baseline" else 0.0
        row["q_opt"] = episode_mean(input_dir, config, "q_opt").mean()
        row.update({f"corr_seed{s}": c for s, c in zip(SEEDS, corr_seed)})
        rows.append(row)
    return rows


def render(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    return buf.getvalue()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=TASK / "data" / "inputs")
    parser.add_argument("--out", type=Path, default=TASK / "data" / "qtrace.csv")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render(table(args.input_dir))
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            raise SystemExit(f"Stale or missing output: {args.out}")
        print("ok")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
