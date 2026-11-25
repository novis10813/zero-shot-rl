"""Fit the Walker reward formula, with free parameters, to the implied rewards of found latents.

The DMC Walker rewards are

    stand = (3 * tol(h, [1.2, inf), margin 0.6) + (1 + u) / 2) / 4
    walk  = stand * (5 * tol(vx, [1, inf), margin 0.5) + 1) / 6
    run   = stand * (5 * tol(vx, [8, inf), margin 4) + 1) / 6
    flip  = stand * (5 * tol(omega, [5, inf), margin 5) + 1) / 6

with torso height h, torso upright u, horizontal velocity vx and angular momentum omega. The
template generalises all four with 13 parameters:

    r(s) = a * [w * T(h; h0, inf, m_h) + (1 - w) * (1 + u) / 2]
             * [(1 - b_v) + b_v * T(vx; v_lo, v_hi, m_v)]
             * [(1 - b_o) + b_o * T(omega; o_lo, o_hi, m_o)] + c

where T is a Gaussian tolerance (1 inside [lo, hi], 0.1 at distance `margin` outside). b_v = b_o = 0
gives stand, b_v = 5/6 gives walk and run, b_o = 5/6 gives flip.

Targets, all on the reference states of `data/inputs/reference_states.npz`:
    true    the true task reward (checks that the template recovers the DMC parameters)
    z_r     r_{z_r}(s) = B(s)^T Sigma_B^{-1} z_r of the zero-shot latent
    <cond>  r_z of the latents in `data/inputs/found_z.npz`, averaged over seeds, per OOD env

Since r_z is linear in z, the seed average of r_z is r_z of the mean latent. Each target is fitted by
least squares from the DMC parameters and from random starts; the best fit is kept. Per-seed fits,
started from the seed-average solution, give the parameter spread over seeds.

Outputs `data/reward_template.csv` (one row per fit) and `data/reward_template.md`.

Usage (repo root):
    PYTHONPATH=.:tasks/T6.3-sensors uv run python -m analyze.reward_template
"""

import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analyze.z_reward import affine_scale, implied_rewards

PARAMS = ("a", "c", "w", "h0", "m_h", "b_v", "v_lo", "v_width", "m_v", "b_o", "o_lo", "o_width", "m_o")
LOWER = np.array([0.0, -10, 0, 0.0, 0.05, 0, -6, 0, 0.1, 0, -60, 0, 1.0])
UPPER = np.array([20.0, 10, 1, 2.0, 2.0, 1, 10, 30, 10, 1, 60, 200, 60])
# DMC parameters per task (a, c chosen so that the template equals the true reward).
DMC = {
    "stand": [1, 0, 0.75, 1.2, 0.6, 0, 0, 30, 1, 0, 0, 200, 5],
    "walk": [1, 0, 0.75, 1.2, 0.6, 5 / 6, 1, 30, 0.5, 0, 0, 200, 5],
    "run": [1, 0, 0.75, 1.2, 0.6, 5 / 6, 8, 30, 4, 0, 0, 200, 5],
    "flip": [1, 0, 0.75, 1.2, 0.6, 0, 0, 30, 1, 5 / 6, 5, 200, 5],
}
_K = math.sqrt(-2 * math.log(0.1))


def tolerance(x: np.ndarray, lo: float, hi: float, margin: float) -> np.ndarray:
    """1 inside [lo, hi]; Gaussian decay outside, equal to 0.1 at distance `margin`."""
    d = np.where(x < lo, lo - x, np.where(x > hi, x - hi, 0.0)) / margin
    return np.exp(-0.5 * (_K * d) ** 2)


def template(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Template reward for parameters `p` (order PARAMS) on sensors `s` [N, 4] = (h, u, vx, omega)."""
    a, c, w, h0, m_h, b_v, v_lo, v_w, m_v, b_o, o_lo, o_w, m_o = p
    h, u, vx, om = s.T
    stand = w * tolerance(h, h0, np.inf, m_h) + (1 - w) * (1 + u) / 2
    move = (1 - b_v) + b_v * tolerance(vx, v_lo, v_lo + v_w, m_v)
    spin = (1 - b_o) + b_o * tolerance(om, o_lo, o_lo + o_w, m_o)
    return a * stand * move * spin + c


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit(s: np.ndarray, y: np.ndarray, starts: Sequence[np.ndarray]) -> np.ndarray:
    """Best least-squares fit over the given starts."""
    best, best_cost = None, np.inf
    for x0 in starts:
        res = least_squares(lambda p: template(p, s) - y, np.clip(x0, LOWER, UPPER), bounds=(LOWER, UPPER),
                            x_scale="jac", max_nfev=2000)
        if res.cost < best_cost:
            best, best_cost = res.x, res.cost
    return best


def random_starts(task: str, n: int, rng: np.random.Generator) -> List[np.ndarray]:
    starts = [np.array(DMC[task], dtype=float)]
    starts += [LOWER + rng.random(len(PARAMS)) * (UPPER - LOWER) for _ in range(n)]
    return starts


def run(reference: Path, found_z: Path, n_starts: int = 8, seed: int = 0) -> pd.DataFrame:
    ref = np.load(reference)
    b = ref["backward"].astype(np.float64)
    s = ref["sensors"].astype(np.float64)
    true_r = ref["rewards"].astype(np.float64)
    tasks = [str(t) for t in ref["tasks"]]
    fz = np.load(found_z)
    z_found, conditions, envs, seeds = fz["z"], fz["conditions"], fz["envs"], fz["seeds"]
    if [str(t) for t in fz["tasks"]] != tasks:
        raise ValueError("task order differs between reference states and found latents")

    z_r = b.T @ true_r / len(b)
    z_r = math.sqrt(b.shape[1]) * z_r / np.linalg.norm(z_r, axis=0)  # [d, T], as weighted_z
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    def record(task, target, env, seed_label, y, p):
        rows.append({"task": task, "target": target, "env": env, "seed": seed_label, "r2": r2(y, template(p, s)),
                     **dict(zip(PARAMS, p))})

    for ti, task in enumerate(tasks):
        r_zr = implied_rewards(b, z_r[:, ti][None])[:, 0]
        a_s, b_s = affine_scale(r_zr, true_r[:, ti])
        record(task, "true", "-", "-", true_r[:, ti], fit(s, true_r[:, ti], random_starts(task, n_starts, rng)))
        y_zr = a_s * r_zr + b_s
        record(task, "z_r", "-", "-", y_zr, fit(s, y_zr, random_starts(task, n_starts, rng)))
        for ci, cond in enumerate(conditions):
            for ei, env in enumerate(envs):
                zs = z_found[ci, ei, ti]  # [seeds, d]
                y_seeds = a_s * implied_rewards(b, zs) + b_s  # [N, seeds]
                y_mean = y_seeds.mean(axis=1)
                p_mean = fit(s, y_mean, random_starts(task, n_starts, rng))
                record(task, str(cond), str(env), "mean", y_mean, p_mean)
                for si, sd in enumerate(seeds):
                    record(task, str(cond), str(env), int(sd), y_seeds[:, si], fit(s, y_seeds[:, si], [p_mean]))
    return pd.DataFrame(rows)


def formula(row: pd.Series) -> str:
    """Readable template formula for one fitted row."""
    def bounds(lo, width, hi_cap):
        hi = lo + width
        return f"[{lo:.2f}, {'∞' if hi >= hi_cap else f'{hi:.2f}'}]"

    parts = [f"{row.a:.2f} · ({row.w:.2f}·T(h, [{row.h0:.2f}, ∞), {row.m_h:.2f}) + {1 - row.w:.2f}·(1+u)/2)"]
    if row.b_v >= 0.05:
        parts.append(f"({1 - row.b_v:.2f} + {row.b_v:.2f}·T(vx, {bounds(row.v_lo, row.v_width, 4.0)}, {row.m_v:.2f}))")
    if row.b_o >= 0.05:
        parts.append(f"({1 - row.b_o:.2f} + {row.b_o:.2f}·T(ω, {bounds(row.o_lo, row.o_width, 60.0)}, {row.m_o:.1f}))")
    return " · ".join(parts) + f" {'+' if row.c >= 0 else '-'} {abs(row.c):.2f}"


def to_markdown(df: pd.DataFrame) -> str:
    main = df[df.seed.astype(str).isin(["-", "mean"])]
    spread = df[~df.seed.astype(str).isin(["-", "mean"])].groupby(["task", "target", "env"])["r2"].agg(["mean", "std"])
    lines = ["| task | target | env | R² | per-seed R² | formula |", "|---|---|---|---:|---:|---|"]
    for _, r in main.iterrows():
        key = (r.task, r.target, r.env)
        seed_r2 = f"{spread.loc[key, 'mean']:.2f} ± {spread.loc[key, 'std']:.2f}" if key in spread.index else "-"
        lines.append(f"| {r.task} | {r.target} | {r.env} | {r.r2:.3f} | {seed_r2} | {formula(r)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=Path("tasks/T6.3-sensors/data/inputs/reference_states.npz"))
    parser.add_argument("--found_z", type=Path, default=Path("tasks/T6.3-sensors/data/inputs/found_z.npz"))
    parser.add_argument("--out_dir", type=Path, default=Path("tasks/T6.3-sensors/data"))
    parser.add_argument("--n_starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    df = run(args.reference, args.found_z, args.n_starts, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "reward_template.csv", index=False, float_format="%.6f")
    (args.out_dir / "reward_template.md").write_text(to_markdown(df))
    print(to_markdown(df))


if __name__ == "__main__":
    main()
