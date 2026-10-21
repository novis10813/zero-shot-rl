"""One comparison table: reward type, bias strength, return, falls, latent similarity, behaviour.

Each cell is computed per seed (20 episodes per task), averaged over the 4 tasks, then
averaged over the 3 seeds; +- is the std over seeds. Behaviour is described against a random
rotation of z_r by the same angle (C1), so only effects beyond "z moved this far" are named.

Usage (repo root):
    uv run python tasks/T05-opt-to-pessi/comparison_table.py
writes data/comparison_table.md and prints it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

ROOT = Path(__file__).parent
JOBS = ["zero-shot-rl-18", "zero-shot-rl-19", "zero-shot-rl-20"]
ROWS = [  # family, reward type, angles shown
    ("G0", "task reward only (linear z_r)", [0]),
    ("G0p", "task reward only (paper softmax z)", [0]),
    ("G1", "CQL(H), alpha > 0", [10, 35, 70]),
    ("G1neg", "CQL(H), alpha < 0", [10, 35, 70]),
    ("G3", "ReBRAC", [10, 35, 70]),
    ("G3c", "ReBRAC, centred", [10, 35, 70]),
    ("G5", "E2E practical, c > 0", [5, 20]),
    ("G5neg", "E2E practical, c < 0", [5, 20, 50]),
    ("G6", "ReBRAC + E2E practical", [10, 35]),
    ("C1", "control: random rotation", [10, 35, 70]),
]
SAME_AS = {"G3": "E2E c < 0 (cos 0.98)", "G5neg": "ReBRAC (cos 0.98)", "G3c": "ReBRAC + E2E (cos 0.99)",
           "G6": "ReBRAC centred (cos 0.99)", "G1neg": "CQL alpha > 0, reversed (cos -1.00)",
           "G1": "own direction", "G5": "own direction (= u_1)", "C1": "none (random)"}
# a behaviour is named when it differs from C1 at the same angle by more than this
THRESHOLDS = {"fall_frac": (0.02, "falls more", "falls less"),
              "knn_dist": (0.05, "farther from data", "closer to data"),
              "speed": (0.05, "faster", "slower"),
              "sat_frac": (0.02, "more saturated actions", "fewer saturated actions")}


def main():
    d = pd.concat([pd.read_csv(ROOT / "data" / f"rollouts_{j}.csv").assign(seed=s)
                   for s, j in enumerate(JOBS)])
    d["target_angle"] = d["target_angle"].fillna(0.0)
    cell = d.groupby(["seed", "task", "family", "target_angle"]).agg(
        ret=("return", lambda x: trim_mean(x, 0.25)), angle=("angle", "mean"),
        strength=("strength", "mean"), **{m: (m, "mean") for m in THRESHOLDS}).reset_index()
    base = cell[cell.family == "G0"].set_index(["seed", "task"])
    cell["rel"] = cell.ret.values / base.loc[list(zip(cell.seed, cell.task)), "ret"].values - 1
    cell["cos"] = np.cos(np.radians(cell.angle))
    seed = cell.groupby(["seed", "family", "target_angle"]).mean(numeric_only=True).reset_index()
    tab = seed.groupby(["family", "target_angle"]).agg(["mean", "std"])
    tasks_reached = cell.groupby(["seed", "family", "target_angle"]).size().groupby(
        ["family", "target_angle"]).min()

    lines = ["| Reward type | Strength: angle to z_r (coefficient, median) | Return IQM (change vs z_r) "
             "| Fall fraction | Latent: cos to z_r / direction same as | Main behaviour vs random rotation |",
             "|---|---|---|---|---|---|"]
    for fam, name, angles in ROWS:
        for a in angles:
            r = tab.loc[(fam, float(a))]
            coef = cell[(cell.family == fam) & (cell.target_angle == a)].strength.median()
            strength = "0" if fam in ("G0", "G0p") else (
                f"{a} deg" if fam == "C1" else f"{a} deg ({coef:+.3g})")
            if tasks_reached[(fam, float(a))] < 4:
                strength += " *"
            ret = f"{r['ret', 'mean']:.0f} ({100 * r['rel', 'mean']:+.0f}% +- {100 * r['rel', 'std']:.0f})"
            fall = f"{r['fall_frac', 'mean']:.2f}"
            latent = f"{r['cos', 'mean']:.2f}"
            if fam in SAME_AS:
                latent += f" / {SAME_AS[fam]}"
            if fam in ("G0", "C1") or a == 0:
                behaviour = "reference" if fam == "G0" else "-"
            else:
                ref = tab.loc[("C1", float(a))] if ("C1", float(a)) in tab.index else None
                words = []
                if ref is not None:
                    for m, (th, up, down) in THRESHOLDS.items():
                        diff = r[m, "mean"] - ref[m, "mean"]
                        if abs(diff) > th:
                            words.append(f"{up if diff > 0 else down} ({diff:+.2f})")
                behaviour = ", ".join(words) if words else (
                    "like random rotation" if ref is not None else "no C1 at this angle")
            lines.append(f"| {name} | {strength} | {ret} | {fall} | {latent} | {behaviour} |")
    lines += ["", "Return: mean over 4 tasks of the IQM of 20 episodes, mean over 3 seeds; "
              "change vs z_r +- std over seeds. Fall fraction: share of steps with torso height < 0.6. "
              "* = angle not reached on every task in every seed (fewer tasks averaged). "
              "Behaviour thresholds against C1: fall 0.02, kNN distance 0.05, speed 0.05, saturation 0.02."]
    text = "\n".join(lines)
    (ROOT / "data" / "comparison_table.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
