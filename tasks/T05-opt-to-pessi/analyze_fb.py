"""Standard FB results: one comparison table, a latent PCA figure and performance against
the pessimistic-to-optimistic level.

Inputs are data/zs_fb_<job>.npz (implicit_z_fb.py) and data/rollouts_fb_<job>.csv
(rollout_eval.py), one job per seed. Every cell is computed per seed and task (IQM of 20
episodes), averaged over the 4 tasks, then over the 3 seeds; +- is the std over seeds.

Common axis d (per seed and task): the unit direction of the AX latents, built in
implicit_z_fb.py from the tangents of E2E-bi minus the mean of CQL and ReBRAC. The level
of a latent z is its signed angle along d,

    level = asin(z^T d / |z|)   (degrees, negative = pessimistic side)

and its direction is cos(t, d), t the unit tangent of z away from z_r.

Usage (repo root):
    uv run python tasks/T05-opt-to-pessi/analyze_fb.py
writes data/fb_comparison_table.md, figures/fb_pca.png, figures/fb_perf_vs_level.png.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import trim_mean  # noqa: E402

ROOT = Path(__file__).parent
JOBS = ["zero-shot-rl-30", "zero-shot-rl-31", "zero-shot-rl-32"]
NAMES = {  # family: reward type
    "z_r": "task reward only",
    "CQL": "CQL(H), -alpha (mu/pi_beta - 1)",
    "ReBRAC": "ReBRAC, -beta ||pi(s') - a'||^2",
    "ReBRAC+E2E": "ReBRAC + E2E practical",
    "E2E": "E2E practical, +c on data",
    "E2E-bi": "E2E bilevel, +beta pi/pi_beta",
    "AX": "common axis",
    "C1": "control: random rotation",
}
SHOWN = [5, 10, 35, 70]  # angles shown in the table when reached (5 only without 10)
TANGENT_FAMILIES = ["CQL", "ReBRAC", "ReBRAC+E2E", "E2E", "E2E-bi", "C1"]
COLORS = {"CQL": "tab:blue", "ReBRAC": "tab:green", "ReBRAC+E2E": "tab:olive",
          "E2E": "tab:orange", "E2E-bi": "tab:red", "AX": "black", "C1": "grey"}
# a behaviour is named when it differs from C1 at the same angle by more than this
THRESHOLDS = {"fall_frac": (0.02, "falls more", "falls less"),
              "knn_dist": (0.05, "farther from data", "closer to data"),
              "speed": (0.05, "faster", "slower"),
              "sat_frac": (0.02, "more saturated actions", "fewer saturated actions")}


def unit(v):
    return v / np.linalg.norm(v)


def tangent(z, z_r):
    u = unit(z_r)
    return unit(z - (z @ u) * u)


def latent_frame(zs, seed):
    """Per latent: cos to z_r, level along the common axis, cos of its tangent to the axis."""
    rows, tangents = [], {}
    for task in dict.fromkeys(zs["task"]):
        idx = np.flatnonzero(zs["task"] == task)
        fam, tgt = zs["family"][idx], zs["target_angle"][idx]
        z_r = zs["z"][idx[fam == "z_r"][0]]
        d = tangent(zs["z"][idx[(fam == "AX") & (tgt == 10)][0]], z_r)
        for i in idx:
            z = zs["z"][i]
            moved = zs["angle"][i] > 0.5
            t = tangent(z, z_r) if moved else np.full_like(z, np.nan)
            rows.append({"seed": seed, "task": task, "row": i, "cos": unit(z) @ unit(z_r),
                         "level": np.degrees(np.arcsin(np.clip(unit(z) @ d, -1, 1))),
                         "cos_axis": t @ d})
        for f in TANGENT_FAMILIES:
            sel = idx[(fam == f) & (tgt <= 10)]
            if len(sel):  # 10 degrees (first C1 rotation), or 5 if 10 is not reached
                at10 = sel[zs["target_angle"][sel] == 10]
                best = at10[0] if len(at10) else sel[0]
                tangents[(seed, task, f)] = tangent(zs["z"][best], z_r)
    return pd.DataFrame(rows), tangents


def load(data_dir, jobs):
    rollouts, lat, tangents, zs_all = [], [], {}, {}
    for seed, job in enumerate(jobs):
        zs = dict(np.load(data_dir / f"zs_fb_{job}.npz"))
        zs_all[seed] = zs
        frame, tg = latent_frame(zs, seed)
        lat.append(frame)
        tangents.update(tg)
        rollouts.append(pd.read_csv(data_dir / f"rollouts_fb_{job}.csv").assign(seed=seed))
    d = pd.concat(rollouts, ignore_index=True).merge(pd.concat(lat), on=["seed", "task", "row"])
    d["target_angle"] = d["target_angle"].fillna(0.0)
    d = d[d.family != "C2"]
    cell = d.groupby(["seed", "task", "family", "target_angle"]).agg(
        ret=("return", lambda x: trim_mean(x, 0.25)), angle=("angle", "mean"),
        strength=("strength", "mean"), cos=("cos", "mean"), level=("level", "mean"),
        cos_axis=("cos_axis", "mean"), **{m: (m, "mean") for m in THRESHOLDS}).reset_index()
    base = cell[cell.family == "z_r"].set_index(["seed", "task"])
    cell["rel"] = cell.ret.values / base.loc[list(zip(cell.seed, cell.task)), "ret"].values - 1
    return cell, tangents, zs_all


def table(cell, tangents, n_tasks):
    seed = cell.groupby(["seed", "family", "target_angle"]).mean(numeric_only=True).reset_index()
    tab = seed.groupby(["family", "target_angle"]).agg(["mean", "std"])
    reached = cell.groupby(["seed", "family", "target_angle"]).size().groupby(
        ["family", "target_angle"]).min()

    def shown(fam):
        have = set(tab.loc[fam].index)
        if fam == "z_r":
            return [0.0]
        if fam == "AX":
            return [a for a in (-35.0, -10.0, 10.0, 35.0) if a in have]
        angles = [float(a) for a in SHOWN if a in have]
        return [a for a in angles if not (a == 5 and 10.0 in have)]

    lines = ["| Reward type | Strength: angle to z_r (coefficient, median) | Return IQM (change vs z_r) "
             "| Fall fraction | Latent: cos to z_r / level on axis / cos to axis "
             "| Main behaviour vs random rotation |",
             "|---|---|---|---|---|---|"]
    for fam, name in NAMES.items():
        if fam not in tab.index.get_level_values(0):
            continue
        for a in shown(fam):
            r = tab.loc[(fam, a)]
            coef = cell[(cell.family == fam) & (cell.target_angle == a)].strength.median()
            label = name
            if fam == "AX":
                label += ", pessimistic side" if a < 0 else ", optimistic side"
            strength = "0" if fam == "z_r" else (
                f"{abs(a):g} deg" if fam in ("AX", "C1") else f"{a:g} deg ({coef:+.3g})")
            if reached[(fam, a)] < n_tasks:
                strength += " *"
            ret = f"{r['ret', 'mean']:.0f} ({100 * r['rel', 'mean']:+.0f}% +- {100 * r['rel', 'std']:.0f})"
            latent = f"{r['cos', 'mean']:.2f} / {r['level', 'mean']:+.0f} deg"
            if fam != "z_r":
                latent += f" / {r['cos_axis', 'mean']:+.2f}"
            if fam in ("z_r", "C1"):
                behaviour = "reference" if fam == "z_r" else "-"
            else:
                key = ("C1", abs(a))
                words = []
                if key in tab.index:
                    for m, (th, up, down) in THRESHOLDS.items():
                        diff = r[m, "mean"] - tab.loc[key][m, "mean"]
                        if abs(diff) > th:
                            words.append(f"{up if diff > 0 else down} ({diff:+.2f})")
                behaviour = ", ".join(words) if words else (
                    "like random rotation" if key in tab.index else "no C1 at this angle")
            lines.append(f"| {label} | {strength} | {ret} | {r['fall_frac', 'mean']:.2f} "
                         f"| {latent} | {behaviour} |")

    # pairwise cosines of the family directions
    keys = sorted({(s, t) for s, t, _ in tangents})
    fams = [f for f in TANGENT_FAMILIES if any((s, t, f) in tangents for s, t in keys)]
    lines += ["", "Direction of each family away from z_r (unit tangent at 10 deg, 5 deg if 10 is not "
              "reached): cosine, mean over seeds and tasks.", "",
              "| | " + " | ".join(fams) + " |", "|---" * (len(fams) + 1) + "|"]
    for f in fams:
        vals = []
        for g in fams:
            c = [tangents[(s, t, f)] @ tangents[(s, t, g)] for s, t in keys
                 if (s, t, f) in tangents and (s, t, g) in tangents]
            vals.append(f"{np.mean(c):+.2f}")
        lines.append(f"| {f} | " + " | ".join(vals) + " |")
    lines += ["", "Return: mean over 4 tasks of the IQM of 20 episodes, mean over 3 seeds; change vs "
              "z_r +- std over seeds. Fall fraction: share of steps with torso height < 0.6. "
              "Level: signed angle along the common axis (negative = pessimistic side). "
              "* = angle not reached on every task in every seed (fewer tasks averaged). "
              "Behaviour thresholds against C1 at the same angle: fall 0.02, kNN distance 0.05, "
              "speed 0.05, saturation 0.02."]
    return "\n".join(lines)


def plot_level(cell, path):
    seed = cell.groupby(["seed", "family", "target_angle"]).mean(numeric_only=True).reset_index()
    tab = seed.groupby(["family", "target_angle"]).agg(["mean", "std"]).reset_index()
    panels = [("rel", "return change vs z_r (%)", 100), ("fall_frac", "fall fraction", 1),
              ("knn_dist", "kNN distance to data", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    c1 = tab[tab.family == "C1"].sort_values("target_angle")
    z_r = tab[tab.family == "z_r"]
    for ax, (m, title, k) in zip(axes, panels):
        for sign in (-1, 1):  # C1 at +-angle: what any rotation by that angle does
            x = np.concatenate([[0], sign * c1.target_angle.values])
            y = np.concatenate([z_r[m, "mean"].values, c1[m, "mean"].values]) * k
            s = np.concatenate([z_r[m, "std"].values, c1[m, "std"].values]) * k
            s = np.nan_to_num(s)
            ax.fill_between(x, y - s, y + s, color="grey", alpha=0.2,
                            label="random rotation (C1)" if sign == 1 else None)
        for fam, color in COLORS.items():
            if fam == "C1":
                continue
            f = tab[tab.family == fam].sort_values("target_angle")
            f = pd.concat([z_r, f]) if fam != "AX" else pd.concat([f[f.target_angle < 0], z_r,
                                                                   f[f.target_angle > 0]])
            f = f.sort_values(("level", "mean"))
            ax.errorbar(f["level", "mean"], f[m, "mean"] * k, yerr=f[m, "std"].fillna(0) * k, color=color,
                        marker="o", ms=4, capsize=2, lw=2.5 if fam == "AX" else 1.2,
                        ls="--" if fam == "AX" else "-", label=fam)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("level on common axis (deg), pessimistic <- 0 -> optimistic")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_pca(zs, path):
    tasks = list(dict.fromkeys(zs["task"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 4.2))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        idx = np.flatnonzero((zs["task"] == task) & (zs["family"] != "C2"))
        z = zs["z"][idx] / np.sqrt(zs["z"].shape[1])
        mean = z.mean(0)
        _, s, vt = np.linalg.svd(z - mean, full_matrices=False)
        p = (z - mean) @ vt[:2].T
        fam, tgt = zs["family"][idx], zs["target_angle"][idx]
        r0 = p[fam == "z_r"][0]
        c1 = fam == "C1"
        ax.scatter(p[c1, 0], p[c1, 1], s=8, color="grey", alpha=0.5, label="C1")
        for f, color in COLORS.items():
            if f in ("C1",):
                continue
            sel = np.flatnonzero(fam == f)
            sel = sel[np.argsort(tgt[sel])]
            if f == "AX":
                pts = np.vstack([p[sel[tgt[sel] < 0]], r0, p[sel[tgt[sel] > 0]]])
            else:
                pts = np.vstack([r0, p[sel]])
            ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3, color=color, label=f,
                    lw=2.5 if f == "AX" else 1.2, ls="--" if f == "AX" else "-")
        ax.scatter(*r0, marker="*", s=200, color="gold", edgecolor="k", zorder=5, label="z_r")
        var = s[:2] ** 2 / (s ** 2).sum()
        ax.set_title(f"{task} (seed 0)")
        ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--figure_dir", type=str, default=str(ROOT / "figures"))
    parser.add_argument("--jobs", nargs="+", default=JOBS)
    args = parser.parse_args()
    data_dir, figure_dir = Path(args.data_dir), Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    cell, tangents, zs = load(data_dir, args.jobs)
    text = table(cell, tangents, cell.task.nunique())
    (data_dir / "fb_comparison_table.md").write_text(text + "\n")
    print(text)
    plot_level(cell, figure_dir / "fb_perf_vs_level.png")
    plot_pca(zs[0], figure_dir / "fb_pca.png")


if __name__ == "__main__":
    main()
