"""Tables and figures from the P2 latents (zs_*.npz) and rollouts (rollouts_*.csv).

Performance and behaviour
    data/summary_per_task.csv   per task, family and target angle, episodes pooled over seeds:
                                return IQM / median / Q1 / Q3 and behaviour means
    data/summary_overall.csv    per family and target angle: change against G0 of the same
                                seed and task, averaged over tasks, mean and std over seeds
    figures/perf_vs_angle.png   the overall changes against the angle to z_r
    figures/return_by_task.png  return IQM against the angle, per task (seeds pooled)

Latent geometry (computed inside each seed: latent spaces of different seeds are not aligned)
    data/latent_to_g0.csv       cosine and distance of every z to the G0 latent of its task
    data/latent_tangent_cos.csv cosine between the directions in which families move z away
                                from z_r (component orthogonal to z_r), per seed, mean over tasks
    figures/latent_tangent_cos.png  the same, mean over seeds
    figures/pca.png             PCA of each seed's latents, one panel per seed and task
    figures/pca_3d.html         the same with three components, rotatable (plotly.js from a CDN)

Usage (repo root):
    uv run python tasks/T05-opt-to-pessi/analyze.py
"""

from pathlib import Path

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

ROOT = Path(__file__).parent
JOBS = {"zero-shot-rl-18": 0, "zero-shot-rl-19": 1, "zero-shot-rl-20": 2}
TASKS = ["stand", "walk", "run", "flip"]
LABELS = {
    "G0": "linear z_r",
    "G0p": "paper softmax(10r) z",
    "G1": "CQL(H), alpha > 0",
    "G1neg": "CQL(H), alpha < 0",
    "G3": "ReBRAC",
    "G3c": "ReBRAC, centred",
    "G5": "E2E practical, c > 0",
    "G5neg": "E2E practical, c < 0",
    "G6": "ReBRAC + E2E practical",
    "C1": "random rotation (control)",
    "C2": "z_r on 1/5 of the data (control)",
}
SWEEPS = ["G1", "G1neg", "G3", "G3c", "G5", "G5neg", "G6", "C1"]
TANGENT_ANGLE = 20.0
BEHAVIOUR = ["fall_frac", "speed", "upright", "abs_action", "sat_frac", "knn_dist", "m7"]
COLORS = dict(zip(SWEEPS, plt.get_cmap("tab10").colors))


def iqm(x):
    return trim_mean(x, 0.25)


def load():
    rollouts, latents = [], {}
    for job, seed in JOBS.items():
        d = pd.read_csv(ROOT / "data" / f"rollouts_{job}.csv")
        d["seed"] = seed
        rollouts.append(d)
        latents[seed] = dict(np.load(ROOT / "data" / f"zs_{job}.npz"))
    d = pd.concat(rollouts, ignore_index=True)
    d["target_angle"] = d["target_angle"].fillna(0.0)
    return d, latents


def performance_tables(d):
    keys = ["task", "family", "target_angle"]
    per_task = d.groupby(keys).agg(
        episodes=("return", "size"),
        angle=("angle", "mean"),
        return_iqm=("return", iqm),
        return_median=("return", "median"),
        return_q1=("return", lambda x: x.quantile(0.25)),
        return_q3=("return", lambda x: x.quantile(0.75)),
        **{m: (m, "mean") for m in BEHAVIOUR},
    ).reset_index()

    # per seed and task, then the change against G0 of the same seed and task
    cell = d.groupby(["seed"] + keys).agg(
        return_iqm=("return", iqm), **{m: (m, "mean") for m in BEHAVIOUR}
    ).reset_index()
    base = cell[cell.family == "G0"].set_index(["seed", "task"])
    metrics = ["return_iqm"] + BEHAVIOUR
    for m in metrics:
        cell["d_" + m] = cell[m].values - base.loc[list(zip(cell.seed, cell.task)), m].values
    cell["rel_return"] = cell["d_return_iqm"] / base.loc[
        list(zip(cell.seed, cell.task)), "return_iqm"
    ].values
    deltas = ["d_" + m for m in metrics] + ["rel_return"]
    per_seed = cell.groupby(["seed", "family", "target_angle"])[deltas].mean().reset_index()
    overall = per_seed.groupby(["family", "target_angle"])[deltas].agg(["mean", "std"])
    overall.columns = [f"{a}_{b}" for a, b in overall.columns]
    overall = overall.reset_index()
    overall.insert(1, "label", overall.family.map(LABELS))
    return per_task, overall


def plot_performance(per_task, overall):
    panels = [("d_return_iqm", "change in return IQM"), ("d_fall_frac", "change in fall fraction"),
              ("d_knn_dist", "change in kNN distance to data"), ("d_sat_frac", "change in saturated actions")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    for ax, (m, title) in zip(axes, panels):
        for fam in SWEEPS:
            s = overall[(overall.family == fam) | (overall.family == "G0")].sort_values("target_angle")
            if len(s) < 3:
                continue
            ls = "--" if fam == "C1" else "-"
            ax.errorbar(s.target_angle, s[m + "_mean"], yerr=s[m + "_std"], label=LABELS[fam],
                        color=COLORS[fam], ls=ls, marker="o", ms=3, capsize=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="target angle to z_r (deg)", title=title)
    axes[0].legend(fontsize=7)
    fig.suptitle("Mean over tasks of the change against G0; error bars: std over 3 seeds")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "perf_vs_angle.png", dpi=150)

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.2 * len(TASKS), 3.8))
    for ax, task in zip(axes, TASKS):
        t = per_task[per_task.task == task]
        for fam in SWEEPS:
            s = t[(t.family == fam) | (t.family == "G0")].sort_values("target_angle")
            if len(s) < 3:
                continue
            ax.plot(s.target_angle, s.return_iqm, marker="o", ms=3, color=COLORS[fam],
                    ls="--" if fam == "C1" else "-", label=LABELS[fam])
        ax.axhline(t[t.family == "G0p"].return_iqm.item(), color="grey", ls=":", label=LABELS["G0p"])
        ax.set(xlabel="target angle to z_r (deg)", ylabel="return IQM", title=task)
    axes[0].legend(fontsize=7)
    fig.suptitle("Return IQM over 3 seeds x 20 episodes")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "return_by_task.png", dpi=150)


def latent_tables(latents):
    rows, cos_rows = [], []
    for seed, z in latents.items():
        for task in TASKS:
            in_task = z["task"] == task
            g0 = z["z"][in_task & (z["family"] == "G0")][0]
            for i in np.flatnonzero(in_task):
                v = z["z"][i]
                rows.append({
                    "seed": seed, "task": task, "family": z["family"][i],
                    "target_angle": z["target_angle"][i], "strength": z["strength"][i],
                    "converged": z["converged"][i], "multi": z["multi"][i],
                    "cos_to_g0": v @ g0 / (np.linalg.norm(v) * np.linalg.norm(g0)),
                    "dist_to_g0": np.linalg.norm(v - g0),
                })
            # unit directions orthogonal to z_r in which each family moves z
            unit = g0 / np.linalg.norm(g0)

            def tangent(v):
                v = v - (v @ unit) * unit
                return v / np.linalg.norm(v)

            dirs = {"u_1": tangent(z["u_1"])}
            for fam in SWEEPS:
                k = in_task & (z["family"] == fam) & (z["target_angle"] <= TANGENT_ANGLE)
                if k.any():
                    i = np.flatnonzero(k)[np.argmax(z["target_angle"][k])]
                    dirs[fam] = tangent(z["z"][i] - g0)
            for a in dirs:
                for b in dirs:
                    cos_rows.append({"seed": seed, "task": task, "a": a, "b": b,
                                     "cos": dirs[a] @ dirs[b]})
    to_g0 = pd.DataFrame(rows)
    cos = pd.DataFrame(cos_rows).groupby(["seed", "a", "b"]).cos.mean().reset_index()
    return to_g0, cos


def plot_latents(latents, cos):
    names = SWEEPS + ["u_1"]
    mean = cos.groupby(["a", "b"]).cos.mean().unstack().reindex(index=names, columns=names)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean.values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(mean.values[i, j]):
                ax.text(j, i, f"{mean.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(names)), names, rotation=45)
    ax.set_yticks(range(len(names)), names)
    ax.set_title(f"Cosine between directions away from z_r (at <= {TANGENT_ANGLE:g} deg)\n"
                 "mean over 4 tasks and 3 seeds; C1 = first random direction")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "latent_tangent_cos.png", dpi=150)

    fig, axes = plt.subplots(len(latents), len(TASKS), figsize=(4.2 * len(TASKS), 4 * len(latents)))
    for r, (seed, z) in enumerate(latents.items()):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            k = (z["task"] == task) & np.isin(z["family"], SWEEPS[:-1] + ["G0", "G0p", "C2"])
            x = z["z"][k] - z["z"][k].mean(0)
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            p = x @ vt[:2].T
            var = s[:2] ** 2 / (s ** 2).sum()
            fam, ang = z["family"][k], z["target_angle"][k]
            g0 = p[fam == "G0"][0]
            for f in SWEEPS[:-1]:
                m = fam == f
                if m.any():
                    order = np.argsort(ang[m])
                    pts = np.vstack([g0, p[m][order]])
                    ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3, color=COLORS[f], label=LABELS[f])
            ax.scatter(*g0, color="k", s=40, zorder=5, label=LABELS["G0"])
            ax.scatter(*p[fam == "G0p"].T, color="grey", marker="*", s=80, zorder=5, label=LABELS["G0p"])
            ax.scatter(*p[fam == "C2"].T, color="k", marker="x", s=20, label=LABELS["C2"])
            ax.set(title=f"seed {seed}, {task}", xlabel=f"PC1 ({var[0]:.0%})", ylabel=f"PC2 ({var[1]:.0%})")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("PCA of each seed and task's latents; lines run from G0 through increasing angle")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "pca.png", dpi=120)


PCA_3D_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>T05 latent PCA</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body style="font-family: sans-serif">
<p>PCA of one seed and task's latents, three components. Lines run from G0 through increasing
target angle. Drag to rotate, scroll to zoom, click a legend entry to hide it.
<select id="view"></select></p>
<div id="plot" style="width: 100%; height: 85vh"></div>
<script>
const VIEWS = __DATA__;
const select = document.getElementById("view");
Object.keys(VIEWS).forEach(k => select.add(new Option(k, k)));
function draw() {
  const v = VIEWS[select.value];
  const traces = v.traces.map(t => ({
    type: "scatter3d", mode: t.lines ? "lines+markers" : "markers", name: t.name,
    x: t.x, y: t.y, z: t.z, text: t.text, hoverinfo: "text+name",
    marker: {size: t.size, color: t.color, symbol: t.symbol}, line: {color: t.color, width: 5},
  }));
  const axis = i => ({title: `PC${i + 1} (${(100 * v.var[i]).toFixed(0)}%)`});
  Plotly.react("plot", traces, {scene: {xaxis: axis(0), yaxis: axis(1), zaxis: axis(2)},
    margin: {l: 0, r: 0, t: 30, b: 0}, title: select.value});
}
select.onchange = draw;
draw();
</script></body></html>
"""


def plot_pca_3d(latents):
    hex_colors = {f: matplotlib.colors.to_hex(c) for f, c in COLORS.items()}
    views = {}
    for seed, z in latents.items():
        for task in TASKS:
            k = (z["task"] == task) & np.isin(z["family"], SWEEPS[:-1] + ["G0", "G0p", "C2"])
            x = z["z"][k] - z["z"][k].mean(0)
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            p = x @ vt[:3].T
            fam, ang, strength = z["family"][k], z["target_angle"][k], z["strength"][k]
            g0 = p[fam == "G0"][0]

            def trace(name, pts, text, color, lines, size=4, symbol="circle"):
                return {"name": name, "x": pts[:, 0].tolist(), "y": pts[:, 1].tolist(),
                        "z": pts[:, 2].tolist(), "text": text, "color": color, "lines": lines,
                        "size": size, "symbol": symbol}

            traces = []
            for f in SWEEPS[:-1]:
                m = fam == f
                if m.any():
                    order = np.argsort(ang[m])
                    text = ["G0"] + [f"{a:g} deg, strength {s:+.3g}"
                                     for a, s in zip(ang[m][order], strength[m][order])]
                    traces.append(trace(LABELS[f], np.vstack([g0, p[m][order]]), text,
                                        hex_colors[f], True))
            traces.append(trace(LABELS["G0"], g0[None], ["G0"], "#000000", False, 8))
            traces.append(trace(LABELS["G0p"], p[fam == "G0p"], ["G0p"], "#808080", False, 8,
                                "diamond"))
            traces.append(trace(LABELS["C2"], p[fam == "C2"], ["C2"] * int((fam == "C2").sum()),
                                "#000000", False, 4, "x"))
            views[f"seed {seed}, {task}"] = {"traces": traces,
                                            "var": (s[:3] ** 2 / (s ** 2).sum()).tolist()}
    html = PCA_3D_HTML.replace("__DATA__", json.dumps(views, separators=(",", ":")))
    (ROOT / "figures" / "pca_3d.html").write_text(html)


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    d, latents = load()
    per_task, overall = performance_tables(d)
    per_task.to_csv(ROOT / "data" / "summary_per_task.csv", index=False, float_format="%.4g")
    overall.to_csv(ROOT / "data" / "summary_overall.csv", index=False, float_format="%.4g")
    plot_performance(per_task, overall)
    to_g0, cos = latent_tables(latents)
    to_g0.to_csv(ROOT / "data" / "latent_to_g0.csv", index=False, float_format="%.4g")
    cos.to_csv(ROOT / "data" / "latent_tangent_cos.csv", index=False, float_format="%.4g")
    plot_latents(latents, cos)
    plot_pca_3d(latents)

    pd.set_option("display.width", 200)
    cols = ["family", "target_angle", "d_return_iqm_mean", "d_return_iqm_std", "rel_return_mean",
            "d_fall_frac_mean", "d_knn_dist_mean", "d_sat_frac_mean"]
    print(overall[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
