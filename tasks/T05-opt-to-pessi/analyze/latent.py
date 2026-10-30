"""Within-checkpoint/task Standard-FB latent visualization and common-axis diagnostics.

No model execution. Only the three zs_fb archives are read. Figures use the common
10-degree slice; tables also retain the balanced 5-degree sensitivity slice.

Run from the repository root:
    PYTHONPATH=.:tasks/T05-opt-to-pessi .venv/bin/python -m analyze.latent [--check]
"""

import argparse
from io import BytesIO
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze.q1 import analyze as analyze_q1, summarize
from common import (FIGURES, INPUTS, MATCHED_ANGLES, METHODS, OUTPUTS, SOURCES, TASKS, csv_text,
                    write_or_check)
from geometry import cosine, tangent, unit

FAMILIES = tuple(METHODS)
RESIDUAL_TOL = 1e-5  # norm relative to an already unit tangent; never amplify zero
STAGES = ("tangent", "constant_removed")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def remove_constant(direction, constant):
    """Remove a unit constant-reward tangent. Keep magnitude before renormalizing."""
    direction, constant = unit(direction), unit(constant)
    residual = direction - (direction @ constant) * constant
    norm = float(np.linalg.norm(residual))
    return residual, norm


def decomposition(vectors, centered=False):
    """PCA centers rows; the common-axis SVD intentionally does not center them."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) < 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected at least two finite vector rows")
    mean = vectors.mean(axis=0) if centered else np.zeros(vectors.shape[1])
    matrix = vectors - mean
    _, singular, axes = np.linalg.svd(matrix, full_matrices=False)
    # Resolve the arbitrary sign for reproducible plots, not for optimism ordering.
    for axis in axes:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    energy = singular ** 2
    if energy.sum() <= 1e-20:
        raise ValueError("Degenerate decomposition")
    return mean, singular, axes, energy / energy.sum()


def analyze(data_dir):
    analyze_q1(data_dir)  # validates saved solutions, coverage, angles and norms
    pairs, directions, spectra, points, pca_variance = [], [], [], [], []
    for seed, source in SOURCES:
        with np.load(data_dir / source, allow_pickle=False) as archive:
            data = dict(archive)
        if data["u_1"].shape != (50,):
            raise ValueError(f"{source}: malformed constant-reward vector")
        for task in TASKS:
            mask = data["task"] == task
            baseline_index = int(np.flatnonzero(mask & (data["family"] == "z_r"))[0])
            baseline = unit(data["z"][baseline_index])
            constant = tangent(data["u_1"], baseline)
            for target in MATCHED_ANGLES:
                common = dict(source=source, training_seed=seed, task=task,
                              target_angle_deg=target)
                indices = [int(np.flatnonzero(mask & (data["family"] == family)
                                              & (data["target_angle"] == target))[0])
                           for family in FAMILIES]
                latent = np.array([unit(data["z"][i]) for i in indices])
                tangents = np.array([tangent(z, baseline) for z in latent])
                residuals, valid = {}, {}
                for family, index, direction in zip(FAMILIES, indices, tangents):
                    residual, norm = remove_constant(direction, constant)
                    valid[family] = norm > RESIDUAL_TOL
                    if valid[family]:
                        residuals[family] = residual / norm
                    directions.append(dict(
                        common, family=family, method=METHODS[family], source_row=index,
                        constant_cosine=float(direction @ constant),
                        retained_norm=norm, retained_energy=norm ** 2,
                        residual_defined=valid[family],
                    ))
                if valid["E2E"] or cosine(tangents[FAMILIES.index("E2E")], constant) < 0.999999:
                    raise ValueError(f"{source}/{task}: E2E is not the constant control")
                for a, b in combinations(range(len(FAMILIES)), 2):
                    f, g = FAMILIES[a], FAMILIES[b]
                    defined = valid[f] and valid[g]
                    pairs.append(dict(
                        common, family_a=f, family_b=g,
                        latent_cosine=cosine(latent[a], latent[b]),
                        latent_distance=float(np.linalg.norm(data["z"][indices[a]].astype(float)
                                                            - data["z"][indices[b]])),
                        unit_latent_distance=float(np.linalg.norm(latent[a] - latent[b])),
                        tangent_cosine=cosine(tangents[a], tangents[b]),
                        residual_defined=defined,
                        residual_cosine=cosine(residuals[f], residuals[g]) if defined else "",
                    ))
                stage_vectors = {"latent": (np.vstack([baseline, latent]),
                                             ("z_r",) + FAMILIES,
                                             [baseline_index] + indices),
                                 "tangent": (tangents, FAMILIES, indices)}
                remaining = tuple(f for f in FAMILIES if valid[f])
                stage_vectors["constant_removed"] = (
                    np.array([residuals[f] for f in remaining]), remaining,
                    [indices[FAMILIES.index(f)] for f in remaining])
                for stage, (vectors, families, row_indices) in stage_vectors.items():
                    if stage != "latent":
                        _, singular, _, energy = decomposition(vectors)
                        for k, (s, fraction) in enumerate(zip(singular, energy), 1):
                            spectra.append(dict(common, stage=stage, n_methods=len(vectors),
                                                component=k, singular_value=float(s),
                                                energy_fraction=float(fraction),
                                                cumulative_energy=float(energy[:k].sum())))
                    mean, _, axes, variance = decomposition(vectors, centered=True)
                    pca_variance.append(dict(common, stage=stage, n_fit_points=len(vectors),
                                             pc1_variance=float(variance[0]),
                                             pc2_variance=float(variance[1]),
                                             top2_variance=float(variance[:2].sum())))
                    projected = list(zip(families, row_indices, vectors, [True] * len(vectors)))
                    # The subsets diagnose raw-latent variability but never influence PCA fitting.
                    if stage == "latent":
                        projected.extend(("C2", int(i), unit(data["z"][i]), False)
                                         for i in np.flatnonzero(mask & (data["family"] == "C2")))
                    for family, index, vector, fitted in projected:
                        scores = (vector - mean) @ axes[:2].T
                        points.append(dict(common, stage=stage, family=family,
                                           source_row=index, fitted=fitted,
                                           pc1=float(scores[0]), pc2=float(scores[1])))
    first = [r for r in spectra if r["component"] == 1]
    tables = {
        "latent_fb_pairs.csv": pairs,
        "latent_fb_directions.csv": directions,
        "latent_fb_spectrum.csv": spectra,
        "latent_fb_pca.csv": points,
        "latent_fb_pca_variance.csv": pca_variance,
        "latent_fb_axis_summary.csv": summarize(first, ("stage", "target_angle_deg"),
                                                ("energy_fraction",)),
        "latent_fb_retention_summary.csv": summarize(directions, ("family", "target_angle_deg"),
                                                     ("retained_energy",)),
    }
    # Q1 supplies baseline/subset statistics already reproduced by its own script.
    return tables


def report_tables(tables):
    lines = ["| 方向矩陣 | 目標角度 | 第一奇異方向能量占比：平均［最小，最大］ |",
             "|---|---|---|"]
    labels = {"tangent": "移除原始 reward 方向，五種方法",
              "constant_removed": "再移除常數 reward 方向，四種方法"}
    for row in tables["latent_fb_axis_summary.csv"]:
        if row["task"] == "all four tasks":
            lines.append(f"| {labels[row['stage']]} | {row['target_angle_deg']:.0f} 度 "
                         f"| {row['energy_fraction_mean']:.3f}［{row['energy_fraction_min']:.3f}，"
                         f"{row['energy_fraction_max']:.3f}］ |")
    retained = ["| Reward 修正 | 移除常數方向後保留的偏移能量：平均［最小，最大］ |",
                "|---|---|"]
    for row in tables["latent_fb_retention_summary.csv"]:
        if row["task"] == "all four tasks" and row["target_angle_deg"] == 10:
            retained.append(f"| {row['family']} | {100 * row['retained_energy_mean']:.2f}%"
                            f"［{100 * row['retained_energy_min']:.2f}%，"
                            f"{100 * row['retained_energy_max']:.2f}%］ |")
    variance = [r["top2_variance"] for r in tables["latent_fb_pca_variance.csv"]
                if r["stage"] == "latent" and r["target_angle_deg"] == 10]
    pca = (f"完整單位 latent 的前兩主成分在各面板保留 "
           f"{100 * min(variance):.1f}%–{100 * max(variance):.1f}% 變異（目標 10 度）。")
    return "\n".join(lines) + "\n\n" + "\n".join(retained) + "\n\n" + pca


def panel_rows(rows, seed, task):
    return [r for r in rows if r["training_seed"] == seed and r["task"] == task
            and r["target_angle_deg"] == 10]


def heatmap_data(tables, seed, task, metric):
    residual = metric == "residual_cosine"
    families = tuple(f for f in FAMILIES if f != "E2E") if residual else FAMILIES
    matrix = np.full((len(families), len(families)), np.nan)
    np.fill_diagonal(matrix, 0 if metric == "latent_distance" else 1)
    if residual:
        for row in panel_rows(tables["latent_fb_directions.csv"], seed, task):
            if row["family"] in families and not row["residual_defined"]:
                index = families.index(row["family"])
                matrix[index, index] = np.nan
    for pair in panel_rows(tables["latent_fb_pairs.csv"], seed, task):
        if pair["family_a"] not in families or pair["family_b"] not in families:
            continue
        if pair[metric] == "":
            continue
        a, b = families.index(pair["family_a"]), families.index(pair["family_b"])
        matrix[a, b] = matrix[b, a] = pair[metric]
    return families, matrix


def figures(tables):
    """Yield deterministic PNGs. Each panel has its own checkpoint/task basis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 130})

    def panels(title):
        fig, axes = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
        fig.suptitle(title + " | target 10 deg | Standard FB only", fontsize=15)
        for row, (seed, _) in enumerate(SOURCES):
            for col, task in enumerate(TASKS):
                axes[row, col].set_title(f"seed {seed} / {task}")
        return fig, axes

    def save(fig):
        stream = BytesIO()
        fig.savefig(stream, format="png", metadata={"Software": "T05 Standard FB latent analysis"})
        plt.close(fig)
        return stream.getvalue()

    for metric, title in (("latent_cosine", "Complete latent cosine"),
                          ("latent_distance", "Complete latent Euclidean distance (norm sqrt(50))"),
                          ("tangent_cosine", "Unit tangent cosine (raw direction removed)"),
                          ("residual_cosine", "Residual direction cosine (raw + constant removed)")):
        fig, axes = panels(title)
        for row, (seed, _) in enumerate(SOURCES):
            for col, task in enumerate(TASKS):
                ax = axes[row, col]
                families, matrix = heatmap_data(tables, seed, task, metric)
                is_distance = metric == "latent_distance"
                lo, hi = (0, 2.5) if is_distance else ((0.93, 1) if metric == "latent_cosine" else (-1, 1))
                im = ax.imshow(matrix, vmin=lo, vmax=hi, cmap="viridis" if is_distance else "coolwarm")
                for a in range(len(families)):
                    for b in range(len(families)):
                        ax.text(b, a, f"{matrix[a, b]:.2f}" if np.isfinite(matrix[a, b]) else "N/A",
                                ha="center", va="center", fontsize=8, color="black",
                                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1))
                ax.set_xticks(range(len(families)), families, rotation=35, ha="right")
                ax.set_yticks(range(len(families)), families)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65)
        yield f"latent_fb_{metric}.png", save(fig)

    for stage, title in (("latent", "PCA of unit latents (subsets projected, not fitted)"),
                         ("tangent", "Centered PCA of unit tangent directions"),
                         ("constant_removed", "Centered PCA after constant removal and renormalization")):
        fig, axes = panels(title)
        for row, (seed, _) in enumerate(SOURCES):
            for col, task in enumerate(TASKS):
                ax = axes[row, col]
                records = [r for r in panel_rows(tables["latent_fb_pca.csv"], seed, task)
                           if r["stage"] == stage]
                var = next(r for r in panel_rows(tables["latent_fb_pca_variance.csv"], seed, task)
                           if r["stage"] == stage)
                base = next((r for r in records if r["family"] == "z_r"), None)
                for r in records:
                    family = r["family"]
                    color = COLORS[FAMILIES.index(family)] if family in FAMILIES else "black"
                    marker = "*" if family == "z_r" else ("x" if family == "C2" else "o")
                    ax.scatter(r["pc1"], r["pc2"], color=color, marker=marker,
                               s=100 if family == "z_r" else 40, alpha=0.5 if family == "C2" else 1)
                    if base and family in FAMILIES:
                        ax.plot([base["pc1"], r["pc1"]], [base["pc2"], r["pc2"]],
                                color=color, alpha=0.5, linewidth=1)
                ax.axhline(0, color="grey", alpha=0.2)
                ax.axvline(0, color="grey", alpha=0.2)
                ax.set_xlabel(f"PC1 ({100 * var['pc1_variance']:.1f}%)")
                ax.set_ylabel(f"PC2 ({100 * var['pc2_variance']:.1f}%)")
                ax.set_aspect("equal", adjustable="datalim")
        handles = [Line2D([], [], color=c, marker="o", linestyle="", label=f)
                   for f, c in zip(FAMILIES, COLORS) if stage != "constant_removed" or f != "E2E"]
        if stage == "latent":
            handles += [Line2D([], [], color="black", marker=m, linestyle="", label=f)
                        for f, m in (("raw reward", "*"), ("subset control", "x"))]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        yield f"latent_fb_pca_{stage}.png", save(fig)

    fig, axes = panels("Uncentered direction SVD: energy fraction per singular component")
    for row, (seed, _) in enumerate(SOURCES):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            for stage, color in zip(STAGES, (COLORS[0], COLORS[1])):
                records = [r for r in panel_rows(tables["latent_fb_spectrum.csv"], seed, task)
                           if r["stage"] == stage]
                ax.plot([r["component"] for r in records], [r["energy_fraction"] for r in records],
                        "o-", label=stage, color=color)
            ax.set(ylim=(-0.02, 1.02), xlabel="Singular component", ylabel="Energy fraction",
                   xticks=range(1, 6))
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    yield "latent_fb_spectrum.png", save(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUTS)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify all CSV and PNG bytes without writing")
    mode.add_argument("--report-tables", action="store_true", help="Print report tables without writing")
    args = parser.parse_args()
    tables = analyze(args.input_dir)
    if args.report_tables:
        print(report_tables(tables))
        return

    for filename, rows in tables.items():
        write_or_check(args.out_dir / filename, csv_text(rows).encode(), args.check)
    for filename, content in figures(tables):
        write_or_check(args.figure_dir / filename, content, args.check)

if __name__ == "__main__":
    main()
