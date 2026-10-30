"""Saved Standard-FB coefficient trajectories and candidate-AX behavior sweeps.

No inference or rollout. Normal generation reads data/inputs/ only. --export-missing
extracts a sanitized missing-target input from the three archived inference logs.
AX is a constructed rotation axis, not a validated conservatism scale.

Run from the repository root:
    PYTHONPATH=.:tasks/T05-opt-to-pessi .venv/bin/python -m analyze.sweeps [--check]
"""

import argparse
import csv
from io import BytesIO
from pathlib import Path
import re

import numpy as np

from analyze.q1 import analyze as analyze_q1
from analyze.q2 import EPISODES, METRICS, aggregate_seeds, load_rollouts
from common import (FIGURES, INPUTS, METHODS, MODELS, OUTPUTS, ROOT, SOURCES, TASKS, csv_text,
                    write_or_check)
from geometry import angle_deg, cosine, tangent, unit

TARGETS = (5., 10., 20., 35., 50., 70.)
AX_ANGLES = (-50., -35., -20., -10., 0., 10., 20., 35., 50.)
MISSING_FILE = "sweep_fb_missing.csv"
STATUS_LABELS = {
    "unbracketed_grid": "收斂網格未括住目標",
    "bisection_not_converged": "二分搜尋求解未收斂",
    "target_mismatch": "重試後仍未匹配目標",
}
NUMERIC_TOL = 1e-8  # numerical ties only, not practical or statistical equivalence
METRIC_LABELS = {
    "return": "原始任務 return", "speed": "水平速度", "upright": "直立程度",
    "fall_frac": "低軀幹高度時間比例", "sat_frac": "動作飽和比例",
    "knn_dist": "最近訓練樣本距離", "abs_action": "動作絕對值平均",
}


def missing_keys(data):
    expected = {(task, family, target) for task in TASKS for family in METHODS for target in TARGETS}
    observed = {(str(t), str(f), float(a)) for t, f, a in
                zip(data["task"], data["family"], data["target_angle"]) if f in METHODS}
    if not observed <= expected:
        raise ValueError("Unexpected family/target coverage")
    return expected - observed


def parse_missing_log(text, seed, source, data):
    """Export only structured failed-target facts, never raw logs or host paths."""
    pattern = re.compile(
        r"^(stand|walk|run|flip)\s+(CQL|ReBRAC\+E2E|ReBRAC|E2E-bi|E2E)\s+"
        r"target\s+([\d.]+)\s+(unreachable \(max angle ([\d.]+)\)|no fixed point|discontinuous)$")
    rows, seen = [], set()
    for line_number, line in enumerate(text.splitlines(), 1):
        match = pattern.fullmatch(line.strip())
        if not match:
            continue
        task, family, target, reason, maximum = match.groups()
        key = (task, family, float(target))
        if key in seen:
            raise ValueError(f"{source}: repeated missing target")
        seen.add(key)
        status = ("unbracketed_grid" if maximum else
                  "bisection_not_converged" if reason == "no fixed point" else "target_mismatch")
        rows.append(dict(source=source, training_seed=seed, task=task, family=family,
                         target_angle_deg=float(target), status=status,
                         converged_grid_max_angle_deg=float(maximum) if maximum else "",
                         source_log_line=line_number))
    if seen != missing_keys(data):
        raise ValueError(f"{source}: log missing-target coverage differs from NPZ")
    return rows


def load_missing(data_dir):
    with (data_dir / MISSING_FILE).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    sources = {source: seed for seed, source in SOURCES}
    for row in rows:
        source, seed = row["source"], int(row["training_seed"])
        key = (source, row["task"], row["family"], float(row["target_angle_deg"]))
        if source not in sources or seed != sources[source] or key in result:
            raise ValueError("Invalid or duplicate missing-target source")
        if row["status"] not in STATUS_LABELS or int(row["source_log_line"]) <= 0:
            raise ValueError("Invalid missing-target status/provenance")
        maximum = row["converged_grid_max_angle_deg"]
        if row["status"] == "unbracketed_grid":
            if maximum == "" or not 0 <= float(maximum) < key[-1]:
                raise ValueError("Invalid maximum converged-grid angle")
        elif maximum != "":
            raise ValueError("Unexpected converged-grid maximum")
        result[key] = row
    return result


def rank_correlation(x, y):
    """Spearman ranks with average ties, without significance testing."""
    def ranks(values):
        values = np.asarray(values)
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[order[stop]] == values[order[start]]:
                stop += 1
            result[order[start:stop]] = (start + stop - 1) / 2
            start = stop
        return result
    a, b = ranks(x), ranks(y)
    a -= a.mean()
    b -= b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / norm, -1, 1)) if norm > 0 else ""


def trend(values):
    values = np.asarray(values, dtype=float)
    differences = np.diff(values)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite curve with at least two points")
    signs = np.sign(differences[np.abs(differences) > NUMERIC_TOL])
    positive, negative = int(np.sum(signs > 0)), int(np.sum(signs < 0))
    label = ("mixed" if positive and negative else "nondecreasing" if positive else
             "nonincreasing" if negative else "constant")
    return dict(n_increases=positive, n_decreases=negative,
                n_numerical_ties=int(len(differences) - len(signs)),
                sign_changes=int(np.sum(signs[1:] != signs[:-1])), trend=label,
                net_change=float(values[-1] - values[0]),
                total_variation=float(np.abs(differences).sum()),
                max_abs_adjacent_change=float(np.abs(differences).max()))


def candidate_axis(data, task):
    mask = data["task"] == task
    raw_index = int(np.flatnonzero(mask & (data["family"] == "z_r"))[0])
    raw = unit(data["z"][raw_index])
    directions = {}
    for family in ("CQL", "ReBRAC", "E2E-bi"):
        indices = np.flatnonzero(mask & (data["family"] == family) & (data["target_angle"] == 10))
        if len(indices) != 1:
            raise ValueError("Missing or duplicate AX construction endpoint")
        directions[family] = tangent(data["z"][indices[0]], raw)
    axis = directions["E2E-bi"] - (directions["CQL"] + directions["ReBRAC"]) / 2
    axis = unit(axis - (axis @ raw) * raw)
    indices = np.flatnonzero(mask & (data["family"] == "AX"))
    targets = list(map(float, data["target_angle"][indices]))
    if len(targets) != 8 or set(targets) != set(AX_ANGLES) - {0.}:
        raise ValueError("Incomplete or duplicated AX rotation angles")
    selection = {0.: raw_index}
    for i in indices:
        target = float(data["target_angle"][i])
        theta = np.deg2rad(target)
        expected = np.cos(theta) * raw + np.sin(theta) * axis
        if (not np.allclose(unit(data["z"][i]), expected, atol=2e-6, rtol=0)
                or not data["converged"][i] or data["multi"][i]
                or data["strength"][i] != 0
                or abs(float(data["angle"][i]) - abs(target)) > 0.01):
            raise ValueError("Saved AX latent differs from constructed rotation")
        selection[target] = int(i)
    return raw_index, selection


def analyze(data_dir):
    q1 = analyze_q1(data_dir)  # saved-solution validation, including all method strengths
    missing = load_missing(data_dir)
    points, adjacent, curves, coverage = [], [], [], []
    axis_pairs, axis_seeds, axis_steps, axis_trends = [], [], [], []
    expected_missing = set()
    for seed, source in SOURCES:
        with np.load(data_dir / source, allow_pickle=False) as archive:
            data = dict(archive)
        expected_missing |= {(source,) + key for key in missing_keys(data)}
        rollout_source = source.replace("zs_fb_", "rollouts_fb_").replace(".npz", ".csv")
        rollouts = load_rollouts(data_dir / rollout_source, data)
        for task in TASKS:
            common = dict(source=source, training_seed=seed, task=task)
            raw_index, axis_selection = candidate_axis(data, task)
            raw = data["z"][raw_index]
            for family, method in METHODS.items():
                group = [r for r in q1["q1_fb_baseline.csv"] if r["source"] == source
                         and r["task"] == task and r["method"] == method]
                group.sort(key=lambda r: r["coefficient"])
                if not group or not all(np.isfinite(r["coefficient"]) and r["coefficient"] > 0 for r in group):
                    raise ValueError("Expected positive finite coefficients")
                coefficients = [0.] + [r["coefficient"] for r in group]
                if np.any(np.diff(coefficients) <= 0):
                    raise ValueError("Nonunique coefficients")
                indices = [raw_index] + [r["source_row"] for r in group]
                targets = [0.] + [r["target_angle_deg"] for r in group]
                angles = [0.] + [r["angle_to_raw_deg"] for r in group]
                tangents = {i: tangent(data["z"][i], raw) for i in indices[1:]}
                for index, coefficient, target, angle in zip(indices, coefficients, targets, angles):
                    points.append(dict(common, family=family, source_row=index, coefficient=coefficient,
                                       target_angle_deg=target, angle_to_raw_deg=angle,
                                       direction_from_first_deg=angle_deg(tangents[index], tangents[indices[1]])
                                       if index != raw_index else ""))
                for j in range(1, len(indices)):
                    a, b = indices[j - 1:j + 1]
                    gap = sum(min(targets[j-1], targets[j]) < t < max(targets[j-1], targets[j]) for t in TARGETS)
                    adjacent.append(dict(
                        common, family=family, from_row=a, to_row=b,
                        from_coefficient=coefficients[j-1], to_coefficient=coefficients[j],
                        coefficient_gap=coefficients[j] - coefficients[j-1],
                        from_target_angle_deg=targets[j-1], to_target_angle_deg=targets[j],
                        skipped_target_count=gap,
                        raw_angle_increment_deg=angles[j] - angles[j-1],
                        latent_step_angle_deg=angle_deg(data["z"][a], data["z"][b]),
                        tangent_cosine=cosine(tangents[a], tangents[b]) if j > 1 else "",
                        tangent_turn_deg=angle_deg(tangents[a], tangents[b]) if j > 1 else "",
                    ))
                turns = [r["tangent_turn_deg"] for r in adjacent
                         if r["source"] == source and r["task"] == task and r["family"] == family
                         and r["tangent_turn_deg"] != ""]
                curves.append(dict(common, family=family, n_saved=len(group),
                                   n_unavailable=len(TARGETS) - len(group),
                                   max_saved_target_deg=max(targets),
                                   n_raw_angle_decreases=int(np.sum(np.diff(angles) < -NUMERIC_TOL)),
                                   max_adjacent_tangent_turn_deg=float(max(turns)),
                                   max_direction_from_first_deg=float(max(
                                       angle_deg(tangents[i], tangents[indices[1]]) for i in indices[1:]))))
                saved = {r["target_angle_deg"]: r for r in group}
                for target in TARGETS:
                    if target in saved:
                        record = dict(status="saved", coefficient=saved[target]["coefficient"],
                                      source_row=saved[target]["source_row"], converged_grid_max_angle_deg="")
                    else:
                        failure = missing.get((source, task, family, target))
                        if failure is None:
                            raise ValueError("Missing sanitized target-status record")
                        record = dict(status=failure["status"], coefficient="", source_row="",
                                      converged_grid_max_angle_deg=failure["converged_grid_max_angle_deg"])
                    coverage.append(dict(common, family=family, target_angle_deg=target, **record))

            # Historical rollout pairs: each angle shares the same initialization seeds.
            episode_values = {}
            for angle in AX_ANGLES:
                index = axis_selection[angle]
                context = dict(source=rollout_source, training_seed=seed, task=task,
                               signed_angle_deg=angle, source_row=index, baseline_row=raw_index)
                rows = []
                for episode in range(EPISODES):
                    record = dict(context, episode=episode)
                    for metric in METRICS:
                        value, baseline = rollouts[index, episode][metric], rollouts[raw_index, episode][metric]
                        record[metric] = value
                        record[f"baseline_{metric}"] = baseline
                        record[f"delta_{metric}"] = value - baseline
                    rows.append(record)
                axis_pairs.extend(rows)
                summary = dict(context, n_paired_episodes=EPISODES)
                for metric in METRICS:
                    values = np.array([r[metric] for r in rows])
                    delta = np.array([r[f"delta_{metric}"] for r in rows])
                    summary[f"{metric}_mean"] = float(values.mean())
                    summary[f"{metric}_se"] = float(values.std(ddof=1) / np.sqrt(EPISODES))
                    summary[f"delta_{metric}_mean"] = float(delta.mean())
                    summary[f"delta_{metric}_se"] = float(delta.std(ddof=1) / np.sqrt(EPISODES))
                    episode_values[angle, metric] = values
                axis_seeds.append(summary)
            for metric in METRICS:
                means = [float(episode_values[angle, metric].mean()) for angle in AX_ANGLES]
                axis_trends.append(dict(source=rollout_source, training_seed=seed, task=task,
                                        metric=metric, n_angles=len(AX_ANGLES), n_steps=len(AX_ANGLES)-1,
                                        spearman_rho=rank_correlation(AX_ANGLES, means), **trend(means)))
                for a, b in zip(AX_ANGLES, AX_ANGLES[1:]):
                    delta = episode_values[b, metric] - episode_values[a, metric]
                    axis_steps.append(dict(
                        source=rollout_source, training_seed=seed, task=task, metric=metric,
                        from_angle_deg=a, to_angle_deg=b, angle_gap_deg=b-a,
                        from_row=axis_selection[a], to_row=axis_selection[b],
                        mean_paired_change=float(delta.mean()),
                        paired_change_se=float(delta.std(ddof=1)/np.sqrt(EPISODES)),
                        n_positive_episodes=int(np.sum(delta > 0)),
                        n_negative_episodes=int(np.sum(delta < 0)),
                    ))
    if set(missing) != expected_missing:
        raise ValueError("Missing-target file has stale or extra coverage")
    method_summary = []
    for family in METHODS:
        c = [r for r in curves if r["family"] == family]
        steps = [r["tangent_turn_deg"] for r in adjacent if r["family"] == family and r["tangent_turn_deg"] != ""]
        small = [r["tangent_turn_deg"] for r in adjacent if r["family"] == family
                 and r["from_target_angle_deg"] == 5 and r["to_target_angle_deg"] == 10]
        record = dict(family=family, n_curves=len(c), n_saved=sum(r["n_saved"] for r in c),
                      n_unavailable=sum(r["n_unavailable"] for r in c),
                      max_saved_target_min_deg=min(r["max_saved_target_deg"] for r in c),
                      max_saved_target_max_deg=max(r["max_saved_target_deg"] for r in c),
                      n_raw_angle_decreases=sum(r["n_raw_angle_decreases"] for r in c),
                      n_adjacent_tangent_steps=len(steps))
        for label, values in (("tangent_turn_deg", steps), ("five_to_ten_turn_deg", small)):
            for suffix, func in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                record[f"{label}_{suffix}"] = float(func(values))
        method_summary.append(record)
    trend_summary = []
    for metric in METRICS:
        rows = [r for r in axis_trends if r["metric"] == metric]
        record = dict(metric=metric, n_curves=len(rows), n_training_seeds=len(SOURCES), n_tasks=len(TASKS))
        for label in ("mixed", "nondecreasing", "nonincreasing", "constant"):
            record[f"n_{label}"] = sum(r["trend"] == label for r in rows)
        rho = [r["spearman_rho"] for r in rows if r["spearman_rho"] != ""]
        record.update(spearman_min=float(min(rho)), spearman_max=float(max(rho)),
                      n_positive_net_change=sum(r["net_change"] > NUMERIC_TOL for r in rows),
                      n_negative_net_change=sum(r["net_change"] < -NUMERIC_TOL for r in rows))
        trend_summary.append(record)
    return {
        "sweep_fb_coverage.csv": coverage,
        "sweep_fb_points.csv": points,
        "sweep_fb_adjacent.csv": adjacent,
        "sweep_fb_curves.csv": curves,
        "sweep_fb_method_summary.csv": method_summary,
        "axis_fb_pairs.csv": axis_pairs,
        "axis_fb_seed_summary.csv": axis_seeds,
        "axis_fb_task_summary.csv": aggregate_seeds(
            axis_seeds, ("task", "signed_angle_deg"),
            tuple(f"{prefix}{metric}_mean" for metric in METRICS for prefix in ("", "delta_"))),
        "axis_fb_adjacent.csv": axis_steps,
        "axis_fb_trends.csv": axis_trends,
        "axis_fb_trend_summary.csv": trend_summary,
    }


def report_tables(tables):
    coverage = tables["sweep_fb_coverage.csv"]
    statuses = [f"{STATUS_LABELS[status]} {sum(r['status'] == status for r in coverage)} 個"
                for status in STATUS_LABELS if any(r["status"] == status for r in coverage)]
    counts = (f"共 {len(coverage)} 個方法／模型／任務／目標角度組合，"
              f"保存 {sum(r['status'] == 'saved' for r in coverage)} 個；" + "，".join(statuses) + "。")
    methods = ["| Reward 修正 | 已保存／目標組合 | 各模型任務的最大已存目標角度範圍 | 相鄰偏移方向轉角：平均［最小，最大］ | 5→10 度偏移方向轉角：平均［最小，最大］ |",
               "|---|---|---|---|---|"]
    for r in tables["sweep_fb_method_summary.csv"]:
        methods.append(
            f"| {r['family']} | {r['n_saved']}/{r['n_saved']+r['n_unavailable']} "
            f"| {r['max_saved_target_min_deg']:.0f}–{r['max_saved_target_max_deg']:.0f} 度 "
            f"| {r['tangent_turn_deg_mean']:.3f}［{r['tangent_turn_deg_min']:.3f}，{r['tangent_turn_deg_max']:.3f}］度 "
            f"| {r['five_to_ten_turn_deg_mean']:.3f}［{r['five_to_ten_turn_deg_min']:.3f}，{r['five_to_ten_turn_deg_max']:.3f}］度 |")
    axis = ["| 行為指標 | 非遞減曲線數 | 非遞增曲線數 | 有升有降曲線數 | Spearman 相關範圍 | 正端點高於／低於負端點的曲線數 |",
            "|---|---|---|---|---|---|"]
    for r in tables["axis_fb_trend_summary.csv"]:
        axis.append(f"| {METRIC_LABELS[r['metric']]} | {r['n_nondecreasing']} | {r['n_nonincreasing']} "
                    f"| {r['n_mixed']} | {r['spearman_min']:.3f}–{r['spearman_max']:.3f} "
                    f"| {r['n_positive_net_change']}／{r['n_negative_net_change']} |")
    return {"coverage": counts, "methods": "\n".join(methods), "axis": "\n".join(axis)}


def figures(tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 130})
    colors = ("#0072B2", "#D55E00", "#009E73")

    def png(fig):
        stream = BytesIO()
        fig.savefig(stream, format="png", metadata={"Software": "T05 Standard FB sweep analysis"})
        plt.close(fig)
        return stream.getvalue()

    for kind in ("coefficient_angle", "adjacent_tangent"):
        fig, axes = plt.subplots(5, 4, figsize=(18, 18), constrained_layout=True)
        fig.suptitle("Standard FB: " + ("coefficient vs raw-latent angle" if kind == "coefficient_angle"
                                       else "adjacent unit-tangent turn vs later coefficient")
                     + " | saved discrete samples only", fontsize=15)
        for row, family in enumerate(METHODS):
            for col, task in enumerate(TASKS):
                ax = axes[row, col]
                for (seed, _), color in zip(SOURCES, colors):
                    filename = "sweep_fb_points.csv" if kind == "coefficient_angle" else "sweep_fb_adjacent.csv"
                    records = [r for r in tables[filename] if r["training_seed"] == seed
                               and r["task"] == task and r["family"] == family]
                    if kind == "coefficient_angle":
                        records = [r for r in records if r["coefficient"] > 0]
                        x = [r["coefficient"] for r in records]
                        y = [r["angle_to_raw_deg"] for r in records]
                    else:
                        records = [r for r in records if r["tangent_turn_deg"] != ""]
                        x = [r["to_coefficient"] for r in records]
                        y = [r["tangent_turn_deg"] for r in records]
                    ax.plot(x, y, "o--", color=color, label=f"seed {seed}", markersize=4, linewidth=1)
                ax.set(xscale="log", title=f"{family} / {task}", xlabel="Coefficient" if kind == "coefficient_angle" else "Later coefficient",
                       ylabel="Angle to raw (deg)" if kind == "coefficient_angle" else "Adjacent tangent turn (deg)")
                # Narrow log ranges otherwise produce overlapping minor tick labels.
                coefficients = np.concatenate([line.get_xdata() for line in ax.lines])
                ticks = np.geomspace(coefficients.min(), coefficients.max(), 3)
                ax.set_xticks(ticks, [f"{value:.3g}" for value in ticks])
                ax.set_xticks([], minor=True)
                ax.grid(alpha=0.2)
        handles = [Line2D([], [], color=color, marker="o", linestyle="--", label=f"seed {seed}")
                   for (seed, _), color in zip(SOURCES, colors)]
        fig.legend(handles=handles, loc="outside lower center", ncol=3)
        yield f"sweep_fb_{kind}.png", png(fig)

    fig, axes = plt.subplots(5, 3, figsize=(14, 15), constrained_layout=True)
    fig.suptitle("Target coverage: saved / unbracketed on converged grid / bisection not converged", fontsize=13)
    codes = {"saved": 0, "unbracketed_grid": 1, "bisection_not_converged": 2, "target_mismatch": 3}
    cmap = ListedColormap(["#56B4E9", "#DDDDDD", "#D55E00", "#CC79A7"])
    for row, family in enumerate(METHODS):
        for col, (seed, _) in enumerate(SOURCES):
            ax = axes[row, col]
            matrix = np.full((4, 6), np.nan)
            for r in tables["sweep_fb_coverage.csv"]:
                if r["training_seed"] == seed and r["family"] == family:
                    matrix[TASKS.index(r["task"]), TARGETS.index(r["target_angle_deg"])] = codes[r["status"]]
            ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
            for i in range(4):
                for j in range(6):
                    ax.text(j, i, ("S", "U", "N", "D")[int(matrix[i,j])], ha="center", va="center")
            ax.set_xticks(range(6), [f"{t:g}" for t in TARGETS])
            ax.set_yticks(range(4), TASKS)
            ax.set(title=f"{family} / seed {seed}", xlabel="Target angle (deg)")
    yield "sweep_fb_coverage.png", png(fig)

    for metric in METRICS:
        fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
        fig.suptitle(f"Standard FB candidate AX rotation: {metric} | mean +/- episode SE | not an optimism scale", fontsize=14)
        for row, (seed, _) in enumerate(SOURCES):
            for col, task in enumerate(TASKS):
                ax = axes[row, col]
                records = [r for r in tables["axis_fb_seed_summary.csv"]
                           if r["training_seed"] == seed and r["task"] == task]
                x = [r["signed_angle_deg"] for r in records]
                y = [r[f"{metric}_mean"] for r in records]
                error = [r[f"{metric}_se"] for r in records]
                ax.errorbar(x, y, yerr=error, fmt="o--", capsize=3, color=colors[row], linewidth=1)
                baseline = next(r[f"{metric}_mean"] for r in records if r["signed_angle_deg"] == 0)
                ax.axhline(baseline, color="black", linestyle=":", linewidth=0.7)
                ax.axvline(0, color="grey", alpha=0.3)
                ax.set(title=f"seed {seed} / {task}", xlabel="Signed candidate-axis angle (deg)",
                       ylabel=metric, xticks=AX_ANGLES)
                ax.tick_params(axis="x", labelsize=8)
                ax.grid(alpha=0.2)
        yield f"axis_fb_{metric}.png", png(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUTS)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--log-dir", type=Path, default=ROOT.parent.parent / "artifacts" / ROOT.name)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--report-tables", action="store_true")
    mode.add_argument("--export-missing", action="store_true", help="Extract missing targets from archived inference logs only")
    args = parser.parse_args()
    if args.export_missing:
        rows = []
        analyze_q1(args.input_dir)
        for (seed, source), model in zip(SOURCES, MODELS):
            with np.load(args.input_dir / source, allow_pickle=False) as archive:
                data = dict(archive)
            log = args.log_dir / model["job"] / "output.log"
            rows.extend(parse_missing_log(log.read_text(), seed, source, data))
        args.input_dir.mkdir(parents=True, exist_ok=True)
        (args.input_dir / MISSING_FILE).write_text(csv_text(rows))
        print(f"Exported {len(rows)} structured missing-target records")
        return
    tables = analyze(args.input_dir)
    if args.report_tables:
        for text in report_tables(tables).values():
            print(text, end="\n\n")
        return

    for name, rows in tables.items():
        write_or_check(args.out_dir / name, csv_text(rows).encode(), args.check)
    for name, content in figures(tables):
        write_or_check(args.figure_dir / name, content, args.check)

if __name__ == "__main__":
    main()
