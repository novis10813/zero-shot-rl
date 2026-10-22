"""Paired Q2 analysis of saved standard-FB rollouts, without new evaluation."""

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_q1_fb import (
    MATCHED_ANGLES, METHODS, REPORT_LABELS, ROOT, SOURCES, TASKS, analyze as analyze_q1,
    csv_text, report_tables as q1_report_tables, validate,
)

EPISODES = 20
METRICS = ("return", "speed", "upright", "fall_frac", "sat_frac", "knn_dist", "abs_action")
BASELINE = "Original task reward"


def load_rollouts(path, data):
    """Reject partial, duplicate, mismatched or nonfinite episode records."""
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    indexed = {}
    for record in records:
        row, episode = int(record["row"]), int(record["episode"])
        if not 0 <= row < len(data["z"]) or not 0 <= episode < EPISODES:
            raise ValueError(f"{path}: row/episode outside saved coverage")
        if (row, episode) in indexed:
            raise ValueError(f"{path}: duplicate row/episode")
        for key in ("task", "family"):
            if record[key] != data[key][row]:
                raise ValueError(f"{path}: mismatched {key}")
        for key, saved in (("strength", "strength"), ("target_angle", "target_angle"),
                           ("angle", "angle")):
            # rollout_eval.py writes six significant digits.
            if not np.isclose(float(record[key]), data[saved][row], rtol=7e-6, atol=1e-6, equal_nan=True):
                raise ValueError(f"{path}: mismatched {key}")
        values = {metric: float(record[metric]) for metric in METRICS}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"{path}: nonfinite metric")
        for metric in ("fall_frac", "sat_frac", "abs_action"):
            if not 0 <= values[metric] <= 1:
                raise ValueError(f"{path}: invalid fraction/action magnitude")
        if not -1 <= values["upright"] <= 1 or values["knn_dist"] < 0:
            raise ValueError(f"{path}: invalid uprightness or distance")
        if not 0 <= values["return"] <= 1000:
            raise ValueError(f"{path}: return outside 1000-step task reward range")
        indexed[row, episode] = values
    if len(indexed) != len(data["z"]) * EPISODES:
        raise ValueError(f"{path}: incomplete rollout coverage")
    return indexed


def aggregate_seeds(rows, keys, metrics):
    """Equal-weight seed means, per task only. Ranges are not confidence limits."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for key, group in groups.items():
        if len(group) != len(SOURCES) or len({r["training_seed"] for r in group}) != len(SOURCES):
            raise ValueError("Incomplete or repeated training seeds")
        summary = dict(zip(keys, key))
        summary.update(n_training_seeds=len(group), n_episodes_per_seed=EPISODES)
        for metric in metrics:
            values = np.array([row[metric] for row in group])
            for suffix, value in (("mean", values.mean()), ("min", values.min()), ("max", values.max())):
                summary[f"{metric}_{suffix}"] = float(value)
        result.append(summary)
    return result


def analyze(data_dir):
    # Reuse Q1's convergence, uniqueness and matched-angle validation.
    q1 = analyze_q1(data_dir)
    selected = [row for row in q1["q1_fb_baseline.csv"]
                if row["target_angle_deg"] in MATCHED_ANGLES]
    pairs, seeds, baseline = [], [], []
    for seed, source in SOURCES:
        with np.load(data_dir / source, allow_pickle=False) as archive:
            data = dict(archive)
        validate(data, source)
        rollout_source = source.replace("zs_fb_", "rollouts_fb_").replace(".npz", ".csv")
        indexed = load_rollouts(data_dir / rollout_source, data)
        raw_rows = {}
        for task in TASKS:
            raw_row = int(np.flatnonzero((data["task"] == task) & (data["family"] == "z_r"))[0])
            raw_rows[task] = raw_row
            record = dict(source=rollout_source, training_seed=seed, task=task,
                          baseline_row=raw_row, n_episodes=EPISODES)
            for metric in METRICS:
                record[f"{metric}_mean"] = float(np.mean(
                    [indexed[raw_row, episode][metric] for episode in range(EPISODES)]))
            baseline.append(record)
        for latent in (row for row in selected if row["source"] == source):
            row, task = latent["source_row"], latent["task"]
            raw_row = raw_rows[task]
            common = dict(source=rollout_source, training_seed=seed, task=task,
                          method=latent["method"], target_angle_deg=latent["target_angle_deg"],
                          source_row=row, baseline_row=raw_row, coefficient=latent["coefficient"],
                          achieved_angle_deg=latent["angle_to_raw_deg"])
            group = []
            for episode in range(EPISODES):
                record = dict(common, episode=episode)
                for metric in METRICS:
                    value, reference = indexed[row, episode][metric], indexed[raw_row, episode][metric]
                    record[metric] = value
                    record[f"baseline_{metric}"] = reference
                    record[f"delta_{metric}"] = value - reference
                pairs.append(record)
                group.append(record)
            record = dict(common, n_paired_episodes=EPISODES)
            for metric in METRICS:
                delta = np.array([r[f"delta_{metric}"] for r in group])
                record[f"{metric}_mean"] = float(np.mean([r[metric] for r in group]))
                record[f"baseline_{metric}_mean"] = float(np.mean([r[f"baseline_{metric}"] for r in group]))
                record[f"delta_{metric}_mean"] = float(delta.mean())
                record[f"delta_{metric}_std"] = float(delta.std(ddof=1))
                record[f"delta_{metric}_se"] = float(delta.std(ddof=1) / np.sqrt(EPISODES))
                record[f"delta_{metric}_positive_episodes"] = int((delta > 0).sum())
            seeds.append(record)
    means = tuple(f"{prefix}{metric}_mean" for metric in METRICS for prefix in ("", "baseline_", "delta_"))
    return {
        "q2_fb_pairs.csv": pairs,
        "q2_fb_seed_summary.csv": seeds,
        "q2_fb_task_summary.csv": aggregate_seeds(seeds, ("task", "method", "target_angle_deg"), means),
        "q2_fb_baseline.csv": baseline,
        "q2_fb_baseline_summary.csv": aggregate_seeds(baseline, ("task",), tuple(f"{m}_mean" for m in METRICS)),
    }


def report_tables(tables, q1):
    """Produce the final comparison-table cells and task-specific behavior data."""
    task_labels = {"stand": "站", "walk": "走", "run": "跑", "flip": "翻"}
    baseline = {r["task"]: r for r in tables["q2_fb_baseline_summary.csv"]}
    matched = {(r["method"], r["task"]): r for r in tables["q2_fb_task_summary.csv"]
               if r["target_angle_deg"] == 10}
    outcomes = {}
    outcomes[BASELINE] = ("；".join(f"{task_labels[t]} {baseline[t]['return_mean_mean']:.1f}" for t in TASKS),
                          "平均水平速度：" + "；".join(
                              f"{task_labels[t]} {baseline[t]['speed_mean_mean']:.3f}" for t in TASKS))
    for method in METHODS.values():
        outcomes[method] = ("；".join(
            f"{task_labels[t]} {matched[method, t]['return_mean_mean']:.1f}"
            f"（{matched[method, t]['delta_return_mean_mean']:+.1f}）" for t in TASKS),
            "Δ 平均水平速度：" + "；".join(
                f"{task_labels[t]} {matched[method, t]['delta_speed_mean_mean']:+.3f}" for t in TASKS))
    comparison = q1_report_tables(q1)["comparison"].splitlines()
    method_by_label = {label: method for method, label in REPORT_LABELS.items()}
    method_by_label["原始任務 reward（基準）"] = BASELINE
    for index in range(2, len(comparison)):
        cells = [cell.strip() for cell in comparison[index].strip("|").split("|")]
        cells[2], cells[4] = outcomes[method_by_label[cells[0]]]
        comparison[index] = "| " + " | ".join(cells) + " |"
    behavior = [
        "| 任務 | Reward 類型 | Δ return：平均［seed 最小，最大］ | Δ 水平速度 | Δ 直立程度 | Δ 低軀幹高度時間比例（百分點） | Δ 動作飽和比例（百分點） | Δ 最近訓練樣本距離 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for task in TASKS:
        for method, label in REPORT_LABELS.items():
            row = matched[method, task]
            behavior.append(
                f"| {task_labels[task]} | {label} | {row['delta_return_mean_mean']:+.1f}"
                f"［{row['delta_return_mean_min']:+.1f}，{row['delta_return_mean_max']:+.1f}］"
                f" | {row['delta_speed_mean_mean']:+.3f} | {row['delta_upright_mean_mean']:+.3f}"
                f" | {100 * row['delta_fall_frac_mean_mean']:+.2f}"
                f" | {100 * row['delta_sat_frac_mean_mean']:+.2f}"
                f" | {row['delta_knn_dist_mean_mean']:+.3f} |"
            )
    reference = [
        "| 任務 | Return | 水平速度 | 直立程度 | 低軀幹高度時間比例（%） | 動作飽和比例（%） | 最近訓練樣本距離 |",
        "|---|---|---|---|---|---|---|",
    ]
    for task in TASKS:
        row = baseline[task]
        reference.append(
            f"| {task_labels[task]} | {row['return_mean_mean']:.1f} | {row['speed_mean_mean']:.3f}"
            f" | {row['upright_mean_mean']:.3f} | {100 * row['fall_frac_mean_mean']:.2f}"
            f" | {100 * row['sat_frac_mean_mean']:.2f} | {row['knn_dist_mean_mean']:.3f} |"
        )
    uncertainty = [
        "| 任務 | Reward 修正 | 訓練 seed | Δ return：平均 ± SE | Δ 水平速度：平均 ± SE | Δ 直立程度：平均 ± SE |",
        "|---|---|---|---|---|---|",
    ]
    for row in tables["q2_fb_seed_summary.csv"]:
        if row["target_angle_deg"] != 10 or (row["task"], row["method"]) not in (
                ("run", METHODS["CQL"]), ("flip", METHODS["ReBRAC"])):
            continue
        uncertainty.append(
            f"| {task_labels[row['task']]} | {REPORT_LABELS[row['method']]} | {row['training_seed']}"
            f" | {row['delta_return_mean']:+.1f} ± {row['delta_return_se']:.1f}"
            f" | {row['delta_speed_mean']:+.3f} ± {row['delta_speed_se']:.3f}"
            f" | {row['delta_upright_mean']:+.3f} ± {row['delta_upright_se']:.3f} |"
        )
    return {"comparison": "\n".join(comparison), "baseline_behavior": "\n".join(reference),
            "behavior": "\n".join(behavior), "uncertainty": "\n".join(uncertainty)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report-tables", action="store_true")
    args = parser.parse_args()
    outputs = analyze(args.data_dir)
    if args.report_tables:
        for content in report_tables(outputs, analyze_q1(args.data_dir)).values():
            print(content, end="\n\n")
        return
    if not args.check:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in outputs.items():
        path, content = args.out_dir / filename, csv_text(rows)
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Stale or missing output: {path}")
        else:
            path.write_text(content)
        print(f"{'Verified' if args.check else 'Wrote'} {filename}: {len(rows)} rows")


if __name__ == "__main__":
    main()
