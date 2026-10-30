"""Task-specific metrics from NEW fixed-FB replay traces, never historical CSVs.

Run from the repository root:
    PYTHONPATH=.:tasks/T05-opt-to-pessi .venv/bin/python -m analyze.task_behavior [--check]
"""

import argparse
from pathlib import Path

import numpy as np

from common import (INPUTS, METHODS, MODELS, OUTPUTS, REPORT_LABELS, SOURCES, TASKS, csv_text,
                    validate, write_or_check)
from zero_shot_rl.latent.archive import matched_rows
from zero_shot_rl.rollout import trace_format
from zero_shot_rl.rollout.trace_format import DT, EPISODES

WALK_SPEED = 1.0
METRICS = ("return", "speed", "net_displacement_estimate", "walk_speed_mae", "height", "upright",
           "positive_turns", "negative_turns", "net_turns", "angular_momentum")


def behavior_rows(data):
    """Exactly one raw latent and each of five 10-degree corrections per task."""
    return matched_rows(data, TASKS, ("z_r", *METHODS), 10.0)


def rotation_metrics(angles):
    """Count +/- complete turns from a reset anchor, cancelling partial reversals.

    Positive is rotation about world +Y, matching Walker's flip angular momentum.
    The anchor moves by +/-2pi only after a full revolution. Never sum absolute
    per-step angle changes: oscillation can accumulate distance without a turn.
    A full forward turn followed by a full backward turn counts once in each
    direction. Net turns retain fractional turns and can cancel.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 1 or len(angles) < 2 or not np.isfinite(angles).all():
        raise ValueError("Expected a finite one-dimensional angle trajectory")
    continuous = np.unwrap(angles)
    anchor = continuous[0]
    positive = negative = 0
    tau = 2 * np.pi
    for value in continuous[1:]:
        while value - anchor >= tau - 1e-10:
            positive += 1
            anchor += tau
        while value - anchor <= -tau + 1e-10:
            negative += 1
            anchor -= tau
    return positive, negative, float((continuous[-1] - continuous[0]) / tau)


def episode_metrics(trace, latent, episode):
    positive, negative, net = rotation_metrics(trace["angle"][latent, episode])
    return dict(
        return_=float(trace["reward"][latent, episode].sum(dtype=np.float64)),
        speed=float(trace["speed"][latent, episode].mean(dtype=np.float64)),
        # Right-endpoint velocity integration, not measured end-minus-start position.
        net_displacement_estimate=float(trace["speed"][latent, episode].sum(dtype=np.float64) * DT),
        walk_speed_mae=float(np.abs(trace["speed"][latent, episode].astype(np.float64)
                                    - WALK_SPEED).mean()),
        height=float(trace["height"][latent, episode].mean(dtype=np.float64)),
        upright=float(trace["upright"][latent, episode].mean(dtype=np.float64)),
        positive_turns=positive, negative_turns=negative, net_turns=net,
        angular_momentum=float(trace["angular_momentum"][latent, episode].mean(dtype=np.float64)),
    )


def trace_name(seed):
    return next(Path(m["trace"]).name for m in MODELS if m["training_seed"] == seed)


def validate_trace(trace, data):
    trace_format.validate_trace(trace, data, behavior_rows(data))


def analyze(data_dir):
    episodes, seeds = [], []
    for seed, source in SOURCES:
        with np.load(data_dir / source, allow_pickle=False) as archive:
            data = dict(archive)
        validate(data, source)
        with np.load(data_dir / trace_name(seed), allow_pickle=False) as archive:
            trace = dict(archive)  # decompress each array once, not once per episode
        validate_trace(trace, data)
        for index, row in enumerate(behavior_rows(data)):
            common = dict(training_seed=seed, task=str(data["task"][row]),
                          family=str(data["family"][row]), source_row=row)
            for episode in range(EPISODES):
                metrics = episode_metrics(trace, index, episode)
                metrics["return"] = metrics.pop("return_")
                episodes.append(dict(common, episode=episode, **metrics))
    indexed = {(r["training_seed"], r["task"], r["family"], r["episode"]): r for r in episodes}
    for record in episodes:
        reference = indexed[record["training_seed"], record["task"], "z_r", record["episode"]]
        for metric in METRICS:
            record[f"delta_{metric}"] = record[metric] - reference[metric]
    for seed, _ in SOURCES:
        for task in TASKS:
            for family in ("z_r", *METHODS):
                group = [indexed[seed, task, family, episode] for episode in range(EPISODES)]
                summary = dict(training_seed=seed, task=task, family=family, n_episodes=EPISODES)
                for metric in METRICS:
                    values = np.array([r[metric] for r in group])
                    delta = np.array([r[f"delta_{metric}"] for r in group])
                    summary[f"{metric}_mean"] = float(values.mean())
                    summary[f"delta_{metric}_mean"] = float(delta.mean())
                    summary[f"delta_{metric}_se"] = float(delta.std(ddof=1) / np.sqrt(EPISODES))
                seeds.append(summary)
    tasks = []
    for task in TASKS:
        for family in ("z_r", *METHODS):
            group = [r for r in seeds if r["task"] == task and r["family"] == family]
            summary = dict(task=task, family=family, n_training_seeds=len(group), n_episodes_per_seed=EPISODES)
            for metric in METRICS:
                for prefix in ("", "delta_"):
                    values = np.array([r[f"{prefix}{metric}_mean"] for r in group])
                    for suffix, value in (("mean", values.mean()), ("min", values.min()), ("max", values.max())):
                        summary[f"{prefix}{metric}_{suffix}"] = float(value)
            tasks.append(summary)
    return {"behavior_fb_episodes.csv": episodes, "behavior_fb_seed_summary.csv": seeds,
            "behavior_fb_task_summary.csv": tasks}


def report_table(tables):
    rows = ["| 任務 | Reward 修正 | Return（Δ） | 主要行為指標（Δ） | 補充指標 |",
            "|---|---|---:|---|---|"]
    for row in tables["behavior_fb_task_summary.csv"]:
        def metric(key, digits=3):
            return f"{row[key + '_mean']:.{digits}f}（{row['delta_' + key + '_mean']:+.{digits}f}）"
        task = row["task"]
        if task == "stand":
            main, extra = f"高度 {metric('height')} m", f"直立 {metric('upright')}"
        elif task == "walk":
            main = f"距離／速度 {metric('net_displacement_estimate', 2)} m ／ {metric('speed')} m/s"
            extra = f"距 1 m/s 的逐步 MAE {metric('walk_speed_mae')} m/s"
        elif task == "run":
            main = f"距離／速度 {metric('net_displacement_estimate', 2)} m ／ {metric('speed')} m/s"
            extra = f"直立 {metric('upright')}"
        else:
            main = f"正向完整圈數 {metric('positive_turns', 2)}；反向 {metric('negative_turns', 2)}"
            extra = f"淨旋轉 {metric('net_turns', 2)} 圈；角動量 {metric('angular_momentum', 2)}"
        family = row["family"]
        label = "原始 reward" if family == "z_r" else REPORT_LABELS[METHODS[family]]
        rows.append(f"| {dict(stand='站', walk='走', run='跑', flip='翻')[task]} | {label} | {metric('return', 1)} | {main} | {extra} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUTS)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report-tables", action="store_true")
    args = parser.parse_args()
    tables = analyze(args.input_dir)
    if args.report_tables:
        print(report_table(tables))
        return
    for name, rows in tables.items():
        write_or_check(args.out_dir / name, csv_text(rows).encode(), args.check)

if __name__ == "__main__":
    main()
