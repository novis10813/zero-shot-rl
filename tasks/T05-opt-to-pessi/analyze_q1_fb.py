"""Q1 geometry of existing standard-FB latents, without inference or rollouts.

Run from the repository root:
    .venv/bin/python tasks/T05-opt-to-pessi/analyze_q1_fb.py
    .venv/bin/python tasks/T05-opt-to-pessi/analyze_q1_fb.py --check
"""

import argparse
import csv
import io
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TASKS = ("stand", "walk", "run", "flip")
# Filenames and training seeds are recorded in exp/T05-zero-shot-rl-{30,31,32}.
SOURCES = ((0, "zs_fb_zero-shot-rl-30.npz"),
           (1, "zs_fb_zero-shot-rl-31.npz"),
           (2, "zs_fb_zero-shot-rl-32.npz"))
METHODS = {
    "CQL": "CQL implicit-reward projection",
    "ReBRAC": "ReBRAC action-deviation penalty",
    "ReBRAC+E2E": "ReBRAC penalty plus norm-matched constant reward",
    "E2E": "Positive constant reward",
    "E2E-bi": "Deterministic-policy next-state bonus (bilevel E2E-style)",
}
# These two targets exist for every method, task and training seed.
MATCHED_ANGLES = (5.0, 10.0)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError("Expected a finite, nonzero vector")
    return vector / norm


def cosine(a, b):
    return float(np.clip(unit(a) @ unit(b), -1.0, 1.0))


def angle_deg(a, b):
    return float(np.degrees(np.arccos(cosine(a, b))))


def tangent(z, baseline):
    """Unit component orthogonal to the raw-reward baseline, not z - baseline."""
    u = unit(baseline)
    return unit(unit(z) - cosine(z, baseline) * u)


def validate(data, source):
    n = len(data["task"])
    if data["z"].shape != (n, 50):
        raise ValueError(f"{source}: expected 50-dimensional latents")
    for key in ("family", "strength", "target_angle", "angle", "converged", "multi"):
        if data[key].shape != (n,):
            raise ValueError(f"{source}: malformed {key}")
    if set(data["task"]) != set(TASKS):
        raise ValueError(f"{source}: unexpected tasks")
    if not np.isfinite(data["z"]).all():
        raise ValueError(f"{source}: nonfinite latent")
    if not np.allclose(np.linalg.norm(data["z"], axis=1), np.sqrt(50), atol=2e-6,
                       rtol=0):
        raise ValueError(f"{source}: latent norm differs from sqrt(50)")


def summarize(rows, keys, metrics):
    """Descriptive metric summaries. Never average coordinates across models."""
    groups = {}
    for row in rows:
        for task in (row["task"], "all four tasks"):
            key = (task,) + tuple(row[k] for k in keys)
            groups.setdefault(key, []).append(row)
    result = []
    for (task, *values), group in groups.items():
        record = dict(zip(keys, values))
        record.update(task=task, n_comparisons=len(group),
                      n_training_seeds=len({r["training_seed"] for r in group}),
                      n_tasks=len({r["task"] for r in group}))
        for metric in metrics:
            x = np.array([r[metric] for r in group])
            for suffix, value in (("mean", x.mean()), ("min", x.min()), ("max", x.max())):
                record[f"{metric}_{suffix}"] = float(value)
        result.append(record)
    return result


def analyze(data_dir):
    baseline_rows, subset_rows, pair_rows = [], [], []
    for seed, source in SOURCES:
        with np.load(data_dir / source, allow_pickle=False) as archive:
            data = dict(archive)
        validate(data, source)
        for task in TASKS:
            task_mask = data["task"] == task
            raw = np.flatnonzero(task_mask & (data["family"] == "z_r"))
            subsets = np.flatnonzero(task_mask & (data["family"] == "C2"))
            if len(raw) != 1 or len(subsets) != 5:
                raise ValueError(f"{source}/{task}: expected one baseline and five subsets")
            z_raw = data["z"][raw[0]].astype(np.float64)
            noise = [angle_deg(data["z"][i], z_raw) for i in subsets]
            common = dict(source=source, training_seed=seed, task=task)
            for index, i in enumerate(subsets):
                subset_rows.append(dict(common, subset_index=index,
                                        angle_to_full_data_deg=noise[index],
                                        cosine_to_full_data=cosine(data["z"][i], z_raw)))
            noise_mean, noise_max = float(np.mean(noise)), float(max(noise))
            matched = {}
            for family, method in METHODS.items():
                indices = np.flatnonzero(task_mask & (data["family"] == family))
                for i in indices:
                    if not data["converged"][i] or data["multi"][i]:
                        raise ValueError(f"{source}/{task}/{method}: unreliable saved solution")
                    z = data["z"][i].astype(np.float64)
                    angle = angle_deg(z, z_raw)
                    target = float(data["target_angle"][i])
                    if not np.isfinite(target) or abs(angle - target) > 1.0:
                        raise ValueError(f"{source}/{task}/{method}: unmatched angle")
                    if abs(angle - float(data["angle"][i])) > 0.01:
                        raise ValueError(f"{source}/{task}/{method}: saved angle mismatch")
                    baseline_rows.append(dict(
                        common, method=method, source_row=int(i),
                        coefficient=float(data["strength"][i]), target_angle_deg=target,
                        angle_to_raw_deg=angle, cosine_to_raw=cosine(z, z_raw),
                        normalized_latent_distance=float(np.linalg.norm(z - z_raw)),
                        subset_angle_mean_deg=noise_mean, subset_angle_max_deg=noise_max,
                        angle_exceeds_observed_subset_max=angle > noise_max,
                    ))
                    if target in MATCHED_ANGLES:
                        key = (family, target)
                        if key in matched:
                            raise ValueError(f"{source}/{task}: duplicate matched latent")
                        matched[key] = z
                for target in MATCHED_ANGLES:
                    if (family, target) not in matched:
                        raise ValueError(f"{source}/{task}/{method}: missing {target} degrees")
            for target in MATCHED_ANGLES:
                for f, g in combinations(METHODS, 2):
                    z_f, z_g = matched[f, target], matched[g, target]
                    pair_rows.append(dict(
                        common, method_a=METHODS[f], method_b=METHODS[g],
                        target_angle_deg=target, latent_cosine=cosine(z_f, z_g),
                        latent_angle_deg=angle_deg(z_f, z_g),
                        tangent_cosine=cosine(tangent(z_f, z_raw), tangent(z_g, z_raw)),
                    ))
    return {
        "q1_fb_baseline.csv": baseline_rows,
        "q1_fb_subsets.csv": subset_rows,
        "q1_fb_pairs.csv": pair_rows,
        "q1_fb_baseline_summary.csv": summarize(
            [r for r in baseline_rows if r["target_angle_deg"] in MATCHED_ANGLES],
            ("method", "target_angle_deg"),
            ("angle_to_raw_deg", "cosine_to_raw", "subset_angle_max_deg")),
        "q1_fb_subset_summary.csv": summarize(
            subset_rows, (), ("angle_to_full_data_deg",)),
        "q1_fb_pair_summary.csv": summarize(
            pair_rows, ("method_a", "method_b", "target_angle_deg"),
            ("latent_cosine", "latent_angle_deg", "tangent_cosine")),
    }


def csv_text(rows):
    if not rows:
        raise ValueError("Refusing to write an empty table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: format(v, ".10g") if isinstance(v, float) else v
                         for k, v in row.items()})
    return stream.getvalue()


REPORT_LABELS = {
        METHODS["CQL"]: "CQL 價值偏好動作分布修正",
        METHODS["ReBRAC"]: "ReBRAC 動作偏離懲罰",
        METHODS["ReBRAC+E2E"]: "ReBRAC 動作偏離懲罰＋長度配平的正常數 reward",
        METHODS["E2E"]: "正常數 reward",
        METHODS["E2E-bi"]: "確定性 policy 下一狀態 bonus（bilevel E2E-style）",
    }


def report_tables(tables):
    """Render the filled Q1 cells, keeping future behavioral results blank."""
    labels = REPORT_LABELS
    comparison = [
        "| Reward 類型 | Bias strength：實際修正係數範圍 | Return | Latent similarity：與原始 reward 的平均 cosine／平均夾角 | 主要行為特徵 |",
        "|---|---|---|---|---|",
        "| 原始任務 reward（基準） | 0 | 待填 | 1.000000／0.000 度（自身比較） | 待填 |",
    ]
    for row in tables["q1_fb_baseline_summary.csv"]:
        if row["task"] != "all four tasks" or row["target_angle_deg"] != 10:
            continue
        coefficients = [r["coefficient"] for r in tables["q1_fb_baseline.csv"]
                        if r["method"] == row["method"] and r["target_angle_deg"] == 10]
        comparison.append(
            f"| {labels[row['method']]} | {min(coefficients):.4g}–{max(coefficients):.4g} "
            f"| 待填 | {row['cosine_to_raw_mean']:.6f}／"
            f"{row['angle_to_raw_deg_mean']:.3f} 度 | 待填 |"
        )
    pairs = [
        "| Reward 修正比較 | 偏移方向 cosine：平均［最小，最大］ | 兩個完整 latent 的平均夾角 |",
        "|---|---|---|",
    ]
    for row in tables["q1_fb_pair_summary.csv"]:
        if row["task"] != "all four tasks" or row["target_angle_deg"] != 10:
            continue
        pairs.append(
            f"| {labels[row['method_a']]} vs {labels[row['method_b']]} "
            f"| {row['tangent_cosine_mean']:.3f}［{row['tangent_cosine_min']:.3f}，"
            f"{row['tangent_cosine_max']:.3f}］ | {row['latent_angle_deg_mean']:.3f} 度 |"
        )
    noise = next(r for r in tables["q1_fb_subset_summary.csv"] if r["task"] == "all four tasks")
    subset = (
        f"五個互不重疊推論子集各自重算原始 reward latent，與完整資料估計比較，"
        f"共 60 個子集估計的平均夾角為 {noise['angle_to_full_data_deg_mean']:.3f} 度，"
        f"範圍為 {noise['angle_to_full_data_deg_min']:.3f}–"
        f"{noise['angle_to_full_data_deg_max']:.3f} 度。"
    )
    return {"comparison": "\n".join(comparison), "pairs": "\n".join(pairs), "subsets": subset}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--report-tables", action="store_true",
                        help="Print Q1 report tables and subset result without writing files")
    parser.add_argument("--check", action="store_true",
                        help="Recompute and verify existing CSVs without writing")
    args = parser.parse_args()
    outputs = analyze(args.data_dir)
    if args.report_tables:
        for content in report_tables(outputs).values():
            print(content, end="\n\n")
        return
    if not args.check:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in outputs.items():
        path = args.out_dir / filename
        content = csv_text(rows)
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Stale or missing output: {path}")
            print(f"Verified {filename}: {len(rows)} rows")
        else:
            path.write_text(content)
            print(f"Wrote {path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
