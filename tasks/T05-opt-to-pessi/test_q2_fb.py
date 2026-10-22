"""Independent saved-rollout, pairing, aggregation and report checks for Q2."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_q1_fb import ROOT, SOURCES, TASKS, METHODS, analyze as analyze_q1, report_tables as q1_report_tables
from analyze_q2_fb import EPISODES, METRICS, analyze, load_rollouts, report_tables


class Q2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = analyze(ROOT / "data")
        cls.raw = {}
        for seed, source in SOURCES:
            source = source.replace("zs_fb_", "rollouts_fb_").replace(".npz", ".csv")
            with (ROOT / "data" / source).open() as stream:
                cls.raw[source] = list(csv.DictReader(stream))

    def test_standard_fb_only_exact_coverage(self):
        pairs = self.tables["q2_fb_pairs.csv"]
        self.assertEqual(len(pairs), 2400)
        self.assertEqual(len(self.tables["q2_fb_seed_summary.csv"]), 120)
        self.assertEqual(len(self.tables["q2_fb_task_summary.csv"]), 40)
        self.assertEqual(len(self.tables["q2_fb_baseline.csv"]), 12)
        self.assertEqual(len(self.tables["q2_fb_baseline_summary.csv"]), 4)
        expected = {(seed, task, method, angle, episode)
                    for seed, _ in SOURCES for task in TASKS for method in METHODS.values()
                    for angle in (5, 10) for episode in range(EPISODES)}
        observed = {(r["training_seed"], r["task"], r["method"], r["target_angle_deg"], r["episode"])
                    for r in pairs}
        self.assertEqual(observed, expected)
        self.assertTrue(all(r["source"].startswith("rollouts_fb_") for r in pairs))

    def test_every_difference_matches_independent_episode_join(self):
        for row in self.tables["q2_fb_pairs.csv"]:
            records = self.raw[row["source"]]
            actual = next(r for r in records if int(r["row"]) == row["source_row"]
                          and int(r["episode"]) == row["episode"])
            base = next(r for r in records if r["family"] == "z_r" and r["task"] == row["task"]
                        and int(r["episode"]) == row["episode"])
            self.assertEqual(int(base["row"]), row["baseline_row"])
            for metric in METRICS:
                self.assertEqual(row[metric], float(actual[metric]))
                self.assertEqual(row[f"baseline_{metric}"], float(base[metric]))
                self.assertAlmostEqual(row[f"delta_{metric}"], float(actual[metric]) - float(base[metric]))

    def test_seed_means_std_and_sign_counts(self):
        for row in self.tables["q2_fb_seed_summary.csv"]:
            pairs = [r for r in self.tables["q2_fb_pairs.csv"]
                     if (r["source"], r["source_row"]) == (row["source"], row["source_row"])]
            self.assertEqual(len(pairs), EPISODES)
            for metric in METRICS:
                values = [r[f"delta_{metric}"] for r in pairs]
                self.assertAlmostEqual(row[f"delta_{metric}_mean"], np.mean(values))
                self.assertAlmostEqual(row[f"delta_{metric}_std"], np.std(values, ddof=1))
                self.assertAlmostEqual(row[f"delta_{metric}_se"], np.std(values, ddof=1) / np.sqrt(20))
                self.assertEqual(row[f"delta_{metric}_positive_episodes"], sum(v > 0 for v in values))
                for prefix in ("", "baseline_"):
                    self.assertAlmostEqual(row[f"{prefix}{metric}_mean"],
                                           np.mean([r[f"{prefix}{metric}"] for r in pairs]))

    def test_task_summary_direct_from_csv_not_seed_summary(self):
        for row in self.tables["q2_fb_task_summary.csv"]:
            self.assertEqual(row["n_training_seeds"], 3)
            self.assertEqual(row["n_episodes_per_seed"], EPISODES)
            for metric in METRICS:
                direct = {prefix: [] for prefix in ("", "baseline_", "delta_")}
                for records in self.raw.values():
                    selected = [r for r in records if r["task"] == row["task"]
                                and METHODS.get(r["family"]) == row["method"]
                                and float(r["target_angle"]) == row["target_angle_deg"]]
                    baseline = {int(r["episode"]): float(r[metric]) for r in records
                                if r["task"] == row["task"] and r["family"] == "z_r"}
                    self.assertEqual(len(selected), EPISODES)
                    modified = [float(r[metric]) for r in selected]
                    reference = [baseline[int(r["episode"])] for r in selected]
                    direct[""].append(np.mean(modified))
                    direct["baseline_"].append(np.mean(reference))
                    direct["delta_"].append(np.mean(np.subtract(modified, reference)))
                for prefix, values in direct.items():
                    for suffix, function in (("mean", np.mean), ("min", min), ("max", max)):
                        self.assertAlmostEqual(row[f"{prefix}{metric}_mean_{suffix}"], function(values))

    def test_baseline_summary_direct_from_csv(self):
        for row in self.tables["q2_fb_baseline_summary.csv"]:
            for metric in METRICS:
                values = [np.mean([float(r[metric]) for r in records
                                   if r["task"] == row["task"] and r["family"] == "z_r"])
                          for records in self.raw.values()]
                for suffix, function in (("mean", np.mean), ("min", min), ("max", max)):
                    self.assertAlmostEqual(row[f"{metric}_mean_{suffix}"], function(values))

    def test_input_validation_rejects_corruption(self):
        source = SOURCES[0][1]
        with np.load(ROOT / "data" / source, allow_pickle=False) as archive:
            data = dict(archive)
        records = next(iter(self.raw.values()))
        bad_versions = [records[:-1], records + records[:1]]
        for field, bad in (("return", "nan"), ("task", "wrong"), ("strength", "999"),
                           ("episode", "20"), ("row", "-1"), ("sat_frac", "1.2"),
                           ("family", "wrong"), ("target_angle", "999"), ("angle", "999"),
                           ("speed", "inf"), ("upright", "1.1"), ("knn_dist", "-1")):
            version = [dict(r) for r in records]
            version[0][field] = bad
            bad_versions.append(version)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.csv"
            for records in bad_versions:
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
                with self.assertRaises(ValueError):
                    load_rollouts(path, data)

    def test_rendering_joins_methods_by_label_not_order(self):
        q1 = analyze_q1(ROOT / "data")
        expected = report_tables(self.tables, q1)
        def reordered(tables):
            result = q1_report_tables(tables)
            lines = result["comparison"].splitlines()
            result["comparison"] = "\n".join(lines[:2] + list(reversed(lines[2:])))
            return result
        with patch("analyze_q2_fb.q1_report_tables", side_effect=reordered):
            actual = report_tables(self.tables, q1)
        self.assertEqual(set(expected["comparison"].splitlines()),
                         set(actual["comparison"].splitlines()))
        self.assertEqual(expected["behavior"], actual["behavior"])

    def test_rendered_row_independently_from_raw_rollouts(self):
        # Fixed concrete example, independent of the table generator's aggregation.
        changes = {metric: [] for metric in METRICS}
        for records in self.raw.values():
            baseline = {int(r["episode"]): r for r in records
                        if r["task"] == "run" and r["family"] == "z_r"}
            selected = [r for r in records if r["task"] == "run" and r["family"] == "CQL"
                        and float(r["target_angle"]) == 10]
            for metric in METRICS:
                changes[metric].append(np.mean([float(r[metric]) - float(baseline[int(r["episode"])][metric])
                                                for r in selected]))
        self.assertIn(
            "| 跑 | CQL 價值偏好動作分布修正 | "
            f"{np.mean(changes['return']):+.1f}［{min(changes['return']):+.1f}，{max(changes['return']):+.1f}］"
            f" | {np.mean(changes['speed']):+.3f} | {np.mean(changes['upright']):+.3f}"
            f" | {100 * np.mean(changes['fall_frac']):+.2f}"
            f" | {100 * np.mean(changes['sat_frac']):+.2f}"
            f" | {np.mean(changes['knn_dist']):+.3f} |",
            report_tables(self.tables, analyze_q1(ROOT / "data"))["behavior"],
        )

    def test_report_tables_match_recomputed_data(self):
        report = (ROOT / "REPORT.md").read_text()
        tables = report_tables(self.tables, analyze_q1(ROOT / "data"))
        for content in tables.values():
            self.assertIn(content, report)
        for row in tables["comparison"].splitlines()[2:]:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            self.assertEqual(len(cells), 5)
            self.assertNotEqual(cells[2], "待填")
            self.assertNotEqual(cells[4], "待填")
        self.assertNotIn("成功／失敗率", tables["comparison"])


if __name__ == "__main__":
    unittest.main()
