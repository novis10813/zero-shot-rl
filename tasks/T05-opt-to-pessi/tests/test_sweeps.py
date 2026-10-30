"""Independent tests for Standard-FB saved coefficient and candidate-axis sweeps."""

import copy
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import spearmanr

from analyze.q2 import EPISODES, METRICS
from analyze.sweeps import (
    AX_ANGLES, MISSING_FILE, NUMERIC_TOL, TARGETS, analyze, candidate_axis,
    load_missing, missing_keys, parse_missing_log, rank_correlation,
    report_tables, trend,
)
from common import INPUTS, METHODS, ROOT, SOURCES, TASKS, csv_text


# Absolute paths: the inherited PYTHONPATH may be relative to the repository root.
ENVIRONMENT = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT.parents[1]), str(ROOT))))

class NumericalTests(unittest.TestCase):
    def test_trends_and_average_rank_ties(self):
        self.assertEqual(trend([0, 1, 2])["trend"], "nondecreasing")
        self.assertEqual(trend([2, 1, 0])["trend"], "nonincreasing")
        self.assertEqual(trend([2, 2, 2])["trend"], "constant")
        result = trend([0, 1, 1, -1, 2])
        self.assertEqual((result["n_increases"], result["n_decreases"],
                          result["n_numerical_ties"], result["sign_changes"]), (2, 1, 1, 2))
        self.assertEqual(result["total_variation"], 6)
        self.assertEqual(result["max_abs_adjacent_change"], 3)
        self.assertEqual(trend([0, NUMERIC_TOL / 2])["trend"], "constant")
        for a, b in (([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [3, 2, 1]),
                     ([1, 2, 3, 4, 5], [1, 1, 4, 3, 3])):
            self.assertAlmostEqual(rank_correlation(a, b), spearmanr(a, b).statistic)
        self.assertEqual(rank_correlation([1, 2, 3], [1, 1, 1]), "")
        for bad in ([1], [0, np.nan], [0, np.inf]):
            with self.assertRaises(ValueError):
                trend(bad)

    def test_sanitized_log_extraction_and_validation(self):
        all_keys = {(t, f, a) for t in TASKS for f in METHODS for a in TARGETS}
        missing = {("stand", "E2E", 70.), ("walk", "E2E-bi", 20.),
                   ("run", "ReBRAC+E2E", 50.)}
        saved = sorted(all_keys - missing)
        data = {"task": np.array([k[0] for k in saved]),
                "family": np.array([k[1] for k in saved]),
                "target_angle": np.array([k[2] for k in saved])}
        text = ("unrelated private log line\n"
                "stand E2E target 70.0 unreachable (max angle 40.1)\n"
                "walk E2E-bi target 20.0 no fixed point\n"
                "run ReBRAC+E2E target 50.0 discontinuous\n")
        rows = parse_missing_log(text, 0, SOURCES[0][1], data)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["status"] for r in rows],
                         ["unbracketed_grid", "bisection_not_converged", "target_mismatch"])
        self.assertEqual([r["source_log_line"] for r in rows], [2, 3, 4])
        self.assertNotIn("private", csv_text(rows))
        self.assertEqual(missing_keys(data), missing)
        for bad in (text + text, "", text.replace("stand E2E target", "stand CQL target")):
            with self.assertRaises(ValueError):
                parse_missing_log(bad, 0, SOURCES[0][1], data)


class SavedSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = analyze(INPUTS)
        cls.data, cls.rollouts = {}, {}
        for seed, source in SOURCES:
            with np.load(INPUTS / source, allow_pickle=False) as archive:
                cls.data[seed] = dict(archive)
            rollout_source = source.replace("zs_fb_", "rollouts_fb_").replace(".npz", ".csv")
            with (INPUTS / rollout_source).open(newline="") as stream:
                cls.rollouts[seed] = {(int(r["row"]), int(r["episode"])): r
                                      for r in csv.DictReader(stream)}

    def test_exact_standard_fb_coverage(self):
        counts = {"sweep_fb_coverage.csv": 360, "sweep_fb_points.csv": 339,
                  "sweep_fb_adjacent.csv": 279, "sweep_fb_curves.csv": 60,
                  "sweep_fb_method_summary.csv": 5, "axis_fb_pairs.csv": 2160,
                  "axis_fb_seed_summary.csv": 108, "axis_fb_task_summary.csv": 36,
                  "axis_fb_adjacent.csv": 672, "axis_fb_trends.csv": 84,
                  "axis_fb_trend_summary.csv": 7}
        for filename, count in counts.items():
            self.assertEqual(len(self.tables[filename]), count, filename)
        expected = {(source, task, family, target) for _, source in SOURCES
                    for task in TASKS for family in METHODS for target in TARGETS}
        rows = self.tables["sweep_fb_coverage.csv"]
        self.assertEqual({(r["source"], r["task"], r["family"], r["target_angle_deg"]) for r in rows}, expected)
        self.assertEqual(sum(r["status"] == "saved" for r in rows), 279)
        missing = load_missing(INPUTS)
        self.assertEqual(len(missing), 81)
        self.assertEqual(sum(r["status"] == "unbracketed_grid" for r in missing.values()), 79)
        self.assertEqual(sum(r["status"] == "bisection_not_converged" for r in missing.values()), 2)
        self.assertTrue(all(r["source"].startswith("zs_fb_") for r in rows))
        self.assertTrue(all(r["source"].startswith("rollouts_fb_") for r in self.tables["axis_fb_pairs.csv"]))

    def test_coefficient_points_and_adjacent_geometry_independently(self):
        for row in self.tables["sweep_fb_points.csv"]:
            data = self.data[row["training_seed"]]
            vector = data["z"][row["source_row"]].astype(float)
            raw = data["z"][(data["task"] == row["task"]) & (data["family"] == "z_r")][0].astype(float)
            expected_angle = np.degrees(np.arccos(np.clip(vector @ raw / np.linalg.norm(vector) / np.linalg.norm(raw), -1, 1)))
            self.assertAlmostEqual(row["angle_to_raw_deg"], expected_angle, places=5)
            self.assertEqual(row["coefficient"], float(data["strength"][row["source_row"]]))
        for row in self.tables["sweep_fb_adjacent.csv"]:
            data = self.data[row["training_seed"]]
            a, b = [data["z"][row[k]].astype(float) for k in ("from_row", "to_row")]
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            expected = np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
            self.assertAlmostEqual(row["latent_step_angle_deg"], expected)
            self.assertGreater(row["coefficient_gap"], 0)
            if row["from_coefficient"] == 0:
                self.assertEqual(row["tangent_turn_deg"], "")
            else:
                raw = data["z"][(data["task"] == row["task"]) & (data["family"] == "z_r")][0].astype(float)
                raw /= np.linalg.norm(raw)
                ta, tb = a - (a @ raw) * raw, b - (b @ raw) * raw
                c = np.clip(ta @ tb / np.linalg.norm(ta) / np.linalg.norm(tb), -1, 1)
                self.assertAlmostEqual(row["tangent_cosine"], c)
                self.assertAlmostEqual(row["tangent_turn_deg"], np.degrees(np.arccos(c)), places=6)

    def test_candidate_axis_is_rotation_not_endpoint_interpolation(self):
        for seed, data in self.data.items():
            for task in TASKS:
                raw_index, selected = candidate_axis(data, task)
                self.assertEqual(set(selected), set(AX_ANGLES))
                raw = data["z"][raw_index].astype(float)
                raw /= np.linalg.norm(raw)
                vectors = []
                for family in ("CQL", "ReBRAC", "E2E-bi"):
                    z = data["z"][(data["task"] == task) & (data["family"] == family)
                                  & (data["target_angle"] == 10)][0].astype(float)
                    z -= (z @ raw) * raw
                    vectors.append(z / np.linalg.norm(z))
                axis = vectors[2] - (vectors[0] + vectors[1]) / 2
                axis -= (axis @ raw) * raw
                axis /= np.linalg.norm(axis)
                for angle, index in selected.items():
                    z = data["z"][index].astype(float)
                    z /= np.linalg.norm(z)
                    expected = np.cos(np.deg2rad(angle)) * raw + np.sin(np.deg2rad(angle)) * axis
                    np.testing.assert_allclose(z, expected, atol=2e-6, rtol=0)
        corrupt = copy.deepcopy(self.data[0])
        index = np.flatnonzero((corrupt["task"] == "stand") & (corrupt["family"] == "AX"))[0]
        corrupt["z"][index] *= -1
        with self.assertRaises(ValueError):
            candidate_axis(corrupt, "stand")
        corrupt = copy.deepcopy(self.data[0])
        corrupt["target_angle"][index] = 999
        with self.assertRaises(ValueError):
            candidate_axis(corrupt, "stand")

    def test_all_axis_episode_pairs_and_seed_statistics_from_raw_csv(self):
        for row in self.tables["axis_fb_pairs.csv"]:
            raw = self.rollouts[row["training_seed"]]
            actual, baseline = raw[row["source_row"], row["episode"]], raw[row["baseline_row"], row["episode"]]
            self.assertEqual(actual["family"], "z_r" if row["signed_angle_deg"] == 0 else "AX")
            self.assertEqual(baseline["family"], "z_r")
            for metric in METRICS:
                self.assertEqual(row[metric], float(actual[metric]))
                self.assertEqual(row[f"delta_{metric}"], float(actual[metric]) - float(baseline[metric]))
        for row in self.tables["axis_fb_seed_summary.csv"]:
            raw = self.rollouts[row["training_seed"]]
            for metric in METRICS:
                values = np.array([float(raw[row["source_row"], e][metric]) for e in range(EPISODES)])
                baseline = np.array([float(raw[row["baseline_row"], e][metric]) for e in range(EPISODES)])
                self.assertAlmostEqual(row[f"{metric}_mean"], values.mean())
                self.assertAlmostEqual(row[f"{metric}_se"], values.std(ddof=1) / np.sqrt(EPISODES))
                self.assertAlmostEqual(row[f"delta_{metric}_mean"], (values - baseline).mean())
                self.assertAlmostEqual(row[f"delta_{metric}_se"], (values - baseline).std(ddof=1) / np.sqrt(EPISODES))

    def test_axis_adjacent_pairing_and_monotonicity_from_raw_csv(self):
        for row in self.tables["axis_fb_adjacent.csv"]:
            raw = self.rollouts[row["training_seed"]]
            values = np.array([float(raw[row["to_row"], e][row["metric"]])
                               - float(raw[row["from_row"], e][row["metric"]]) for e in range(EPISODES)])
            self.assertAlmostEqual(row["mean_paired_change"], values.mean())
            self.assertAlmostEqual(row["paired_change_se"], values.std(ddof=1) / np.sqrt(EPISODES))
            self.assertEqual(row["n_positive_episodes"], int(np.sum(values > 0)))
            self.assertEqual(row["n_negative_episodes"], int(np.sum(values < 0)))
        for row in self.tables["axis_fb_trends.csv"]:
            raw = self.rollouts[row["training_seed"]]
            means = []
            for angle in AX_ANGLES:
                records = [r for r in raw.values() if r["task"] == row["task"] and
                           ((angle == 0 and r["family"] == "z_r") or
                            (r["family"] == "AX" and float(r["target_angle"]) == angle))]
                self.assertEqual(len(records), EPISODES)
                means.append(np.mean([float(r[row["metric"]]) for r in records]))
            differences = np.diff(means)
            self.assertEqual(row["n_increases"], int(np.sum(differences > NUMERIC_TOL)))
            self.assertEqual(row["n_decreases"], int(np.sum(differences < -NUMERIC_TOL)))
            self.assertAlmostEqual(row["spearman_rho"], spearmanr(AX_ANGLES, means).statistic)
            self.assertAlmostEqual(row["total_variation"], np.abs(differences).sum())

    def test_summaries_match_details(self):
        for row in self.tables["sweep_fb_method_summary.csv"]:
            curves = [r for r in self.tables["sweep_fb_curves.csv"] if r["family"] == row["family"]]
            steps = [r["tangent_turn_deg"] for r in self.tables["sweep_fb_adjacent.csv"]
                     if r["family"] == row["family"] and r["tangent_turn_deg"] != ""]
            self.assertEqual(row["n_saved"], sum(r["n_saved"] for r in curves))
            self.assertEqual(row["n_adjacent_tangent_steps"], len(steps))
            self.assertAlmostEqual(row["tangent_turn_deg_mean"], np.mean(steps))
        for row in self.tables["axis_fb_task_summary.csv"]:
            group = [r for r in self.tables["axis_fb_seed_summary.csv"]
                     if r["task"] == row["task"] and r["signed_angle_deg"] == row["signed_angle_deg"]]
            self.assertEqual(len(group), 3)
            for metric in METRICS:
                for prefix in ("", "delta_"):
                    values = [r[f"{prefix}{metric}_mean"] for r in group]
                    for suffix, func in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                        self.assertAlmostEqual(row[f"{prefix}{metric}_mean_{suffix}"], func(values))
        for row in self.tables["axis_fb_trend_summary.csv"]:
            curves = [r for r in self.tables["axis_fb_trends.csv"] if r["metric"] == row["metric"]]
            self.assertEqual(len(curves), 12)
            for label in ("mixed", "nondecreasing", "nonincreasing", "constant"):
                self.assertEqual(row[f"n_{label}"], sum(r["trend"] == label for r in curves))

    def test_missing_status_rejects_corrupt_or_extra_rows(self):
        with (INPUTS / MISSING_FILE).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        versions = [rows + rows[:1]]
        for key, value in (("source", "one-step.npz"), ("training_seed", "9"),
                           ("status", "unknown"), ("converged_grid_max_angle_deg", "nan")):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            versions.append(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for version in versions:
                (path / MISSING_FILE).write_text(csv_text(version))
                with self.assertRaises(ValueError):
                    load_missing(path)
        missing = load_missing(INPUTS)
        missing.pop(next(iter(missing)))
        with patch("analyze.sweeps.load_missing", return_value=missing):
            with self.assertRaises(ValueError):
                analyze(INPUTS)

    def test_saved_tables_and_report_are_reproducible(self):
        for name, rows in self.tables.items():
            self.assertEqual((ROOT / "data" / name).read_text(), csv_text(rows))
        report = (ROOT / "REPORT.md").read_text()
        for text in report_tables(self.tables).values():
            self.assertIn(text, report)

    def test_check_refuses_missing_or_stale_outputs_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-m", "analyze.sweeps", "--check",
                       "--out-dir", directory, "--figure-dir", directory]
            result = subprocess.run(command, capture_output=True, text=True, env=ENVIRONMENT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Stale or missing output", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])
            stale = Path(directory) / "sweep_fb_coverage.csv"
            stale.write_text("stale\n")
            result = subprocess.run(command, capture_output=True, text=True, env=ENVIRONMENT)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(stale.read_text(), "stale\n")


if __name__ == "__main__":
    unittest.main()
