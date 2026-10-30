"""Numerical and saved-data checks for the Q1 standard-FB analysis."""

import unittest

import numpy as np

from analyze.q1 import analyze, report_tables
from common import INPUTS, MATCHED_ANGLES, METHODS, ROOT, SOURCES, TASKS, validate
from geometry import angle_deg, cosine, tangent, unit


class GeometryTests(unittest.TestCase):
    def test_known_angles_and_tangents(self):
        baseline = np.array([1.0, 0.0, 0.0])
        theta = np.deg2rad(10)
        first = np.array([np.cos(theta), np.sin(theta), 0])
        same = 3 * first
        opposite = np.array([np.cos(theta), -np.sin(theta), 0])
        orthogonal = np.array([np.cos(theta), 0, np.sin(theta)])
        self.assertAlmostEqual(angle_deg(first, baseline), 10)
        self.assertAlmostEqual(angle_deg(first, opposite), 20)
        self.assertAlmostEqual(cosine(tangent(first, baseline), tangent(same, baseline)), 1)
        self.assertAlmostEqual(cosine(tangent(first, baseline), tangent(opposite, baseline)), -1)
        self.assertAlmostEqual(cosine(tangent(first, baseline), tangent(orthogonal, baseline)), 0)

    def test_rotation_invariance(self):
        rng = np.random.default_rng(3)
        rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        a, b, baseline = rng.normal(size=(3, 5))
        self.assertAlmostEqual(angle_deg(a, b), angle_deg(rotation @ a, rotation @ b))
        self.assertAlmostEqual(
            cosine(tangent(a, baseline), tangent(b, baseline)),
            cosine(tangent(rotation @ a, rotation @ baseline),
                   tangent(rotation @ b, rotation @ baseline)),
        )

    def test_degenerate_vectors_rejected(self):
        for vector in ([0, 0], [np.nan, 1], [np.inf, 1]):
            with self.assertRaises(ValueError):
                unit(vector)
        with self.assertRaises(ValueError):
            tangent([1, 0], [1, 0])

    def test_invalid_saved_norm_rejected(self):
        with np.load(INPUTS / SOURCES[0][1], allow_pickle=False) as archive:
            data = dict(archive)
        data["z"] = data["z"] * 2
        with self.assertRaises(ValueError):
            validate(data, "synthetic bad norm")


class SavedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = analyze(INPUTS)

    def test_standard_fb_only_and_exact_coverage(self):
        baseline = self.tables["q1_fb_baseline.csv"]
        subsets = self.tables["q1_fb_subsets.csv"]
        pairs = self.tables["q1_fb_pairs.csv"]
        self.assertEqual(len(baseline), 279)
        self.assertEqual(len(subsets), 3 * 4 * 5)
        self.assertEqual(len(pairs), 3 * 4 * 2 * 10)
        self.assertEqual({r["source"] for r in baseline}, {name for _, name in SOURCES})
        self.assertEqual({r["method"] for r in baseline}, set(METHODS.values()))
        for seed, _ in SOURCES:
            for task in TASKS:
                for target in MATCHED_ANGLES:
                    selected = [r for r in baseline if r["training_seed"] == seed
                                and r["task"] == task and r["target_angle_deg"] == target]
                    self.assertEqual(len(selected), 5)

    def test_independent_subset_and_pair_recomputation(self):
        for seed, source in SOURCES:
            with np.load(INPUTS / source, allow_pickle=False) as data:
                for task in TASKS:
                    mask = data["task"] == task
                    base = data["z"][mask & (data["family"] == "z_r")][0].astype(float)
                    noise = data["z"][mask & (data["family"] == "C2")].astype(float)
                    expected = np.rad2deg(np.arccos(np.clip(
                        (noise @ base) / (np.linalg.norm(noise, axis=1) * np.linalg.norm(base)),
                        -1, 1)))
                    got = [r["angle_to_full_data_deg"] for r in self.tables["q1_fb_subsets.csv"]
                           if r["training_seed"] == seed and r["task"] == task]
                    np.testing.assert_allclose(got, expected, atol=1e-10)
        reverse_methods = {name: family for family, name in METHODS.items()}
        for row in self.tables["q1_fb_pairs.csv"]:
            with np.load(INPUTS / row["source"], allow_pickle=False) as data:
                mask = data["task"] == row["task"]
                baseline = data["z"][mask & (data["family"] == "z_r")][0].astype(float)
                vectors = []
                for name in (row["method_a"], row["method_b"]):
                    selection = (mask & (data["family"] == reverse_methods[name])
                                 & (data["target_angle"] == row["target_angle_deg"]))
                    vector = data["z"][selection][0].astype(float)
                    vectors.append(vector / np.linalg.norm(vector))
                first, second = vectors
                base_unit = baseline / np.linalg.norm(baseline)
                first_perp = first - (first @ base_unit) * base_unit
                second_perp = second - (second @ base_unit) * base_unit
                expected_cosine = ((first_perp @ second_perp)
                                   / (np.linalg.norm(first_perp) * np.linalg.norm(second_perp)))
                self.assertAlmostEqual(row["tangent_cosine"], expected_cosine)
                self.assertAlmostEqual(row["latent_cosine"], first @ second)
            self.assertTrue(-1 <= row["tangent_cosine"] <= 1)
            self.assertGreater(row["latent_angle_deg"], 0)
            self.assertLessEqual(row["latent_angle_deg"], 2 * row["target_angle_deg"] + 0.1)
        for row in self.tables["q1_fb_baseline.csv"]:
            distance = 2 * np.sqrt(50) * np.sin(np.deg2rad(row["angle_to_raw_deg"]) / 2)
            self.assertAlmostEqual(row["normalized_latent_distance"], distance, places=5)
            self.assertTrue(row["angle_exceeds_observed_subset_max"])

    def test_baseline_and_subset_summary_statistics(self):
        baselines, subsets = [], []
        for seed, source in SOURCES:
            with np.load(INPUTS / source, allow_pickle=False) as data:
                for task in TASKS:
                    mask = data["task"] == task
                    base = data["z"][mask & (data["family"] == "z_r")][0].astype(float)
                    base /= np.linalg.norm(base)
                    def direct_angle(vector):
                        vector = vector.astype(float)
                        return float(np.degrees(np.arccos(np.clip(
                            vector @ base / np.linalg.norm(vector), -1, 1))))
                    for vector in data["z"][mask & (data["family"] == "C2")]:
                        subsets.append((task, direct_angle(vector)))
                    for family, method in METHODS.items():
                        for target in MATCHED_ANGLES:
                            selected = (mask & (data["family"] == family)
                                        & (data["target_angle"] == target))
                            baselines.append((task, method, target,
                                              direct_angle(data["z"][selected][0])))
        for row in self.tables["q1_fb_baseline_summary.csv"]:
            values = [v for task, method, target, v in baselines
                      if method == row["method"] and target == row["target_angle_deg"]
                      and (row["task"] == "all four tasks" or task == row["task"])]
            self.assertEqual(row["n_comparisons"], len(values))
            for suffix, function in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                self.assertAlmostEqual(row[f"angle_to_raw_deg_{suffix}"], function(values))
        for row in self.tables["q1_fb_subset_summary.csv"]:
            values = [v for task, v in subsets
                      if row["task"] == "all four tasks" or task == row["task"]]
            self.assertEqual(row["n_comparisons"], len(values))
            for suffix, function in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                self.assertAlmostEqual(row[f"angle_to_full_data_deg_{suffix}"], function(values))

    def test_report_tables_match_recomputed_data(self):
        report = (ROOT / "REPORT.md").read_text()
        tables = report_tables(self.tables)
        for key in ("pairs", "subsets"):
            self.assertIn(tables[key], report)
        main_table = tables["comparison"].splitlines()
        self.assertEqual(len(main_table), 8)
        saved_rows = [line for line in report.splitlines() if line.startswith("|")][:8]
        self.assertEqual(len(saved_rows), 8)
        for expected, actual in zip(main_table[2:], saved_rows[2:]):
            self.assertEqual(actual.count("|"), 6)
            expected_cells = [cell.strip() for cell in expected.strip("|").split("|")]
            actual_cells = [cell.strip() for cell in actual.strip("|").split("|")]
            # Q2 fills the remaining columns without changing Q1 results.
            for index in (0, 1, 3):
                self.assertEqual(expected_cells[index], actual_cells[index])

    def test_summaries_match_raw_comparisons(self):
        raw = self.tables["q1_fb_pairs.csv"]
        for row in self.tables["q1_fb_pair_summary.csv"]:
            selected = [r for r in raw if r["method_a"] == row["method_a"]
                        and r["method_b"] == row["method_b"]
                        and r["target_angle_deg"] == row["target_angle_deg"]
                        and (row["task"] == "all four tasks" or r["task"] == row["task"])]
            self.assertEqual(row["n_training_seeds"], 3)
            self.assertEqual(row["n_comparisons"], len(selected))
            for metric in ("tangent_cosine", "latent_cosine", "latent_angle_deg"):
                values = [r[metric] for r in selected]
                for suffix, function in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                    self.assertAlmostEqual(row[f"{metric}_{suffix}"], function(values))


if __name__ == "__main__":
    unittest.main()
