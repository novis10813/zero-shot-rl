"""Analytic, independent saved-data and output checks for Standard-FB geometry."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from analyze.latent import (
    FAMILIES, RESIDUAL_TOL, analyze, decomposition, heatmap_data, panel_rows,
    remove_constant, report_tables,
)
from analyze.q1 import analyze as analyze_q1
from common import INPUTS, MATCHED_ANGLES, ROOT, SOURCES, TASKS, csv_text


# Absolute paths: the inherited PYTHONPATH may be relative to the repository root.
ENVIRONMENT = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT.parents[1]), str(ROOT))))

class GeometryTests(unittest.TestCase):
    def test_antipodal_axis_is_uncentered(self):
        vectors = np.array([[1., 0., 0.], [-1., 0., 0.], [1., 0., 0.]])
        mean, s, axes, energy = decomposition(vectors)
        np.testing.assert_allclose(mean, 0)
        self.assertAlmostEqual(s[0], np.sqrt(3))
        self.assertAlmostEqual(energy[0], 1)
        np.testing.assert_allclose(vectors @ axes.T @ axes, vectors, atol=1e-14)
        orthogonal = np.eye(3)
        np.testing.assert_allclose(decomposition(orthogonal)[3], [1/3] * 3)

    def test_centered_pca_and_projected_origin(self):
        vectors = np.array([[2., 0., 0.], [4., 0., 0.], [6., 0., 0.]])
        mean, s, axes, energy = decomposition(vectors, centered=True)
        np.testing.assert_allclose(mean, [4, 0, 0])
        np.testing.assert_allclose((vectors - mean) @ axes[0], [-2, 0, 2])
        self.assertAlmostEqual(s[0], np.sqrt(8))
        self.assertAlmostEqual(energy[0], 1)
        with self.assertRaises(ValueError):
            decomposition(np.ones((3, 4)), centered=True)
        with self.assertRaises(ValueError):
            decomposition([[np.nan, 0], [1, 0]])

    def test_constant_removal_and_degeneracy(self):
        residual, norm = remove_constant([3., 4., 0.], [1., 0., 0.])
        np.testing.assert_allclose(residual, [0, 0.8, 0])
        self.assertAlmostEqual(norm ** 2, 0.64)
        for sign in (-1, 1):
            _, norm = remove_constant([sign, 0., 0.], [1., 0., 0.])
            self.assertLess(norm, RESIDUAL_TOL)

    def test_rotation_invariance(self):
        rng = np.random.default_rng(54)
        vectors = rng.normal(size=(5, 9))
        rotation, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        constant = rng.normal(size=9)
        for centered in (True, False):
            first = decomposition(vectors, centered)
            second = decomposition(vectors @ rotation, centered)
            np.testing.assert_allclose(first[1], second[1], atol=1e-13)
            np.testing.assert_allclose(first[3], second[3], atol=1e-13)
        for vector in vectors:
            residual, norm = remove_constant(vector, constant)
            rotated, rotated_norm = remove_constant(vector @ rotation, constant @ rotation)
            np.testing.assert_allclose(residual @ rotation, rotated, atol=1e-13)
            self.assertAlmostEqual(norm, rotated_norm)


class SavedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = analyze(INPUTS)

    def test_exact_coverage_and_q1_agreement(self):
        self.assertEqual(len(self.tables["latent_fb_pairs.csv"]), 240)
        self.assertEqual(len(self.tables["latent_fb_directions.csv"]), 120)
        self.assertEqual(len(self.tables["latent_fb_spectrum.csv"]), 216)
        self.assertEqual(len(self.tables["latent_fb_pca.csv"]), 480)
        self.assertEqual(len(self.tables["latent_fb_pca_variance.csv"]), 72)
        for rows in self.tables.values():
            if "source" in rows[0]:
                self.assertEqual({r["source"] for r in rows}, {name for _, name in SOURCES})
        q1 = analyze_q1(INPUTS)["q1_fb_pairs.csv"]
        self.assertEqual(len(q1), 240)
        for row, reference in zip(self.tables["latent_fb_pairs.csv"], q1):
            for metric in ("latent_cosine", "tangent_cosine"):
                self.assertAlmostEqual(row[metric], reference[metric])

    def test_independent_residual_spectrum_and_pca_recomputation(self):
        for seed, source in SOURCES:
            with np.load(INPUTS / source, allow_pickle=False) as data:
                for task in TASKS:
                    mask = data["task"] == task
                    raw = data["z"][mask & (data["family"] == "z_r")][0].astype(float)
                    raw /= np.linalg.norm(raw)
                    constant = data["u_1"].astype(float)
                    constant -= (constant @ raw) * raw
                    constant /= np.linalg.norm(constant)
                    for target in MATCHED_ANGLES:
                        z = np.array([data["z"][mask & (data["family"] == family)
                                              & (data["target_angle"] == target)][0].astype(float)
                                      for family in FAMILIES])
                        z /= np.linalg.norm(z, axis=1, keepdims=True)
                        tangent = z - (z @ raw)[:, None] * raw
                        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
                        residual = tangent - (tangent @ constant)[:, None] * constant
                        norm = np.linalg.norm(residual, axis=1)
                        valid = norm > RESIDUAL_TOL
                        self.assertEqual(list(valid), [True, True, True, False, True])
                        residual = residual[valid] / norm[valid, None]

                        def select(filename, **criteria):
                            return [r for r in self.tables[filename]
                                    if r.get("training_seed") == seed and r["task"] == task
                                    and r["target_angle_deg"] == target
                                    and all(r[k] == v for k, v in criteria.items())]

                        direction_rows = select("latent_fb_directions.csv")
                        pair_rows = select("latent_fb_pairs.csv")
                        self.assertEqual(len(direction_rows), 5)
                        self.assertEqual(len(pair_rows), 10)
                        for index, row in enumerate(direction_rows):
                            self.assertAlmostEqual(row["retained_energy"], norm[index] ** 2)
                        for row in pair_rows:
                            a, b = FAMILIES.index(row["family_a"]), FAMILIES.index(row["family_b"])
                            if valid[a] and valid[b]:
                                ra = tangent[a] - (tangent[a] @ constant) * constant
                                rb = tangent[b] - (tangent[b] @ constant) * constant
                                self.assertAlmostEqual(row["residual_cosine"], ra @ rb / norm[a] / norm[b])
                            else:
                                self.assertFalse(row["residual_defined"])
                                self.assertEqual(row["residual_cosine"], "")
                        for stage, vectors in (("latent", np.vstack([raw, z])),
                                               ("tangent", tangent),
                                               ("constant_removed", residual)):
                            if stage != "latent":
                                s = np.linalg.svd(vectors, compute_uv=False)
                                records = select("latent_fb_spectrum.csv", stage=stage)
                                np.testing.assert_allclose([r["singular_value"] for r in records], s)
                                np.testing.assert_allclose([r["energy_fraction"] for r in records],
                                                           s ** 2 / np.sum(s ** 2))
                            mean = vectors.mean(axis=0)
                            centered = vectors - mean
                            _, s, axes = np.linalg.svd(centered, full_matrices=False)
                            points = select("latent_fb_pca.csv", stage=stage, fitted=True)
                            scores = np.array([[r["pc1"], r["pc2"]] for r in points])
                            # Gram comparison ignores arbitrary signs of PCA axes.
                            expected = centered @ axes[:2].T
                            np.testing.assert_allclose(scores @ scores.T, expected @ expected.T, atol=1e-12)
                            var = select("latent_fb_pca_variance.csv", stage=stage)[0]
                            self.assertAlmostEqual(var["top2_variance"], sum(s[:2] ** 2) / sum(s ** 2))
                            if stage == "latent":
                                controls = select("latent_fb_pca.csv", stage=stage, fitted=False)
                                self.assertEqual(len(controls), 5)
                                subset = np.array([data["z"][r["source_row"]].astype(float) for r in controls])
                                subset /= np.linalg.norm(subset, axis=1, keepdims=True)
                                expected = (subset - mean) @ axes[:2].T
                                got = np.array([[r["pc1"], r["pc2"]] for r in controls])
                                np.testing.assert_allclose(got @ got.T, expected @ expected.T, atol=1e-12)

    def test_heatmap_inputs_match_10_degree_csv_slice(self):
        for seed, _ in SOURCES:
            for task in TASKS:
                rows = panel_rows(self.tables["latent_fb_pairs.csv"], seed, task)
                self.assertEqual(len(rows), 10)
                for metric in ("latent_cosine", "latent_distance", "tangent_cosine", "residual_cosine"):
                    families, matrix = heatmap_data(self.tables, seed, task, metric)
                    expected_families = tuple(f for f in FAMILIES if f != "E2E") if metric == "residual_cosine" else FAMILIES
                    self.assertEqual(families, expected_families)
                    np.testing.assert_allclose(matrix, matrix.T)
                    np.testing.assert_allclose(np.diag(matrix), 0 if metric == "latent_distance" else 1)
                    for row in rows:
                        if row["family_a"] in families and row["family_b"] in families:
                            a, b = families.index(row["family_a"]), families.index(row["family_b"])
                            self.assertEqual(matrix[a, b], row[metric])
                for filename in ("latent_fb_pca.csv", "latent_fb_pca_variance.csv", "latent_fb_spectrum.csv"):
                    records = panel_rows(self.tables[filename], seed, task)
                    self.assertTrue(records)
                    self.assertTrue(all(r["target_angle_deg"] == 10 and r["task"] == task
                                        and r["training_seed"] == seed for r in records))

    def test_summary_and_saved_outputs_and_report(self):
        for filename, rows in self.tables.items():
            self.assertEqual((ROOT / "data" / filename).read_text(), csv_text(rows))
        for summary, raw, metrics, keys in (
            ("latent_fb_axis_summary.csv", "latent_fb_spectrum.csv", ("energy_fraction",), ("stage",)),
            ("latent_fb_retention_summary.csv", "latent_fb_directions.csv", ("retained_energy",), ("family",)),
        ):
            for row in self.tables[summary]:
                selected = [r for r in self.tables[raw] if r["target_angle_deg"] == row["target_angle_deg"]
                            and (row["task"] == "all four tasks" or row["task"] == r["task"])
                            and all(row[k] == r[k] for k in keys)
                            and r.get("component", 1) == 1]
                self.assertEqual(len(selected), row["n_comparisons"])
                for metric in metrics:
                    values = [r[metric] for r in selected]
                    for suffix, function in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                        self.assertAlmostEqual(row[f"{metric}_{suffix}"], function(values))
        self.assertIn(report_tables(self.tables), (ROOT / "REPORT.md").read_text())

    def test_check_fails_without_writing_missing_or_stale_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            command = [sys.executable, "-m", "analyze.latent", "--check",
                       "--out-dir", directory, "--figure-dir", directory]
            result = subprocess.run(command, capture_output=True, text=True, env=ENVIRONMENT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Stale or missing output", result.stderr)
            self.assertEqual(list(output.iterdir()), [])
            stale = output / "latent_fb_pairs.csv"
            stale.write_text("stale\n")
            result = subprocess.run(command, capture_output=True, text=True, env=ENVIRONMENT)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(stale.read_text(), "stale\n")


if __name__ == "__main__":
    unittest.main()
