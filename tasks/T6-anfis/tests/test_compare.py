"""Verify T6-anfis analysis reproduces stored tables byte for byte."""

import unittest

import numpy as np

from analyze.compare import DATA_DIR, INPUTS, build_probes_csv, build_scores_csv


class CompareTest(unittest.TestCase):
    def test_scores_csv_reproduced(self):
        expected = (DATA_DIR / "scores.csv").read_text()
        actual = build_scores_csv(INPUTS / "baseline_table6.csv", INPUTS / "table6.csv")
        self.assertEqual(actual, expected)

    def test_probes_csv_reproduced(self):
        expected = (DATA_DIR / "reward_probes.csv").read_text()
        actual = build_probes_csv(np.load(INPUTS / "reward_grid.npz"))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
