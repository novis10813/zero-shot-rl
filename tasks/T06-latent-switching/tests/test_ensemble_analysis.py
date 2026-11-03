"""Forward-ensemble analysis on a synthetic probe."""

import unittest

import numpy as np

from analyze.ensemble import ensemble_rows
from tests.test_stage0 import synthetic_probe


class EnsembleAnalysisTests(unittest.TestCase):
    def test_rows(self):
        probe = synthetic_probe()
        rng = np.random.default_rng(1)
        n = len(probe["G"])
        spread = rng.uniform(0, 2, n)
        probe["q_ens"] = probe["G"][:, None] + spread[:, None] * np.array([1.0, -1.0, 1.0, -1.0])
        probe["c_ens"] = np.full((n, 4), 50.0)
        rows = ensemble_rows(0, probe)
        self.assertEqual(len(rows), 2 * 6)  # 2 tasks, 5 sources and "all"
        pooled = [r for r in rows if r["source"] == "all"]
        self.assertEqual(sum(r["n"] for r in pooled), n)
        for row in rows:
            self.assertAlmostEqual(row["c_ens"], 50.0)
            self.assertAlmostEqual(row["ens_over"], 0.0, places=6)
            self.assertAlmostEqual(row["rho_q_ens_G"], 1.0)
        # members at G +- spread: the std is the spread, exactly
        stand = probe["task"] == "stand"
        self.assertAlmostEqual(pooled[0]["std_ens"], spread[stand].mean())


if __name__ == "__main__":
    unittest.main()
