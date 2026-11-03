"""Budgeted latent selection simulated from logged returns (analyze.selection)."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze import selection


def write_rollouts(path, means):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "target_angle", "episode", "return"])
        writer.writeheader()
        for i, mean in enumerate(means):
            family, angle = ("z_r", "0") if i == 0 else ("CQL", str(i))
            for episode in range(100):
                writer.writerow({"family": family, "target_angle": angle,
                                 "episode": episode, "return": mean + episode % 2})


class SelectionTest(unittest.TestCase):
    def test_pool_and_evaluation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.csv"
            write_rollouts(path, [100, 500])
            candidates = selection.load_candidates(path)
        self.assertEqual(list(candidates), ["z_r 0", "CQL 1"])
        for pool, evaluation in candidates.values():
            self.assertEqual((len(pool), len(evaluation)), (50, 50))

    def test_methods_find_a_clear_best_once_every_candidate_is_played(self):
        pools = [np.full(50, 100.0), np.full(50, 900.0), np.full(50, 300.0)]
        rng = np.random.default_rng(0)
        for method in ("uniform", "ucb"):
            picks = {selection.select(method, pools, 6, rng) for _ in range(20)}
            self.assertEqual(picks, {1})

    def test_one_episode_selects_the_played_candidate_uniformly(self):
        pools = [np.full(50, 100.0), np.full(50, 900.0)]
        rng = np.random.default_rng(0)
        picks = [selection.select("ucb", pools, 1, rng) for _ in range(400)]
        self.assertAlmostEqual(np.mean(picks), 0.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
