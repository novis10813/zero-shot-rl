"""Torque statistics and sample alignment of analyze.action_bound on known samples."""

import unittest

import numpy as np

from analyze.action_bound import GEAR, infeasible, sampled_rows, shortfall


class ActionBoundTest(unittest.TestCase):
    def test_infeasible_counts_entries_above_the_multiplier(self):
        a = np.array([[1.0, -1.0, 0.5, 0.0, 0.5, 0.9]])
        self.assertAlmostEqual(infeasible(a, 0.6), 3 / 6)
        self.assertEqual(infeasible(a, 1.0), 0.0)
        self.assertEqual(infeasible(a, 1.5), 0.0)

    def test_shortfall_is_the_torque_share_above_the_limit(self):
        a = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # one hip at full torque, 100 N m
        self.assertAlmostEqual(shortfall(a, 0.75), 0.25)
        self.assertEqual(shortfall(a, 1.25), 0.0)
        full = np.ones((1, 6))
        self.assertAlmostEqual(shortfall(full, 0.5), 0.5)
        self.assertEqual(GEAR.shape, (6,))

    def test_actions_beyond_the_range_count_as_the_limit(self):
        self.assertEqual(shortfall(np.full((1, 6), 3.0), 1.0), 0.0)

    def test_sampled_rows_follow_the_episode_and_the_step(self):
        bound = dict(episode=np.repeat([300000, 300001], 4), t=np.tile(np.arange(4), 2))
        adapt = dict(episode=np.array([300001, 300000]), t=np.array([2, 3]))
        np.testing.assert_array_equal(sampled_rows(bound, adapt), [6, 3])


if __name__ == "__main__":
    unittest.main()
