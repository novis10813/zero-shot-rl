"""Helpers of analyze.single_step on known samples."""

import unittest

import numpy as np

from analyze.classifier import features
from analyze.single_step import rates, top_share


class SingleStepTest(unittest.TestCase):
    def test_top_share(self):
        gap = np.r_[np.zeros(9), 3.0]
        self.assertEqual(top_share(gap), 1.0)
        self.assertAlmostEqual(top_share(np.ones(20)), 0.1)

    def test_rates(self):
        acc, tpr, fpr = rates(np.array([-1.0, -1.0, 1.0, -1.0]), np.array([1.0, 1.0, -1.0, 1.0]))
        self.assertEqual((tpr, fpr), (0.75, 0.25))
        self.assertEqual(acc, 0.75)

    def test_features_difference(self):
        s, a, s2 = np.ones((2, 3)), np.zeros((2, 1), np.float32), np.full((2, 3), 1.5)
        x = features(s, a, s2)
        self.assertEqual(x.shape, (2, 7))
        np.testing.assert_array_equal(x[:, -3:], 0.5)


if __name__ == "__main__":
    unittest.main()
