"""AUROC over windows and the gain of analyze.context."""

import unittest

import numpy as np

from analyze.context import window_auc


class ContextTest(unittest.TestCase):
    def test_window_auc(self):
        sets = np.array([0, 0, 1, 2])
        dist = np.array([[0.0, 5.0], [1.0, 2.0], [3.0, 4.0], [9.0, 9.0]])
        self.assertAlmostEqual(window_auc(dist, sets, 1), 0.75)
        self.assertAlmostEqual(window_auc(dist, sets, 1, start=0), 1.0)
        self.assertAlmostEqual(window_auc(dist, sets, 1, start=1), 0.5)
        self.assertAlmostEqual(window_auc(dist, sets, 2, rows=np.array([False, True, True, True])), 1.0)


if __name__ == "__main__":
    unittest.main()
