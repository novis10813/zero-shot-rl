"""AUROC and best Youden threshold of analyze.detect on known samples."""

import unittest

import numpy as np

from analyze.detect import auroc, best_youden
from analyze.prediction import r2_dims
from analyze.windows import window_means


class DetectTest(unittest.TestCase):
    def test_auroc_extremes_and_ties(self):
        self.assertEqual(auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])), 1.0)
        self.assertEqual(auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])), 0.0)
        self.assertEqual(auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])), 0.5)

    def test_auroc_matches_pair_count(self):
        rng = np.random.default_rng(0)
        neg, pos = rng.normal(size=200), rng.normal(0.5, 1, size=300)
        expected = (pos[:, None] > neg[None]).mean()
        self.assertAlmostEqual(auroc(neg, pos), expected, places=12)

    def test_best_youden_separates_perfectly(self):
        thr, tpr, fpr = best_youden(np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0]))
        self.assertEqual((tpr, fpr), (1.0, 0.0))
        self.assertTrue(2.0 <= thr < 5.0)

    def test_best_youden_overlapping(self):
        # threshold 3 gives TPR 0.75, FPR 0; every other threshold has a smaller difference
        thr, tpr, fpr = best_youden(np.arange(4.0), np.array([1.5, 4.0, 5.0, 6.0]))
        self.assertEqual((thr, tpr, fpr), (3.0, 0.75, 0.0))


class AnalysisTest(unittest.TestCase):
    def test_window_means_drop_the_remainder(self):
        x = np.arange(14.0).reshape(2, 7)
        np.testing.assert_allclose(window_means(x, 3), [1.0, 4.0, 8.0, 11.0])
        np.testing.assert_allclose(window_means(x, 1), x.ravel())

    def test_r2_dims(self):
        np.testing.assert_allclose(r2_dims(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])), [1.0, 0.0, -1.0])


if __name__ == "__main__":
    unittest.main()
