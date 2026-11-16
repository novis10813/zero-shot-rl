"""Paired differences of analyze.adapt_episode on known samples."""

import unittest

import numpy as np

from analyze.adapt_episode import paired


class AdaptEpisodeTest(unittest.TestCase):
    def test_paired_mean_and_standard_error(self):
        mean, se = paired([3.0, 5.0, 7.0, 9.0], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(mean, 5.0)
        self.assertAlmostEqual(se, np.std([2, 4, 6, 8], ddof=1) / 2)

    def test_identical_samples_have_no_difference(self):
        self.assertEqual(paired([1.0, 2.0], [1.0, 2.0]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
