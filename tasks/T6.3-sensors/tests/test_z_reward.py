"""Tests for the implied-reward analysis r_z(s) = B(s)^T Sigma_B^{-1} z."""

import unittest

import numpy as np

from analyze.z_reward import affine_scale, bin_means, implied_rewards


class ZRewardTest(unittest.TestCase):
    def test_implied_reward_reproduces_z(self):
        rng = np.random.default_rng(0)
        b = rng.normal(size=(2000, 5)) @ rng.normal(size=(5, 5))
        zs = rng.normal(size=(3, 5))
        r = implied_rewards(b, zs)
        np.testing.assert_allclose(b.T @ r / len(b), zs.T, atol=1e-10)

    def test_implied_reward_recovers_reward_in_span(self):
        rng = np.random.default_rng(1)
        b = rng.normal(size=(2000, 4))
        reward = b @ np.array([1.0, -2.0, 0.5, 0.0])
        z = b.T @ reward / len(b)
        np.testing.assert_allclose(implied_rewards(b, z[None])[:, 0], reward, atol=1e-10)

    def test_affine_scale(self):
        x = np.linspace(0, 1, 50)
        self.assertEqual(tuple(np.round(affine_scale(x, 3 * x + 2), 6)), (3.0, 2.0))

    def test_bin_means(self):
        x = np.linspace(0, 1, 1001)
        centers, counts, means = bin_means(x, np.stack([x, 2 * x], axis=1), num_bins=5)
        self.assertEqual(means.shape, (5, 2))
        np.testing.assert_allclose(means[:, 1], 2 * means[:, 0])
        np.testing.assert_allclose(means[:, 0], centers, atol=0.01)
        self.assertEqual(counts.sum(), int(((x >= 0.01) & (x <= 0.99)).sum()))


if __name__ == "__main__":
    unittest.main()
