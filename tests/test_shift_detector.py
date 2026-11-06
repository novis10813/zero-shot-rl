"""Mahalanobis score, covariance and save/load of zero_shot_rl.latent.shift_detector."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.latent.shift_detector import ShiftDetector, covariance, load_detector


def detector(eps=1e-3):
    torch.manual_seed(0)
    det = ShiftDetector(obs_dim=4, action_dim=2, z_dim=3, hidden=8, layers=2, eps=eps)
    det.e_mean.copy_(torch.tensor([0.5, -1.0, 2.0]))
    det.e_std.copy_(torch.tensor([2.0, 1.0, 0.5]))
    return det.eval()


class ShiftDetectorTest(unittest.TestCase):
    def test_distance_is_mahalanobis(self):
        det = detector()
        a = torch.randn(3, 3).double()
        sigma = a @ a.T + torch.eye(3, dtype=torch.float64)
        det.set_covariance(sigma)
        s, act, z, e = torch.randn(10, 4), torch.rand(10, 2), torch.randn(10, 3), torch.randn(10, 3)
        r = (e - det.mu(s, act, z)).double().detach()
        expected = np.einsum("ni,ij,nj->n", r, torch.linalg.inv(sigma + det.eps * torch.eye(3, dtype=torch.float64)), r)
        np.testing.assert_allclose(det.distance(s, act, z, e).numpy(), expected, rtol=1e-4)

    def test_in_distribution_mean_is_dimension(self):
        det = detector(eps=1e-9)
        torch.manual_seed(1)
        r = torch.randn(200_000, 3) @ torch.tensor([[2.0, 0, 0], [0.5, 1, 0], [0, 0.3, 0.1]])
        det.set_covariance(covariance(r))
        d = ((r.double() @ det.precision_chol.double()) ** 2).sum(-1)
        self.assertAlmostEqual(d.mean().item(), 3.0, delta=0.05)

    def test_shifted_residual_scores_higher(self):
        det = detector(eps=1e-6)
        det.set_covariance(torch.eye(3, dtype=torch.float64) * 0.01)
        s, act, z = torch.randn(5, 4), torch.rand(5, 2), torch.randn(5, 3)
        mu = det.mu(s, act, z)
        self.assertTrue(torch.all(det.distance(s, act, z, mu + 1.0) > det.distance(s, act, z, mu + 0.01)))

    def test_save_load_roundtrip(self):
        det = detector()
        det.set_covariance(torch.eye(3, dtype=torch.float64) * 0.3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.pt"
            det.save(path, note="x")
            loaded = load_detector(path)
        s, act, z, e = torch.randn(6, 4), torch.rand(6, 2), torch.randn(6, 3), torch.randn(6, 3)
        torch.testing.assert_close(loaded.distance(s, act, z, e), det.distance(s, act, z, e))


if __name__ == "__main__":
    unittest.main()
