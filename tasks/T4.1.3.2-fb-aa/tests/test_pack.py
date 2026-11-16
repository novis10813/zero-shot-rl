"""k-step and one-step errors of analyze.pack with a model that predicts no motion."""

import unittest

import numpy as np
import torch

from analyze.pack import model_errors
from zero_shot_rl.latent.dynamics_model import DynamicsModel


class PackTest(unittest.TestCase):
    def test_model_errors_of_a_still_model(self):
        model = DynamicsModel(2, 1, hidden=4, layers=1)
        torch.nn.init.zeros_(model.net[-1].weight)
        torch.nn.init.zeros_(model.net[-1].bias)
        s0 = np.zeros((1, 2))
        traj = np.array([[[3.0, 4.0], [6.0, 8.0], [6.0, 8.0]]])
        multi, one = model_errors(model.eval(), s0, traj, np.zeros((1, 3, 1)))
        np.testing.assert_allclose(multi, [[5.0, 10.0, 10.0]], atol=1e-6)
        np.testing.assert_allclose(one, [[5.0, 5.0, 10.0 / 3]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
