"""adapt() of zero_shot_rl.latent.action_adapt on a predictor whose optimum is known."""

import unittest

import torch

from zero_shot_rl.latent.action_adapt import adapt


class Shift:
    """g(s, a, c) = a + c: the best a' is a + c0 - c1."""

    def predict(self, s, a, c):
        return a + c.expand_as(a)


class ActionAdaptTest(unittest.TestCase):
    def test_reaches_the_optimum_and_clips(self):
        a = torch.tensor([[0.0, 0.5], [0.9, -0.9]])
        c0, c1 = torch.zeros(1, 2), torch.tensor([[0.2, -0.3]])
        a2, before, after = adapt(Shift(), torch.zeros(2, 3), a, c1, c0, steps=500, lr=0.01)
        torch.testing.assert_close(a2, torch.tensor([[-0.2, 0.8], [0.7, -0.6]]), atol=1e-3, rtol=0)
        self.assertTrue((after < 1e-5).all() and (before > 0.1).all())
        a2, _, _ = adapt(Shift(), torch.zeros(1, 3), torch.tensor([[0.9, 0.0]]), torch.tensor([[-0.5, 0.0]]),
                         c0, steps=500, lr=0.01)
        self.assertAlmostEqual(a2[0, 0].item(), 1.0, places=6)

    def test_same_context_keeps_the_action(self):
        a = torch.tensor([[0.3, -0.4]])
        a2, _, after = adapt(Shift(), torch.zeros(1, 3), a, torch.zeros(1, 2), torch.zeros(1, 2), steps=50, lr=0.05)
        torch.testing.assert_close(a2, a)
        self.assertEqual(after.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
