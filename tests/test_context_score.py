"""Windows of zero_shot_rl.latent.context_score."""

import unittest

import numpy as np
import torch

from zero_shot_rl.latent.context_encoder import ContextModel
from zero_shot_rl.latent.context_score import contexts


class ContextScoreTest(unittest.TestCase):
    def test_contexts_encode_the_windows(self):
        torch.manual_seed(0)
        model = ContextModel(3, 2, 5, context_dim=4, encoder="mean", width=8, hidden=16).eval()
        obs = np.random.default_rng(0).normal(size=(5, 21, 3)).astype(np.float32)
        act = np.random.default_rng(1).normal(size=(5, 20, 2)).astype(np.float32)
        starts = np.array([0, 7, 15])
        c = contexts(model, obs, act, starts, 4, torch.device("cpu"), chunk=6)
        self.assertEqual(c.shape, (5, 3, 4))
        s = torch.as_tensor(obs[3, 7:11])
        with torch.no_grad():
            direct = model.encode(s[None], torch.as_tensor(act[3, 7:11])[None],
                                  torch.as_tensor(obs[3, 8:12])[None] - s[None])
        np.testing.assert_allclose(c[3, 1], direct[0].numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
