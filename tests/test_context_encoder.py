"""Episode loading, sampling and model shapes of zero_shot_rl.latent.context_encoder."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.latent.context_encoder import ContextModel, Sampler, load_episodes, losses


def write_inputs(directory):
    rng = np.random.default_rng(0)
    multi = Path(directory) / "multi.npz"
    np.savez(multi, observations=rng.normal(size=(6, 11, 3)).astype(np.float32),
             actions=rng.normal(size=(6, 10, 2)).astype(np.float32),
             set=np.array([0, 0, 9, 9, 1, 1]), friction=np.zeros(10), gear=np.ones(10))
    stock = Path(directory) / "stock.npz"
    np.savez(stock, observations=rng.normal(size=(1, 3, 8, 3)).astype(np.float32),
             actions=rng.normal(size=(1, 3, 8, 2)).astype(np.float32), episodes=np.array([0, 39, 40]))
    return multi, stock


class ContextEncoderTest(unittest.TestCase):
    def test_load_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            ep = load_episodes(*([p] for p in write_inputs(d)))
        self.assertEqual(ep["n_sets"], 11)
        np.testing.assert_array_equal(ep["length"], [10] * 6 + [7] * 3)
        np.testing.assert_array_equal(ep["set"], [0, 0, 9, 9, 1, 1, 10, 10, 10])
        np.testing.assert_array_equal(ep["val"], [0, 0, 1, 1, 0, 0, 0, 0, 1])
        self.assertEqual(len(ep["s"]), 6 * 10 + 3 * 7)

    def test_sampler_draws_distinct_sets_and_two_episodes_of_each(self):
        with tempfile.TemporaryDirectory() as d:
            ep = load_episodes(*([p] for p in write_inputs(d)))
        sampler = Sampler(ep, ~ep["val"], torch.device("cpu"))
        g = torch.Generator().manual_seed(0)
        episode_of = np.repeat(np.arange(len(ep["start"])), ep["length"])
        for _ in range(20):
            (seg1, q1), (seg2, q2) = sampler.draw(8, 4, 5, g)
            self.assertEqual(seg1.shape, (3, 4))
            e1, e2 = episode_of[seg1.numpy()], episode_of[seg2.numpy()]
            self.assertTrue((e1 == e1[:, :1]).all() and (e2 == e2[:, :1]).all())
            np.testing.assert_array_equal(np.diff(seg1.numpy(), axis=1), 1)
            self.assertTrue((e1[:, 0] != e2[:, 0]).all())
            np.testing.assert_array_equal(ep["set"][e1[:, 0]], ep["set"][e2[:, 0]])
            self.assertEqual(len(set(ep["set"][e1[:, 0]])), 3)
            self.assertTrue((episode_of[q2.numpy()] == e2[:, :1]).all())

    def test_model_shapes_and_unit_context(self):
        for encoder in ("transformer", "mean"):
            model = ContextModel(3, 2, 5, context_dim=4, encoder=encoder, width=8, heads=2, hidden=16)
            c = model.encode(torch.randn(6, 7, 3), torch.randn(6, 7, 2), torch.randn(6, 7, 3))
            self.assertEqual(c.shape, (6, 4))
            torch.testing.assert_close(c.norm(dim=-1), torch.ones(6))
            self.assertEqual(model.predict(torch.randn(6, 9, 3), torch.randn(6, 9, 2), c[:, None]).shape, (6, 9, 5))

    def test_losses_run(self):
        with tempfile.TemporaryDirectory() as d:
            ep = load_episodes(*([p] for p in write_inputs(d)))
        sampler = Sampler(ep, ~ep["val"], torch.device("cpu"))
        data = tuple(torch.as_tensor(x) for x in (ep["s"], ep["a"], ep["s2"] - ep["s"], np.ones((len(ep["s"]), 5), np.float32)))
        model = ContextModel(3, 2, 5, context_dim=4, width=8, heads=2, hidden=16)
        pred, nce, acc = losses(model, data, sampler.draw(8, 3, 5, torch.Generator().manual_seed(0)), 0.1)
        self.assertTrue(torch.isfinite(pred) and torch.isfinite(nce) and 0 <= acc <= 1)


if __name__ == "__main__":
    unittest.main()
