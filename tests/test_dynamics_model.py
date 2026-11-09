"""Episode split, error, Mahalanobis score and save/load of zero_shot_rl.latent.dynamics_model."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.latent.dynamics_model import (
    DynamicsModel, exorl_statistics, fit, flatten, load_dynamics_model, split_episodes,
)

A = np.array([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3]])
B = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])


def episodes(n, gain=1.0, seed=0):
    """Linear dynamics s' = s + A s + gain B a + noise, [n, 50, dim] arrays."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, 50, 3))
    a = rng.uniform(-1, 1, size=(n, 50, 2))
    s2 = s + s @ A.T + gain * a @ B.T + 0.01 * rng.normal(size=s.shape)
    return {"s": s, "a": a, "s2": s2, "episode": np.arange(n)}


def args(**kw):
    base = dict(lr=3e-3, weight_decay=0.0, epochs=60, patience=10, batch_size=128, log_every=1000)
    return Namespace(**{**base, **kw})


class DynamicsModelTest(unittest.TestCase):
    def test_split_is_per_file_and_reproducible(self):
        masks = split_episodes([20, 20, 20], 4, seed=1)
        self.assertEqual([m.sum() for m in masks], [4, 4, 4])
        again = split_episodes([20, 20, 20], 4, seed=1)
        self.assertTrue(all((a == b).all() for a, b in zip(masks, again)))
        self.assertFalse(all((a == b).all() for a, b in zip(masks, split_episodes([20, 20, 20], 4, seed=2))))

    def test_flatten_splits_episodes(self):
        files = [episodes(10, seed=i) for i in range(2)]
        masks = split_episodes([10, 10], 3, seed=0)
        train, val = flatten(files, masks, False), flatten(files, masks, True)
        self.assertEqual((len(train["s"]), len(val["s"])), (14 * 50, 6 * 50))

    def test_trained_model_scores_shift_higher(self):
        torch.manual_seed(0)
        files = [episodes(20)]
        masks = split_episodes([20], 4, seed=0)
        train, val = flatten(files, masks, False), flatten(files, masks, True)
        model = DynamicsModel(3, 2, hidden=64, layers=2)
        model, best = fit(model, train, val, args(), torch.device("cpu"))
        self.assertLess(best, 0.05)
        model.set_covariance(model.error(val["s"], torch.as_tensor(val["a"], dtype=torch.float32), val["s2"]))
        test, shifted = episodes(5, seed=7), episodes(5, gain=2.0, seed=7)
        a = torch.as_tensor(test["a"], dtype=torch.float32).reshape(-1, 2)
        d0 = model.distance(model.error(test["s"].reshape(-1, 3), a, test["s2"].reshape(-1, 3)))
        d1 = model.distance(model.error(test["s"].reshape(-1, 3), a, shifted["s2"].reshape(-1, 3)))
        self.assertGreater(d1.mean().item(), 10 * d0.mean().item())
        self.assertAlmostEqual(d0.mean().item(), 3.0, delta=1.0)

    def test_distance_is_mahalanobis(self):
        torch.manual_seed(0)
        model = DynamicsModel(3, 2, hidden=8, layers=1, eps=1e-3)
        e = torch.randn(5000, 3) @ torch.tensor([[2.0, 0, 0], [0.5, 1, 0], [0, 0.3, 0.1]])
        model.set_covariance(e)
        sigma = np.cov(e.double().numpy().T) + 1e-3 * np.eye(3)
        x = e[:10].double().numpy()
        expected = np.einsum("ni,ij,nj->n", x, np.linalg.inv(sigma), x)
        np.testing.assert_allclose(model.distance(e[:10]).numpy(), expected, rtol=1e-3)

    def test_exorl_statistics_pair_observation_with_next_action(self):
        rng = np.random.default_rng(0)
        eps = [{"observation": rng.normal(size=(n, 3)), "action": rng.normal(size=(n, 2))} for n in (6, 9)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.npz"
            np.savez(path, **{str(i): np.array(e, dtype=object) for i, e in enumerate(eps)})
            (x_mean, x_std), (y_mean, y_std) = exorl_statistics(path)
        x = np.concatenate([np.concatenate([e["observation"][:-1], e["action"][1:]], 1) for e in eps])
        y = np.concatenate([e["observation"][1:] - e["observation"][:-1] for e in eps])
        for got, expected in ((x_mean, x.mean(0)), (x_std, x.std(0)), (y_mean, y.mean(0)), (y_std, y.std(0))):
            np.testing.assert_allclose(got, expected)

    def test_fixed_statistics_are_used(self):
        torch.manual_seed(0)
        files = [episodes(10)]
        masks = split_episodes([10], 2, seed=0)
        train, val = flatten(files, masks, False), flatten(files, masks, True)
        model = DynamicsModel(3, 2, hidden=8, layers=1)
        stats = ((np.arange(5.0), np.arange(1.0, 6.0)), (np.arange(3.0), np.arange(1.0, 4.0)))
        fit(model, train, val, args(epochs=1), torch.device("cpu"), stats)
        for buffer, expected in ((model.x_mean, stats[0][0]), (model.x_std, stats[0][1]),
                                 (model.y_mean, stats[1][0]), (model.y_std, stats[1][1])):
            np.testing.assert_allclose(buffer.numpy(), expected)

    def test_next_state_target_gives_displacement_error(self):
        torch.manual_seed(0)
        files = [episodes(20)]
        masks = split_episodes([20], 4, seed=0)
        train, val = flatten(files, masks, False), flatten(files, masks, True)
        model = DynamicsModel(3, 2, hidden=64, layers=2, target="next_state")
        model, best = fit(model, train, val, args(), torch.device("cpu"))
        self.assertLess(best, 0.05)
        s, a, s2 = val["s"], torch.as_tensor(val["a"], dtype=torch.float32), val["s2"]
        s32 = torch.as_tensor(s, dtype=torch.float32)
        predicted_next = (model.standardised(s32, a) * model.y_std + model.y_mean).detach()
        np.testing.assert_allclose(model.error(s, a, s2).numpy(), (s2 - predicted_next.numpy()), atol=1e-4)
        np.testing.assert_allclose(model(s32, a).detach().numpy(), (predicted_next - s32).numpy(), atol=1e-5)
        # the next-state statistics are those of s (first obs_dim entries of the input statistics)
        stats = ((np.arange(5.0), np.arange(1.0, 6.0)), (np.zeros(3), np.ones(3)))
        fit(model, train, val, args(epochs=1), torch.device("cpu"), stats)
        np.testing.assert_allclose(model.y_mean.numpy(), np.arange(3.0))
        np.testing.assert_allclose(model.y_std.numpy(), np.arange(1.0, 4.0))

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            DynamicsModel(3, 2, target="state")

    def test_save_load_roundtrip(self):
        torch.manual_seed(0)
        model = DynamicsModel(3, 2, hidden=8, layers=1).eval()
        s, a = torch.randn(4, 3), torch.randn(4, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            model.save(path, note="x")
            loaded, meta = load_dynamics_model(path)
        self.assertEqual(meta["note"], "x")
        torch.testing.assert_close(loaded(s, a), model(s, a))


if __name__ == "__main__":
    unittest.main()
