"""Unit tests for zero_shot_rl.latent.action_transform."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.latent.action_transform import (
    ActionTransformer,
    FeaturePredictor,
    finetune_g1,
    load_action_transformer,
    load_exorl_transitions,
    load_feature_predictor,
    pretrain_identity,
    train_action_transformer,
    train_g0_model,
)


class LinearGainPredictor(torch.nn.Module):
    """Exact linear predictor g(s, a) = gain * a for testing T_theta optimization."""

    def __init__(self, gain):
        super().__init__()
        self.gain = gain

    def forward(self, s, a):
        return self.gain * a


class ActionTransformTest(unittest.TestCase):
    def test_feature_predictor_save_load(self):
        model = FeaturePredictor(obs_dim=4, action_dim=2, feature_dim=3, hidden=16, layers=2)
        model.x_mean.fill_(1.0)
        model.x_std.fill_(2.0)
        s = torch.randn(5, 4)
        a = torch.randn(5, 2)
        expected = model(s, a)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g0.pt"
            model.save(path, tag="test")
            loaded, meta = load_feature_predictor(path)
            self.assertEqual(meta["tag"], "test")
            torch.testing.assert_close(loaded(s, a), expected)

    def test_action_transformer_bounds_and_identity_pretrain(self):
        torch.manual_seed(0)
        s = torch.randn(128, 4)
        a = torch.rand(128, 2) * 1.6 - 0.8
        t_net = ActionTransformer(obs_dim=4, action_dim=2, hidden=32, layers=2)
        pretrain_identity(t_net, s, a, steps=300, batch_size=64, lr=3e-3)
        with torch.no_grad():
            out = t_net(s, a)
        self.assertTrue((out > -1.0).all() and (out < 1.0).all())
        torch.testing.assert_close(out, a, atol=0.1, rtol=0.0)

    def test_action_transformer_recovers_inverse_shift(self):
        torch.manual_seed(0)
        s = torch.randn(256, 3)
        a = torch.rand(256, 2) * 1.2 - 0.6
        g0 = LinearGainPredictor(1.0)
        g1 = LinearGainPredictor(2.0)

        t_free, _ = train_action_transformer(
            g0, g1, s, a, lam=0.0, steps=600, batch_size=128, lr=1e-3,
            hidden=32, layers=2, pretrain_steps=100, seed=0,
        )
        t_reg, _ = train_action_transformer(
            g0, g1, s, a, lam=1.0, steps=600, batch_size=128, lr=1e-3,
            hidden=32, layers=2, pretrain_steps=100, seed=0,
        )
        with torch.no_grad():
            out_free = t_free(s, a)
            out_reg = t_reg(s, a)
        torch.testing.assert_close(out_free, 0.5 * a, atol=0.08, rtol=0.0)
        # Regularized solution should stay closer to a than to 0.5 * a
        self.assertLess((out_reg - a).abs().mean().item(), (out_free - a).abs().mean().item())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.pt"
            t_free.save(path, lam=0.0)
            loaded, meta = load_action_transformer(path)
            self.assertEqual(meta["lam"], 0.0)
            torch.testing.assert_close(loaded(s, a), out_free)

    def test_finetune_g1_does_not_mutate_g0(self):
        torch.manual_seed(0)
        g0 = FeaturePredictor(obs_dim=3, action_dim=2, feature_dim=4, hidden=16, layers=2)
        s = torch.randn(32, 3)
        a = torch.randn(32, 2)
        with torch.no_grad():
            before_g0 = g0(s, a).clone()
        target = before_g0 + 1.0
        g1, loss = finetune_g1(g0, s, a, target, steps=200, batch_size=32, lr=5e-3)
        with torch.no_grad():
            after_g0 = g0(s, a)
            after_g1 = g1(s, a)
        torch.testing.assert_close(after_g0, before_g0)
        self.assertLess(((after_g1 - target) ** 2).mean().item(), 0.05)
        self.assertLess(loss, 0.05)

    def test_train_g0_model_and_load_exorl_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.npz"
            ep0 = {
                "observation": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                "action": np.array([[9.0], [0.5], [-0.5]], dtype=np.float32),
            }
            np.savez(path, ep0=ep0)
            data = load_exorl_transitions(path)
            np.testing.assert_array_equal(data["s"], [[1.0, 2.0], [3.0, 4.0]])
            np.testing.assert_array_equal(data["a"], [[0.5], [-0.5]])
            np.testing.assert_array_equal(data["s2"], [[3.0, 4.0], [5.0, 6.0]])

        rng = np.random.default_rng(0)
        s = rng.normal(size=(100, 3)).astype(np.float32)
        a = rng.uniform(-1, 1, size=(100, 2)).astype(np.float32)
        b = np.concatenate([s, a], axis=-1).astype(np.float32)
        train_mask = np.arange(100) < 80
        val_mask = ~train_mask
        best, final, hist = train_g0_model(
            s, a, b, train_mask, val_mask, hidden=32, layers=2,
            steps=100, batch_size=32, lr=3e-3, val_every=50,
        )
        self.assertGreater(len(hist), 0)
        self.assertLess(hist[-1]["val_mse"], 0.2)


if __name__ == "__main__":
    unittest.main()
