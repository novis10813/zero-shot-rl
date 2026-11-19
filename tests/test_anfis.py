"""Component tests for zero_shot_rl.latent.anfis."""

import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from zero_shot_rl.latent.anfis import (
    ANFIS,
    CONSEQUENTS,
    MEMBERSHIPS,
    OUTPUT_ACTIVATIONS,
    T_NORMS,
    load_anfis,
    weighted_z,
)


class CustomConsequent(torch.nn.Module):
    """Minimal stand-in for a future KAN consequent block."""

    def __init__(self, in_dim, out_dim, num_rules):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, num_rules * out_dim)
        self.num_rules = num_rules
        self.out_dim = out_dim

    def forward(self, x):
        return self.proj(x).view(*x.shape[:-1], self.num_rules, self.out_dim)


class ANFISTest(unittest.TestCase):
    def test_shapes_and_weights_across_variants(self):
        torch.manual_seed(0)
        x2 = torch.randn(16, 24)
        x3 = torch.randn(4, 10, 24)
        for mem in MEMBERSHIPS:
            for cons in CONSEQUENTS:
                for out_dim in (1, 50):
                    model = ANFIS(in_dim=24, out_dim=out_dim, num_rules=8, membership=mem, consequent=cons)
                    y2, w2 = model(x2, return_weights=True)
                    self.assertEqual(y2.shape, (16, out_dim))
                    self.assertEqual(w2.shape, (16, 8))
                    self.assertTrue(torch.all(w2 >= 0))
                    torch.testing.assert_close(w2.sum(dim=-1), torch.ones(16), atol=1e-6, rtol=0)

                    y3 = model(x3)
                    self.assertEqual(y3.shape, (4, 10, out_dim))

    def test_high_dim_stability_and_gradients(self):
        torch.manual_seed(1)
        x = torch.randn(32, 30) * 5.0
        for mem in MEMBERSHIPS:
            for cons in CONSEQUENTS:
                model = ANFIS(in_dim=30, out_dim=1, num_rules=12, membership=mem, consequent=cons)
                y = model(x)
                self.assertTrue(torch.isfinite(y).all())
                loss = (y ** 2).mean()
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, f"missing grad for {mem}/{cons}/{name}")
                    self.assertTrue(torch.isfinite(p.grad).all(), f"non-finite grad for {mem}/{cons}/{name}")
                    self.assertGreater(p.grad.abs().sum().item(), 0.0, f"zero grad for {mem}/{cons}/{name}")

    def test_zero_order_matches_active_rule_center(self):
        model = ANFIS(in_dim=3, out_dim=2, num_rules=2, membership="gaussian", consequent="zero_order",
                      temperature=0.01)
        with torch.no_grad():
            model.membership.centers.copy_(torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))
            model.consequent.bias.copy_(torch.tensor([[1.5, -2.0], [7.0, 3.0]]))
        y, w = model(torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]), return_weights=True)
        torch.testing.assert_close(w, torch.eye(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(y, torch.tensor([[1.5, -2.0], [7.0, 3.0]]), atol=1e-5, rtol=0)

    def test_init_from_data_and_normalization(self):
        torch.manual_seed(2)
        x = torch.randn(200, 24) * 3.0 + 5.0
        model = ANFIS(in_dim=24, out_dim=1, num_rules=10, normalize_inputs=True)
        model.init_from_data(x, seed=0)
        torch.testing.assert_close(model.x_mean, x.mean(0), atol=1e-5, rtol=0)
        torch.testing.assert_close(model.x_std, x.std(0, unbiased=False), atol=1e-5, rtol=0)
        w = model.firing_weights(x)
        # Rules initialized on normalised data should not collapse to a single rule for all samples.
        self.assertGreater(w.argmax(dim=-1).unique().numel(), 1)

    def test_fits_product_membership_reward(self):
        """ANFIS fits a smooth Walker-like product-of-tolerances reward to low MSE."""
        torch.manual_seed(3)
        x = torch.randn(512, 4)
        # Product of two smooth Gaussian tolerances (mimicking standing * moving).
        target = (torch.exp(-0.5 * ((x[:, 0] - 1.0) / 0.8) ** 2)
                  * torch.exp(-0.5 * ((x[:, 1] + 0.5) / 1.0) ** 2)).unsqueeze(-1)
        model = ANFIS(in_dim=4, out_dim=1, num_rules=8, membership="gaussian", consequent="first_order")
        model.init_from_data(x, seed=3)
        opt = torch.optim.Adam(model.parameters(), lr=3e-2)
        for _ in range(250):
            opt.zero_grad()
            loss = F.mse_loss(model(x), target)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 5e-3)

    def test_custom_consequent_injection(self):
        torch.manual_seed(4)
        custom = CustomConsequent(in_dim=6, out_dim=2, num_rules=4)
        model = ANFIS(in_dim=6, out_dim=2, num_rules=4, consequent=custom)
        y = model(torch.randn(5, 6))
        self.assertEqual(y.shape, (5, 2))
        with self.assertRaises(ValueError):
            model.save("/tmp/unused.pt")

    def test_save_and_load_roundtrip(self):
        torch.manual_seed(5)
        x = torch.randn(16, 12) * 2.0 - 1.0
        model = ANFIS(in_dim=12, out_dim=1, num_rules=6, membership="bell", consequent="first_order",
                      normalize_inputs=True)
        model.init_from_data(x, seed=5)
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anfis.pt"
            model.save(path, task="walk", seed=5)
            loaded, meta = load_anfis(path)
        self.assertEqual(meta, {"task": "walk", "seed": 5})
        with torch.no_grad():
            actual = loaded(x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_weighted_z_matches_fb_norm_and_backpropagates_to_anfis(self):
        torch.manual_seed(6)
        states = torch.randn(64, 24)
        B = torch.randn(64, 50)  # frozen backward representations
        model = ANFIS(in_dim=24, out_dim=1, num_rules=8, normalize_inputs=True)
        model.init_from_data(states, seed=6)

        w = model(states)
        z = weighted_z(w, B)
        self.assertEqual(z.shape, (50,))
        self.assertAlmostEqual(z.norm().item(), math.sqrt(50), places=5)

        # Matches direct FB.infer_z formula: sqrt(d) * normalize(w^T B / N)
        ref = math.sqrt(50) * F.normalize((w.T @ B) / len(states), dim=1).squeeze(0)
        torch.testing.assert_close(z, ref, atol=1e-6, rtol=0)

        # Simulated REINFORCE-style surrogate loss on z propagates to both membership & consequent.
        target_dir = F.normalize(torch.randn(50), dim=0)
        loss = -(z * target_dir).sum()
        loss.backward()
        self.assertGreater(model.membership.centers.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.membership.raw_sigma.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.consequent.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.consequent.bias.grad.abs().sum().item(), 0.0)

    def test_from_config_composes_tnorm_mlp_activation_and_roundtrips(self):
        torch.manual_seed(7)
        x = torch.randn(12, 4)
        for t_norm in T_NORMS:
            for act in OUTPUT_ACTIVATIONS:
                cfg = {
                    "num_rules": 5,
                    "normalize_inputs": True,
                    "t_norm": t_norm,
                    "output_activation": act,
                    "membership": {"type": "bell", "min_scale": 0.002, "init_scale": 1.2, "init_slope": 1.5},
                    "consequent": {"type": "mlp", "hidden": 16, "layers": 2, "activation": "silu"},
                }
                model = ANFIS.from_config(cfg, in_dim=4, out_dim=1)
                model.init_from_data(x, seed=7)
                y = model(x)
                self.assertEqual(y.shape, (12, 1))
                if act == "sigmoid":
                    self.assertTrue(torch.all((y > 0) & (y < 1)))
                elif act == "softplus":
                    self.assertTrue(torch.all(y > 0))
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "composed.pt"
                    model.save(path, tag=f"{t_norm}_{act}")
                    loaded, meta = load_anfis(path)
                self.assertEqual(meta["tag"], f"{t_norm}_{act}")
                with torch.no_grad():
                    torch.testing.assert_close(loaded(x), model(x), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
