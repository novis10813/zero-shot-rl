"""Component tests for zero_shot_rl.latent.anfis (Cartesian-grid ANFIS)."""

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
    parse_sensor_specs,
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


def _uniform_sensors(term_counts, membership="gaussian"):
    return [
        {"name": f"s{i}", "num_terms": m, "membership": membership}
        for i, m in enumerate(term_counts)
    ]


class ANFISTest(unittest.TestCase):
    def test_shapes_and_weights_across_all_memberships_and_consequents(self):
        torch.manual_seed(0)
        x2 = torch.randn(16, 3)
        x3 = torch.randn(4, 10, 3)
        terms = (2, 3, 2)
        expected_rules = math.prod(terms)
        for mem in MEMBERSHIPS:
            for cons in CONSEQUENTS:
                for out_dim in (1, 50):
                    model = ANFIS(sensors=_uniform_sensors(terms, mem), out_dim=out_dim, consequent=cons)
                    self.assertEqual(model.num_rules, expected_rules)
                    y2, w2 = model(x2, return_weights=True)
                    self.assertEqual(y2.shape, (16, out_dim))
                    self.assertEqual(w2.shape, (16, expected_rules))
                    self.assertTrue(torch.all(w2 >= 0))
                    torch.testing.assert_close(w2.sum(dim=-1), torch.ones(16), atol=1e-6, rtol=0)

                    y3 = model(x3)
                    self.assertEqual(y3.shape, (4, 10, out_dim))

    def test_single_sensor_edge_case(self):
        torch.manual_seed(0)
        x = torch.randn(12, 1)
        model = ANFIS(sensors=_uniform_sensors((4,), "gaussian"), out_dim=2)
        self.assertEqual(model.num_rules, 4)
        self.assertEqual(model.rule_indices.shape, (4, 1))
        y, w = model(x, return_weights=True)
        self.assertEqual(y.shape, (12, 2))
        self.assertEqual(w.shape, (12, 4))
        torch.testing.assert_close(w.sum(dim=-1), torch.ones(12), atol=1e-6, rtol=0)

    def test_all_six_memberships_heterogeneous_stability_and_gradients(self):
        torch.manual_seed(1)
        # Dense samples spanning [-3, 3] so piecewise slopes (triangular/trapezoidal) all receive gradients.
        x = torch.linspace(-3.0, 3.0, 64).unsqueeze(-1).repeat(1, 6) + 0.1 * torch.randn(64, 6)
        sensors = [
            {"name": "s_gauss", "num_terms": 2, "membership": {"type": "gaussian", "min_sigma": 1e-3, "init_sigma": 1.0}},
            {"name": "s_bell", "num_terms": 3, "membership": {"type": "bell", "min_scale": 1e-3, "init_scale": 1.0, "init_slope": 1.5}},
            {"name": "s_tri", "num_terms": 2, "membership": {"type": "triangular", "min_width": 1e-3, "init_width": 2.0}},
            {"name": "s_trap", "num_terms": 2, "membership": {"type": "trapezoidal", "min_width": 1e-3, "init_slope_width": 1.5, "init_top_width": 0.5}},
            {"name": "s_sig", "num_terms": 2, "membership": {"type": "sigmoid", "init_slope": 2.0}},
            {"name": "s_psig", "num_terms": 2, "membership": {"type": "psig", "min_width": 1e-3, "init_width": 1.0, "init_slope": 2.0}},
        ]
        model = ANFIS(sensors=sensors, out_dim=1, consequent="first_order", normalize_inputs=True)
        self.assertEqual(model.num_rules, 2 * 3 * 2 * 2 * 2 * 2)
        model.init_from_data(x)

        per_sensor = model.sensor_memberships(x)
        self.assertEqual([m.shape for m in per_sensor], [(64, 2), (64, 3), (64, 2), (64, 2), (64, 2), (64, 2)])
        for m in per_sensor:
            self.assertTrue(torch.all((m > 0) & (m <= 1.0 + 1e-6)))

        y = model(x)
        self.assertTrue(torch.isfinite(y).all())
        loss = (y ** 2).mean()
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"missing grad for {name}")
            self.assertTrue(torch.isfinite(p.grad).all(), f"non-finite grad for {name}")
            self.assertGreater(p.grad.abs().sum().item(), 0.0, f"zero grad for {name}")

    def test_zero_order_matches_active_cartesian_rule(self):
        # 2 sensors x 2 terms each -> 4 Cartesian rules: (0,0), (0,1), (1,0), (1,1).
        model = ANFIS(
            sensors=_uniform_sensors((2, 2), "gaussian"),
            out_dim=2,
            consequent="zero_order",
            temperature=0.01,
        )
        with torch.no_grad():
            model.memberships[0].centers.copy_(torch.tensor([0.0, 10.0]))
            model.memberships[1].centers.copy_(torch.tensor([0.0, 10.0]))
            rule_vals = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]])
            model.consequent.bias.copy_(rule_vals)
        pts = torch.tensor([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]])
        y, w = model(pts, return_weights=True)
        torch.testing.assert_close(w, torch.eye(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(y, rule_vals, atol=1e-5, rtol=0)

    def test_init_from_data_and_normalization(self):
        torch.manual_seed(2)
        x = torch.randn(300, 3) * 3.0 + 5.0
        model = ANFIS(sensors=_uniform_sensors((3, 2, 3), "gaussian"), out_dim=1, normalize_inputs=True)
        model.init_from_data(x)
        torch.testing.assert_close(model.x_mean, x.mean(0), atol=1e-5, rtol=0)
        torch.testing.assert_close(model.x_std, x.std(0, unbiased=False), atol=1e-5, rtol=0)
        w = model.firing_weights(x)
        self.assertGreater(w.argmax(dim=-1).unique().numel(), 1)

    def test_fits_product_membership_reward(self):
        """Cartesian-grid ANFIS fits a smooth Walker-like product-of-tolerances reward to low MSE."""
        torch.manual_seed(3)
        x = torch.randn(512, 4)
        target = (
            torch.exp(-0.5 * ((x[:, 0] - 1.0) / 0.8) ** 2)
            * torch.exp(-0.5 * ((x[:, 1] + 0.5) / 1.0) ** 2)
        ).unsqueeze(-1)
        model = ANFIS(
            sensors=_uniform_sensors((3, 3, 2, 2), "gaussian"),
            out_dim=1,
            consequent="first_order",
            normalize_inputs=True,
        )
        model.init_from_data(x)
        opt = torch.optim.Adam(model.parameters(), lr=3e-2)
        for _ in range(200):
            opt.zero_grad()
            loss = F.mse_loss(model(x), target)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 5e-3)

    def test_custom_consequent_injection(self):
        torch.manual_seed(4)
        custom = CustomConsequent(in_dim=2, out_dim=2, num_rules=6)
        model = ANFIS(sensors=_uniform_sensors((2, 3), "gaussian"), out_dim=2, consequent=custom)
        y = model(torch.randn(5, 2))
        self.assertEqual(y.shape, (5, 2))
        with self.assertRaises(ValueError):
            model.save("/tmp/unused.pt")

    def test_max_rules_guard(self):
        with self.assertRaises(ValueError):
            ANFIS(sensors=_uniform_sensors((4, 4, 4, 4), "gaussian"), max_rules=100)

    def test_parse_sensor_specs_isolates_default_membership_kwargs(self):
        cfg = {
            "default_membership": {"type": "gaussian", "min_sigma": 0.002, "init_sigma": 1.2},
            "sensors": {
                "a": {"enabled": True, "num_terms": 3},
                "b": {"enabled": False, "num_terms": 5},
                "c": {"enabled": True, "num_terms": 2, "membership": {"type": "bell", "init_slope": 1.5}},
                "d": {"enabled": True, "num_terms": 2, "membership": "sigmoid"},
            },
        }
        specs = parse_sensor_specs(cfg)
        self.assertEqual([s["name"] for s in specs], ["a", "c", "d"])
        self.assertEqual(specs[0]["membership"], {"type": "gaussian", "min_sigma": 0.002, "init_sigma": 1.2})
        # Bell and sigmoid must NOT inherit gaussian's min_sigma / init_sigma.
        self.assertEqual(specs[1]["membership"], {"type": "bell", "init_slope": 1.5})
        self.assertEqual(specs[2]["membership"], {"type": "sigmoid"})
        model = ANFIS.from_config(cfg, out_dim=1)
        self.assertEqual(model.num_rules, 3 * 2 * 2)

    def test_save_and_load_roundtrip(self):
        torch.manual_seed(5)
        x = torch.randn(16, 4) * 2.0 - 1.0
        sensors = [
            {"name": "h", "num_terms": 3, "membership": "gaussian"},
            {"name": "u", "num_terms": 2, "membership": "bell"},
            {"name": "vx", "num_terms": 3, "membership": "psig"},
            {"name": "ang", "num_terms": 2, "membership": "sigmoid"},
        ]
        model = ANFIS(sensors=sensors, out_dim=1, consequent="first_order", normalize_inputs=True)
        model.init_from_data(x)
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anfis.pt"
            model.save(path, task="walk", seed=5)
            loaded, meta = load_anfis(path)
        self.assertEqual(meta, {"task": "walk", "seed": 5})
        self.assertEqual(loaded.sensor_names, ("h", "u", "vx", "ang"))
        self.assertEqual(loaded.term_counts, (3, 2, 3, 2))
        with torch.no_grad():
            actual = loaded(x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_weighted_z_matches_fb_norm_and_backpropagates_to_anfis(self):
        torch.manual_seed(6)
        states = torch.randn(64, 4)
        B = torch.randn(64, 50)
        model = ANFIS(sensors=_uniform_sensors((3, 2, 2, 2), "gaussian"), out_dim=1, normalize_inputs=True)
        model.init_from_data(states)

        w = model(states)
        z = weighted_z(w, B)
        self.assertEqual(z.shape, (50,))
        self.assertAlmostEqual(z.norm().item(), math.sqrt(50), places=5)

        ref = math.sqrt(50) * F.normalize((w.T @ B) / len(states), dim=1).squeeze(0)
        torch.testing.assert_close(z, ref, atol=1e-6, rtol=0)

        target_dir = F.normalize(torch.randn(50), dim=0)
        loss = -(z * target_dir).sum()
        loss.backward()
        for mem in model.memberships:
            self.assertGreater(mem.centers.grad.abs().sum().item(), 0.0)
            self.assertGreater(mem.raw_sigma.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.consequent.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.consequent.bias.grad.abs().sum().item(), 0.0)

    def test_from_config_composes_tnorm_mlp_activation_and_roundtrips(self):
        torch.manual_seed(7)
        x = torch.randn(12, 3)
        for t_norm in T_NORMS:
            for act in OUTPUT_ACTIVATIONS:
                cfg = {
                    "normalize_inputs": True,
                    "t_norm": t_norm,
                    "output_activation": act,
                    "default_membership": {"type": "bell", "min_scale": 0.002, "init_scale": 1.2, "init_slope": 1.5},
                    "sensors": {
                        "s0": {"enabled": True, "num_terms": 2},
                        "s1": {"enabled": True, "num_terms": 3, "membership": {"type": "triangular"}},
                        "s2": {"enabled": True, "num_terms": 2, "membership": {"type": "trapezoidal"}},
                    },
                    "consequent": {"type": "mlp", "hidden": 16, "layers": 2, "activation": "silu"},
                }
                model = ANFIS.from_config(cfg, out_dim=1)
                model.init_from_data(x)
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

    def test_unnormalised_firing_mode_and_roundtrip(self):
        torch.manual_seed(8)
        x = torch.randn(16, 3)
        model = ANFIS(
            sensors=_uniform_sensors((2, 3, 2), "gaussian"),
            out_dim=1,
            consequent="zero_order",
            normalize_firing=False,
            temperature=1.0,
        )
        y, psi = model(x, return_weights=True)
        self.assertEqual(y.shape, (16, 1))
        self.assertEqual(psi.shape, (16, 12))
        self.assertTrue(torch.all((psi > 0) & (psi <= 1.0)))
        self.assertFalse(torch.allclose(psi.sum(dim=-1), torch.ones(16), atol=1e-2))
        expected = psi @ model.consequent.bias
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=0)
        (y ** 2).mean().backward()
        for mem in model.memberships:
            self.assertGreater(mem.centers.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.consequent.bias.grad.abs().sum().item(), 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unnorm.pt"
            model.save(path)
            loaded, _ = load_anfis(path)
        self.assertFalse(loaded.normalize_firing)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
