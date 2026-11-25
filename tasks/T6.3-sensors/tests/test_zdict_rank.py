"""Tests for the T6.3 rule-dictionary rank analysis."""

import unittest

import numpy as np
import torch

from analyze.zdict_rank import projection_residual, rule_basis, spectrum_stats
from zero_shot_rl.finetune.anfis_z import load_anfis_config, resolve_anfis_sensors
from zero_shot_rl.latent.anfis import ANFIS

# Sensor sets compared in T6.3: (config path, sensors, Z_dict columns K * (D + 1)).
SENSOR_SETS = [
    ("configs/anfis/walker.yaml", 4, 36 * 5),
    ("tasks/T6.1-ood-fuzzy/configs/walker_actuator.yaml", 8, 256 * 9),
    ("tasks/T6.3-sensors/configs/walker_all37_linear.yaml", 37, 38),
]

SENSORS = [{"name": "a", "num_terms": 3}, {"name": "b", "num_terms": 2}]


class RuleBasisTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(64, 2) * 3 + 1

    def _model(self, consequent):
        model = ANFIS(SENSORS, consequent=consequent, normalize_inputs=True)
        model.init_from_data(self.x)
        return model

    def test_first_order_basis_reproduces_output(self):
        model = self._model("first_order")
        c = model.consequent
        theta = torch.cat([c.weight[:, 0, :], c.bias[:, 0:1]], dim=1).reshape(-1)
        psi = rule_basis(model, self.x)
        self.assertEqual(psi.shape, (64, 6 * 3))
        with torch.no_grad():
            torch.testing.assert_close(psi @ theta, model(self.x)[:, 0])

    def test_zero_order_basis_reproduces_output(self):
        model = self._model("zero_order")
        psi = rule_basis(model, self.x)
        self.assertEqual(psi.shape, (64, 6))
        with torch.no_grad():
            torch.testing.assert_close(psi @ model.consequent.bias[:, 0], model(self.x)[:, 0])

    def test_rejects_nonlinear_output(self):
        model = ANFIS(SENSORS, output_activation="softplus")
        with self.assertRaises(ValueError):
            rule_basis(model, self.x)


class SensorSetConfigTest(unittest.TestCase):
    def test_sensor_sets_build(self):
        for path, num_sensors, num_columns in SENSOR_SETS:
            with self.subTest(path=path):
                cfg = load_anfis_config(path)
                _, _, names = resolve_anfis_sensors(cfg)
                self.assertEqual(len(names), num_sensors)
                model = ANFIS.from_config(cfg, out_dim=1)
                x = torch.randn(32, num_sensors)
                model.init_from_data(x)
                self.assertEqual(rule_basis(model, x).shape, (32, num_columns))


class SpectrumTest(unittest.TestCase):
    def test_stats(self):
        stats = spectrum_stats(np.array([4.0, 2.0, 1e-5]), tol=1e-3)
        self.assertEqual(stats["numeric_rank"], 2)
        self.assertAlmostEqual(stats["condition_number"], 4e5)
        self.assertAlmostEqual(spectrum_stats(np.ones(5), 1e-3)["effective_rank"], 5.0)

    def test_projection_residual(self):
        z_dict = np.eye(4)[:, :2]
        self.assertAlmostEqual(projection_residual(z_dict, np.array([1.0, 2.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(projection_residual(z_dict, np.array([0.0, 0.0, 3.0, 4.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
