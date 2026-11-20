"""Validate T5.1-ood-fuzzy ANFIS sensor configs and reinforce config."""

import unittest
from pathlib import Path

import torch

from zero_shot_rl.finetune.anfis_z import load_config, resolve_anfis_sensors
from zero_shot_rl.latent.anfis import ANFIS

REPO = Path(__file__).resolve().parents[3]
TASK_DIR = REPO / "tasks/T5.1-ood-fuzzy"


class T51OodFuzzyConfigTest(unittest.TestCase):
    def test_custom_sensor_configs_build_expected_rules(self):
        exp_cfg = TASK_DIR / "configs/reinforce.yaml"
        expected = {
            "walker_friction.yaml": (7, 128),
            "walker_actuator.yaml": (8, 256),
        }
        for fname, (n_sensors, n_rules) in expected.items():
            cfg = load_config(str(exp_cfg), anfis_config=str(TASK_DIR / "configs" / fname))
            domain, _, names = resolve_anfis_sensors(cfg["anfis"])
            self.assertEqual(domain, "walker")
            self.assertEqual(len(names), n_sensors)
            self.assertEqual(cfg["eval"]["eval_episodes"], 20)
            self.assertEqual(cfg["eval"]["eval_seed_start"], 5)
            model = ANFIS.from_config(cfg["anfis"], out_dim=1)
            self.assertEqual(model.num_rules, n_rules)
            y = model(torch.randn(4, n_sensors))
            self.assertEqual(y.shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
