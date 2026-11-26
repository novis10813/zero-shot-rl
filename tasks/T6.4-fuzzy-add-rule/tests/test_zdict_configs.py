"""Tests for T6.4-fuzzy-add-rule Z_dict config sets."""

import unittest

import torch

from analysis.zdict_rank import rule_basis
from zero_shot_rl.finetune.anfis_z import load_anfis_config, resolve_anfis_sensors
from zero_shot_rl.latent.anfis import ANFIS

ADDITIVE_SETS = [
    ("tasks/T6.4-fuzzy-add-rule/configs/walker4_additive_zero.yaml", 4, 10, 10),
    ("tasks/T6.4-fuzzy-add-rule/configs/walker4_additive_first.yaml", 4, 10, 50),
    ("tasks/T6.4-fuzzy-add-rule/configs/walker8_additive_zero.yaml", 8, 16, 16),
    ("tasks/T6.4-fuzzy-add-rule/configs/walker8_additive_first.yaml", 8, 16, 144),
    ("tasks/T6.4-fuzzy-add-rule/configs/walker_all37_additive_zero.yaml", 37, 74, 74),
    ("tasks/T6.4-fuzzy-add-rule/configs/walker_all37_additive_first.yaml", 37, 74, 2812),
]


class AdditiveConfigTest(unittest.TestCase):
    def test_additive_configs_build_and_match_basis_shape(self):
        torch.manual_seed(0)
        for path, num_sensors, num_rules, num_columns in ADDITIVE_SETS:
            with self.subTest(path=path):
                cfg = load_anfis_config(path)
                _, _, names = resolve_anfis_sensors(cfg)
                self.assertEqual(len(names), num_sensors)
                model = ANFIS.from_config(cfg, out_dim=1)
                self.assertEqual(model.rule_mode, "additive")
                self.assertFalse(model.normalize_firing)
                self.assertEqual(model.num_rules, num_rules)
                x = torch.randn(32, num_sensors)
                model.init_from_data(x)
                self.assertEqual(rule_basis(model, x).shape, (32, num_columns))


if __name__ == "__main__":
    unittest.main()
