"""Component tests for zero_shot_rl.finetune.anfis_z and zero_shot_rl.env.sensors."""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from analysis.table6_aggregate import TASKS as TABLE6_TASKS
from analysis.table6_aggregate import aggregate as table6_aggregate
from zero_shot_rl.env.sensors import (
    DOMAIN_TASKS,
    SUPPORTED_DOMAINS,
    extract_sensors,
    list_sensors,
    make_domain_env,
    validate_sensors,
)
from zero_shot_rl.finetune.anfis_z import (
    load_anfis_config,
    load_config,
    reinforce_step,
    resolve_anfis_sensors,
    sample_perturbations,
    warmstart_anfis,
    write_table6_csv,
)
from zero_shot_rl.latent.anfis import ANFIS, weighted_z
from zero_shot_rl.provenance import no_clobber, sha256

REPO = Path(__file__).resolve().parents[1]


class ANFISZTest(unittest.TestCase):
    def test_all_global_domain_configs_are_anfis_only_and_pair_with_reinforce_config(self):
        expected_rules = {
            "walker": 36,
            "quadruped": 36,
            "point_mass_maze": 36,
            "jaco": 27,
        }
        exp_cfg_path = REPO / "tasks/T6-anfis/configs/reinforce.yaml"
        for domain in SUPPORTED_DOMAINS:
            anfis_path = REPO / f"configs/anfis/{domain}.yaml"
            self.assertTrue(anfis_path.exists(), f"missing global config {anfis_path}")

            # Global ANFIS YAML must contain ONLY the `anfis:` block (no finetune/eval parameters).
            raw = yaml.safe_load(anfis_path.read_text())
            self.assertEqual(set(raw.keys()), {"anfis"}, f"{anfis_path} should only contain 'anfis'")

            # Standalone load_anfis_config covers every registered sensor for this domain.
            anfis_only = load_anfis_config(anfis_path)
            self.assertEqual(
                set(anfis_only["sensors"].keys()),
                set(list_sensors(domain)),
                f"{anfis_path} sensor keys mismatch SENSOR_REGISTRY[{domain!r}]",
            )

            # Pairing global ANFIS config with T6 experiment config resolves domain tasks & sensors.
            cfg = load_config(str(exp_cfg_path), anfis_config=str(anfis_path))
            self.assertEqual(cfg["anfis"]["domain"], domain)
            self.assertEqual(cfg["eval"]["tasks"], list(DOMAIN_TASKS[domain]))

            resolved_domain, specs, names = resolve_anfis_sensors(cfg["anfis"])
            self.assertEqual(resolved_domain, domain)
            self.assertEqual(len(specs), len(names))
            model = ANFIS.from_config(cfg["anfis"], out_dim=1)
            self.assertEqual(model.num_rules, expected_rules[domain])
            self.assertEqual(list(model.sensor_names), names)
            y = model(torch.randn(8, len(names)))
            self.assertEqual(y.shape, (8, 1))

    def test_t6_unnorm_config_inherits_global_walker_anfis_and_overrides_firing(self):
        cfg_path = REPO / "tasks/T6-anfis/configs/walker_unnorm.yaml"
        cfg = load_config(str(cfg_path))
        domain, _, names = resolve_anfis_sensors(cfg["anfis"])
        self.assertEqual(domain, "walker")
        self.assertEqual(names, ["torso_height", "torso_upright", "horizontal_velocity", "angmomentum"])
        self.assertEqual(cfg["eval"]["tasks"], list(DOMAIN_TASKS["walker"]))
        model = ANFIS.from_config(cfg["anfis"], out_dim=1)
        self.assertFalse(model.normalize_firing)
        self.assertEqual(model.num_rules, 36)

    def test_sensor_validation_rejects_cross_domain_sensors(self):
        with self.assertRaises(ValueError):
            validate_sensors("quadruped", ["torso_height"])
        with self.assertRaises(ValueError):
            validate_sensors("walker", ["pinch_pos_x"])
        with self.assertRaises(ValueError):
            validate_sensors("unknown_domain", ["x"])

    def test_extract_sensors_all_domains_and_walker_values(self):
        for domain in SUPPORTED_DOMAINS:
            env = make_domain_env(domain, seed=0)
            try:
                env.reset()
                env.step(np.zeros(env.action_spec().shape, dtype=np.float32))
                all_names = list_sensors(domain)
                vals = extract_sensors(env.physics, domain, all_names)
                self.assertEqual(vals.shape, (len(all_names),))
                self.assertEqual(vals.dtype, np.float32)
                self.assertTrue(np.isfinite(vals).all())

                if domain == "walker":
                    reward_names = ["torso_height", "torso_upright", "horizontal_velocity", "angmomentum"]
                    feats = extract_sensors(env.physics, "walker", reward_names)
                    expected = np.array([
                        env.physics.torso_height(),
                        env.physics.torso_upright(),
                        env.physics.horizontal_velocity(),
                        env.physics.angmomentum(),
                    ], dtype=np.float32)
                    np.testing.assert_allclose(feats, expected, atol=1e-6)
            finally:
                env.close()

    def test_warmstart_aligns_mu_z_with_z_r(self):
        torch.manual_seed(0)
        feats = torch.randn(256, 4)
        target = (
            torch.exp(-0.5 * ((feats[:, 0] - 1.0) / 0.8) ** 2)
            * torch.sigmoid(feats[:, 1])
            * torch.exp(-0.5 * ((feats[:, 2] - 0.5) / 1.0) ** 2)
        ).unsqueeze(-1)
        B = torch.randn(256, 50)
        z_r = weighted_z(target, B).detach()

        sensors = [
            {"name": "h", "num_terms": 3, "membership": "gaussian"},
            {"name": "u", "num_terms": 2, "membership": "sigmoid"},
            {"name": "vx", "num_terms": 3, "membership": "bell"},
            {"name": "ang", "num_terms": 2, "membership": "gaussian"},
        ]
        model = ANFIS(sensors=sensors, out_dim=1, normalize_inputs=True)
        model.init_from_data(feats, seed=0)
        info = warmstart_anfis(model, feats, target, B, z_r, steps=300, lr=2e-2)
        self.assertLess(info["warmstart_mse"], 5e-3)
        self.assertGreater(info["warmstart_cos"], 0.99)

    def test_reinforce_step_moves_mu_z_toward_higher_return_direction(self):
        torch.manual_seed(1)
        feats = torch.randn(128, 4)
        B = torch.randn(128, 50)
        sensors = [
            {"name": f"s{i}", "num_terms": m, "membership": "gaussian"}
            for i, m in enumerate((3, 2, 2, 2))
        ]
        model = ANFIS(sensors=sensors, out_dim=1, normalize_inputs=True)
        model.init_from_data(feats, seed=1)
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)

        v_star = F.normalize(torch.randn(50), dim=0)
        gen = torch.Generator().manual_seed(1)

        with torch.no_grad():
            score_before = float((weighted_z(model(feats), B) @ v_star).item())

        for _ in range(25):
            with torch.no_grad():
                mu_z = weighted_z(model(feats), B)
            z_tilde, z_sphere = sample_perturbations(mu_z, pop_size=16, sigma=0.5, antithetic=True, generator=gen)
            self.assertEqual(z_sphere.shape, (16, 50))
            torch.testing.assert_close(z_sphere.norm(dim=-1), torch.full((16,), math.sqrt(50)), atol=1e-5, rtol=0)
            rets = (z_sphere @ v_star).numpy()
            reinforce_step(model, opt, feats, B, z_tilde, rets, sigma=0.5, grad_clip=1.0)

        with torch.no_grad():
            score_after = float((weighted_z(model(feats), B) @ v_star).item())
        self.assertGreater(score_after, score_before + 1.0)

    def test_table6_csv_compatible_with_table6_aggregate(self):
        iqm_s0 = {t: {0: 400.0 + i * 20, 10: 450.0 + i * 20} for i, t in enumerate(TABLE6_TASKS)}
        iqm_s1 = {t: {0: 410.0 + i * 20, 10: 470.0 + i * 20} for i, t in enumerate(TABLE6_TASKS)}
        with tempfile.TemporaryDirectory() as tmp:
            p0, p1 = Path(tmp) / "s0.csv", Path(tmp) / "s1.csv"
            with no_clobber(p0) as tmp0:
                write_table6_csv(tmp0, TABLE6_TASKS, iqm_s0)
            with no_clobber(p1) as tmp1:
                write_table6_csv(tmp1, TABLE6_TASKS, iqm_s1)
            self.assertEqual(len(sha256(p0)), 64)
            runs = [pd.read_csv(p).set_index("step") for p in (p0, p1)]
            scores = np.stack([run[TABLE6_TASKS].to_numpy() for run in runs])
            agg = table6_aggregate(scores)
            self.assertEqual(agg.shape, (2,))
            self.assertGreater(agg[1], agg[0])

            res = subprocess.run(
                [sys.executable, "-m", "analysis.table6_aggregate", str(p0), str(p1)],
                cwd=REPO,
                capture_output=True,
                text=True,
                env={"PYTHONPATH": str(REPO), "PATH": ""},
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("selected step 10", res.stdout)


if __name__ == "__main__":
    unittest.main()
