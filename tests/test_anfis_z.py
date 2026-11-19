"""Component tests for zero_shot_rl.finetune.anfis_z."""

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

from analysis.table6_aggregate import TASKS as TABLE6_TASKS
from analysis.table6_aggregate import aggregate as table6_aggregate
from zero_shot_rl.env.walker import make_env
from zero_shot_rl.finetune.anfis_z import (
    ALL_TASKS,
    extract_physics_features,
    feature_dim,
    load_config,
    reinforce_step,
    sample_perturbations,
    warmstart_anfis,
    write_table6_csv,
)
from zero_shot_rl.latent.anfis import ANFIS, weighted_z

REPO = Path(__file__).resolve().parents[1]


class ANFISZTest(unittest.TestCase):
    def test_yaml_config_loads_and_builds_anfis(self):
        cfg_path = REPO / "tasks/T6-anfis/configs/anfis_z.yaml"
        cfg = load_config(str(cfg_path))
        self.assertEqual(cfg["eval"]["tasks"], ALL_TASKS)
        dim = feature_dim(cfg["anfis"]["features"])
        self.assertEqual(dim, 4)
        model = ANFIS.from_config(cfg["anfis"], in_dim=dim, out_dim=1)
        y = model(torch.randn(8, dim))
        self.assertEqual(y.shape, (8, 1))

    def test_extract_physics_features_matches_walker_physics(self):
        env = make_env(0)
        try:
            ts = env.reset()
            ts = env.step(np.zeros(6, dtype=np.float32))
            obs = ts.observation["observations"]
            names = ["torso_height", "torso_upright", "horizontal_velocity", "angmomentum"]
            feats = extract_physics_features(env.physics, names, obs)
            expected = np.array([
                env.physics.torso_height(),
                env.physics.torso_upright(),
                env.physics.horizontal_velocity(),
                env.physics.angmomentum(),
            ], dtype=np.float32)
            np.testing.assert_allclose(feats, expected, atol=1e-6)

            with_obs = extract_physics_features(env.physics, ["torso_height", "observations"], obs)
            self.assertEqual(with_obs.shape, (25,))
            self.assertEqual(feature_dim(["torso_height", "observations"]), 25)
        finally:
            env.close()

    def test_warmstart_aligns_mu_z_with_z_r(self):
        torch.manual_seed(0)
        feats = torch.randn(256, 4)
        # Smooth Walker-like tolerance product reward
        target = (
            torch.exp(-0.5 * ((feats[:, 0] - 1.0) / 0.8) ** 2)
            * torch.sigmoid(feats[:, 1])
            * torch.exp(-0.5 * ((feats[:, 2] - 0.5) / 1.0) ** 2)
        ).unsqueeze(-1)
        B = torch.randn(256, 50)
        z_r = weighted_z(target, B).detach()

        model = ANFIS(in_dim=4, out_dim=1, num_rules=8, normalize_inputs=True)
        model.init_from_data(feats, seed=0)
        info = warmstart_anfis(model, feats, target, B, z_r, steps=300, lr=2e-2)
        self.assertLess(info["warmstart_mse"], 5e-3)
        self.assertGreater(info["warmstart_cos"], 0.99)

    def test_reinforce_step_moves_mu_z_toward_higher_return_direction(self):
        torch.manual_seed(1)
        feats = torch.randn(128, 4)
        B = torch.randn(128, 50)
        model = ANFIS(in_dim=4, out_dim=1, num_rules=8, normalize_inputs=True)
        model.init_from_data(feats, seed=1)
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)

        # Synthetic linear reward landscape on the z-sphere: R(z) = z^T v_star
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
            write_table6_csv(p0, TABLE6_TASKS, iqm_s0)
            write_table6_csv(p1, TABLE6_TASKS, iqm_s1)
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
