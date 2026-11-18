"""Tests for unified train/finetune/eval entrypoints, scripts, shims, and separated FB-AA flow."""

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.eval.action_transform import rollout_eval
from zero_shot_rl.eval.value_correction_checkpoints import find, find_final
from zero_shot_rl.finetune.action_transform import finetune_config
from zero_shot_rl.latent.action_transform import FeaturePredictor, load_action_transformer

REPO = Path(__file__).resolve().parents[1]


class _FakeFBNet:
    def backward_representation(self, obs):
        return obs[:, :4]


class _FakeAgent:
    def __init__(self, action_dim=6):
        self.FB = _FakeFBNet()
        self.action_dim = action_dim

    def actor(self, obs, z, std, sample=False):
        out = torch.zeros(obs.shape[0], self.action_dim, device=obs.device, dtype=obs.dtype)
        return out, None


class _FakePhysics:
    def data(self):
        return None


class _FakeEnv:
    def __init__(self, obs_dim=24):
        self.obs_dim = obs_dim
        self.physics = _FakePhysics()
        self._t = 0

    def reset(self):
        self._t = 0
        return type("TS", (), {"observation": {"observations": np.zeros(self.obs_dim, dtype=np.float32)}})()

    def step(self, action):
        self._t += 1
        return type("TS", (), {"observation": {"observations": np.full(self.obs_dim, 0.01 * self._t, dtype=np.float32)}})()

    def close(self):
        pass


class EntrypointTests(unittest.TestCase):
    def test_unified_python_clis_help(self):
        env = dict(os.environ, PYTHONPATH=str(REPO))
        for mod in ("zero_shot_rl.train", "zero_shot_rl.finetune", "zero_shot_rl.eval",
                    "zero_shot_rl.train.g0", "zero_shot_rl.finetune.action_transform",
                    "zero_shot_rl.eval.action_transform"):
            res = subprocess.run([sys.executable, "-m", mod, "--help"], cwd=REPO, env=env,
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"{mod} failed: {res.stderr}")

    def test_shell_scripts_syntax_and_help(self):
        for script in ("scripts/train.sh", "scripts/finetune.sh", "scripts/evaluate.sh"):
            path = REPO / script
            self.assertTrue(path.exists(), f"Missing {script}")
            syntax = subprocess.run(["bash", "-n", str(path)], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            help_res = subprocess.run(["bash", str(path), "--help"], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(help_res.returncode, 0, help_res.stderr)

    def test_backward_compatibility_shims_import(self):
        for name in (
            "zero_shot_rl.latent.context_encoder",
            "zero_shot_rl.latent.context_score",
            "zero_shot_rl.latent.dynamics_model",
            "zero_shot_rl.latent.dynamics_score",
            "zero_shot_rl.latent.cvae_model",
            "zero_shot_rl.latent.cvae_score",
            "zero_shot_rl.eval.value_correction",
            "zero_shot_rl.eval.online_inference",
        ):
            mod = importlib.import_module(name)
            self.assertTrue(hasattr(mod, "main"), f"{name} missing main")

    def test_value_correction_find_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            q_dir = Path(tmp)
            (q_dir / "residual_walk.pt").touch()
            (q_dir / "residual_walk_ep10.pt").touch()
            (q_dir / "direct_stand.pt").touch()
            ep_found = find(q_dir)
            final_found = find_final(q_dir)
            self.assertEqual([(v, t, e) for v, t, e, _ in ep_found], [("residual", "walk", 10)])
            self.assertEqual([(v, t, e) for v, t, e, _ in final_found],
                             [("direct", "stand", "final"), ("residual", "walk", "final")])

    def test_separated_action_transform_finetune_and_eval(self):
        from unittest import mock

        agent = _FakeAgent(action_dim=6)
        g0 = FeaturePredictor(obs_dim=24, action_dim=6, feature_dim=4, hidden=16, layers=2)
        z_of = {"walk": torch.zeros(4, dtype=torch.float32)}
        reward_fns = {"walk": lambda physics: 1.0}

        with mock.patch("zero_shot_rl.finetune.action_transform.make_env", side_effect=lambda s, d: _FakeEnv()), \
             mock.patch("zero_shot_rl.finetune.action_transform.STEPS", 5), \
             mock.patch("zero_shot_rl.eval.action_transform.STEPS", 5):
            _, transformer, metrics = finetune_config(
                agent, g0, None, z_of, ["walk"], reward_fns,
                finetune_episodes=1, finetune_seed=100,
                g1_steps=5, t_steps=5, t_hidden=16, t_layers=2, batch_size=4,
            )
            self.assertIn("g1_loss", metrics)
            with tempfile.TemporaryDirectory() as tmp:
                t_path = Path(tmp) / "transformer.pt"
                transformer.save(t_path, config="baseline")
                loaded, meta = load_action_transformer(t_path)
                self.assertEqual(meta["config"], "baseline")
                envs = [_FakeEnv()]
                z = torch.zeros(1, 4, dtype=torch.float32)
                rew, da = rollout_eval(agent, envs, z, [reward_fns["walk"]], "transform", 1.0, loaded)
                self.assertEqual(rew.shape, (1, 5))
                self.assertEqual(da.shape, (1, 5))
                self.assertAlmostEqual(float(rew.sum()), 5.0)


if __name__ == "__main__":
    unittest.main()
