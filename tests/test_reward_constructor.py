import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rewards import RewardFunctionConstructor


class _DummyEnv:
    def __init__(self, state_dim: int = 4, obs_dim: int = 3):
        self._state = np.zeros(state_dim, dtype=np.float64)
        self._obs_dim = obs_dim
        self.physics = self

    def set_state(self, state):
        self._state = np.array(state, dtype=np.float64)

    def get_state(self):
        return self._state

    def step(self, action):
        self._state = self._state + np.asarray(action, dtype=np.float64).mean()
        return SimpleNamespace(
            observation={"observations": np.ones(self._obs_dim, dtype=np.float32)}
        )


class TestRewardFunctionConstructor(unittest.TestCase):
    def test_mixed_zero_dim_ndarray_and_float64(self):
        dummy_env = _DummyEnv()
        mod_a = SimpleNamespace(reward_function=lambda phys: np.array(phys.get_state()[0]))
        mod_b = SimpleNamespace(reward_function=lambda phys: np.float64(phys.get_state()[1]))

        def _fake_import(name):
            return mod_a if name.endswith(".task_a") else mod_b

        with (
            patch("rewards.cdmc.make", return_value=dummy_env),
            patch("rewards.importlib.import_module", side_effect=_fake_import),
        ):
            ctor = RewardFunctionConstructor(
                domain_name="quadruped",
                task_names=["task_a", "task_b"],
                seed=0,
                device=torch.device("cpu"),
            )

        dummy_env.set_state(np.array([0.25, 0.75, 0.0, 0.0]))
        call_out = ctor(dummy_env.physics)
        self.assertEqual(len(call_out), 2)
        self.assertIsInstance(call_out[0], float)
        self.assertIsInstance(call_out[1], float)
        self.assertEqual(call_out, [0.25, 0.75])

        batch_physics = np.array(
            [
                [0.1, 0.2, 0.0, 0.0],
                [0.3, 0.4, 0.0, 0.0],
                [0.5, 0.6, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        batch_actions = np.ones((3, 2), dtype=np.float64) * 0.5
        obs, rewards_dict = ctor.process_timesteps(batch_physics, batch_actions)

        self.assertEqual(obs.dtype, torch.float32)
        self.assertEqual(obs.shape, (3, 3))
        for task, offset in (("task_a", 0.0), ("task_b", 0.1)):
            self.assertEqual(rewards_dict[task].dtype, torch.float32)
            self.assertEqual(rewards_dict[task].shape, (3, 1))
            expected = torch.tensor(
                [[0.6 + offset], [0.8 + offset], [1.0 + offset]], dtype=torch.float32
            )
            torch.testing.assert_close(rewards_dict[task], expected)

    def test_quadruped_stand_jump_smoke(self):
        ctor = RewardFunctionConstructor(
            domain_name="quadruped",
            task_names=["stand", "jump"],
            seed=0,
            device=torch.device("cpu"),
        )
        ts = ctor._env.reset()
        physics_state = ctor._env.physics.get_state().copy()
        action = np.zeros(ctor._env.action_spec().shape, dtype=np.float32)
        batch_physics = np.stack([physics_state, physics_state], axis=0)
        batch_actions = np.stack([action, action], axis=0)

        obs, rewards_dict = ctor.process_timesteps(batch_physics, batch_actions)
        self.assertEqual(obs.dtype, torch.float32)
        self.assertEqual(obs.shape[0], 2)
        self.assertEqual(obs.shape[1], ts.observation["observations"].shape[0])
        for task in ("stand", "jump"):
            self.assertEqual(rewards_dict[task].dtype, torch.float32)
            self.assertEqual(rewards_dict[task].shape, (2, 1))
            self.assertTrue(torch.isfinite(rewards_dict[task]).all())


if __name__ == "__main__":
    unittest.main()
