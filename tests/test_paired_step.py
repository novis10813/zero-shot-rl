"""One-step replay of zero_shot_rl.eval.paired_step records observations, physics and rewards."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from rewards import RewardFunctionConstructor
from zero_shot_rl.env.walker import make_env
from zero_shot_rl.eval import paired_step


def rollout(steps, reward_fn, seed=0):
    env = make_env(seed)
    obs = [env.reset().observation["observations"]]
    states = [env.physics.get_state().copy()]
    rewards = []
    actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 6)).astype(np.float32)
    for a in actions:
        obs.append(env.step(a).observation["observations"])
        states.append(env.physics.get_state().copy())
        rewards.append(reward_fn(env.physics))
    return np.stack(obs), np.stack(states), actions, np.array(rewards, dtype=np.float64)


class PairedStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rc = RewardFunctionConstructor(
            domain_name="walker", task_names=["walk"], seed=0, device=torch.device("cpu")
        )
        cls.reward_fn = rc.reward_functions["walk"]
        cls.obs, cls.states, cls.actions, cls.rewards = rollout(10, cls.reward_fn)
        paired_step.G.clear()
        paired_step.G.update(
            physics=cls.states[:-1][None],
            actions=cls.actions[None],
            next_rollout=cls.obs[1:][None],
            reward_fn=cls.reward_fn,
        )

    def test_baseline_replay_reproduces_obs_physics_and_rewards(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "baseline.yaml"
            cfg.write_text(yaml.safe_dump({"base": {}}))
            name, nxt_obs, nxt_phys, nxt_rew = paired_step.replay(str(cfg))
        self.assertEqual(name, "baseline")
        self.assertEqual(nxt_obs.shape, (1, 10, 24))
        self.assertEqual(nxt_phys.shape, (1, 10, 18))
        self.assertEqual(nxt_rew.shape, (1, 10))
        np.testing.assert_allclose(nxt_obs[0], self.obs[1:], atol=1e-9)
        np.testing.assert_allclose(nxt_phys[0], self.states[1:], atol=1e-9)
        np.testing.assert_allclose(nxt_rew[0], self.rewards, atol=1e-9)

    def test_shifted_replay_differs_in_obs_and_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "shifted.yaml"
            cfg.write_text(yaml.safe_dump({"base": {"actuator_gear": [50.0] * 6}}))
            name, nxt_obs, nxt_phys, nxt_rew = paired_step.replay(str(cfg))
        self.assertEqual(name, "shifted")
        self.assertGreater(np.abs(nxt_obs[0] - self.obs[1:]).max(), 1e-3)
        self.assertGreater(np.abs(nxt_phys[0] - self.states[1:]).max(), 1e-3)
        self.assertEqual(nxt_rew.shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
