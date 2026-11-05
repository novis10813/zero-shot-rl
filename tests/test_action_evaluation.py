"""Episode runner of zero_shot_rl.env.walker, distances of zero_shot_rl.latent.behavior and
the seed pairing of zero_shot_rl.latent.action_difference, with stand-in agents and data."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from zero_shot_rl.env.walker import STEPS, act, make_env, run_episodes, set_physics
from zero_shot_rl.latent import action_difference
from zero_shot_rl.latent.behavior import Behavior
from zero_shot_rl.latent.task_z import TASKS


class ElementwiseAgent:
    """Actor mean tanh(obs[:6] * z): no matrix products, so batching cannot change a bit."""

    def actor(self, obs, z, std, sample=False):
        return torch.tanh(obs[:, :6] * z), None


class ConstantAgent:
    def __init__(self, value):
        self.value = value

    def actor(self, obs, z, std, sample=False):
        return torch.full((len(obs), 6), self.value), None


class RunEpisodesTests(unittest.TestCase):
    def setUp(self):
        self.agent = ElementwiseAgent()
        self.z = torch.linspace(-2.0, 2.0, 6)
        self.reward_fn = lambda physics: float(physics.named.data.qpos[0])

    def stepped(self, seed, stride):
        """The same episode, one `act` call per step."""
        env = make_env(seed)
        obs = env.reset().observation["observations"]
        observations, actions, rewards = [], [], []
        for t in range(STEPS):
            if t % stride == 0:
                observations.append(obs.copy())
            a = act(self.agent, obs, self.z)
            obs = env.step(a).observation["observations"]
            actions.append(a)
            rewards.append(self.reward_fn(env.physics))
        return np.stack(observations), np.stack(actions), np.array(rewards)

    def test_batched_episodes_match_stepped_episodes(self):
        run = run_episodes(self.agent, self.z, [make_env(0), make_env(1)], self.reward_fn, stride=10)
        self.assertEqual(run["observations"].shape[:2], (2, STEPS // 10))
        for i, seed in enumerate([0, 1]):
            observations, actions, rewards = self.stepped(seed, 10)
            np.testing.assert_array_equal(run["observations"][i], observations)
            np.testing.assert_array_equal(run["actions"][i], actions)
            np.testing.assert_array_equal(run["rewards"][i], rewards)
            self.assertEqual(run["return"][i], sum(rewards))

    def test_kept_physics_replays_the_next_step(self):
        env = make_env(0)
        run = run_episodes(self.agent, self.z, [env], stride=1, physics=True)
        self.assertNotIn("rewards", run)
        observations, physics, actions = run["observations"][0], run["physics"][0], run["actions"][0]
        for t in [0, 500, STEPS - 2]:
            set_physics(env, physics[t])
            obs = env.step(actions[t]).observation["observations"]
            np.testing.assert_allclose(obs, observations[t + 1], atol=1e-6)


class BehaviorTests(unittest.TestCase):
    def test_distances_by_hand(self):
        states = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        actions = torch.tensor([[0.0], [2.0], [-2.0]])
        behavior = Behavior(states, actions, lam=0.5, k=2)
        s, a = torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0]])

        sigma_s, sigma_a = states.std(0).numpy(), actions.std(0).numpy()
        state_term = (((states.numpy() - s.numpy()) / sigma_s) ** 2).sum(1) / 2
        action_term = (((actions.numpy() - a.numpy()) / sigma_a) ** 2).sum(1) / 1
        joint, conditional = behavior.distances(behavior.state_term(s), a)

        np.testing.assert_allclose(joint, [(state_term + 0.5 * action_term).min()], rtol=1e-6)
        nearest = np.argsort(state_term)[:2]
        np.testing.assert_allclose(conditional, [action_term[nearest].min()], rtol=1e-6)


class ActionDifferenceTests(unittest.TestCase):
    def test_pairs_by_position_within_label(self):
        z = torch.zeros(6)
        checkpoints = [
            ("A", 0, ConstantAgent(0.1), {t: z for t in TASKS}),
            ("A", 1, ConstantAgent(0.2), {t: z for t in TASKS}),
            ("B", 0, ConstantAgent(0.5), {t: z for t in TASKS}),
            ("B", 1, ConstantAgent(-0.9), {t: z for t in TASKS}),
        ]
        buffer = SimpleNamespace(storage={"observations": torch.zeros(10, 24)})
        joints = [f"j{i}" for i in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "difference.csv"
            argv = ["action_difference", "--checkpoints", "A", "a0", "a1", "--checkpoints", "B",
                    "b0", "b1", "--dataset_path", "unused", "--out", str(out), "--n_states", "3"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(action_difference.task_z, "inference_set",
                                      return_value=(buffer, None, None)), \
                    mock.patch.object(action_difference.task_z, "load_checkpoints",
                                      return_value=checkpoints), \
                    mock.patch.object(action_difference, "first_state", return_value=(None, joints)), \
                    mock.patch("builtins.print"):
                action_difference.main()
            df = pd.read_csv(out)
        self.assertEqual(len(df), len(TASKS) * 2 * 3 * 6)
        for pair, expected in [(0, 0.5 - 0.1), (1, -0.9 - 0.2)]:
            np.testing.assert_allclose(df[df["pair"] == pair]["delta"], expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
