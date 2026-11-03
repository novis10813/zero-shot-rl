"""ExORL CQL port: networks, critic target without entropy term, updates, checkpoint round trip.

Run from the repo root: PYTHONPATH=. python -m unittest discover -s tests
"""

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from agents.base import Batch, load_agent
from agents.cql_exorl.agent import CQLExORL


def make_agent(**kwargs):
    defaults = dict(
        observation_length=3, action_length=2, device=torch.device("cpu"), name="cql-exorl",
        hidden_dimension=16, learning_rate=1e-3, batch_size=8, tau=0.01, n_samples=3,
        alpha=0.01,
    )
    defaults.update(kwargs)
    return CQLExORL(**defaults)


def make_batch(batch_size=8, observation_length=3, action_length=2):
    return Batch(
        observations=torch.randn(batch_size, observation_length),
        actions=torch.rand(batch_size, action_length) * 2 - 1,
        rewards=torch.rand(batch_size, 1),
        next_observations=torch.randn(batch_size, observation_length),
        future_observations=None,
        discounts=torch.full((batch_size, 1), 0.98),
        not_dones=torch.ones(batch_size, 1),
        physics=None,
        future_goals=None,
    )


class NetworkTest(unittest.TestCase):
    def test_trunk_matches_exorl(self):
        agent = make_agent()
        for trunk in (agent.actor.policy, agent.critic.q1_net, agent.critic.q2_net):
            kinds = [type(layer) for layer in trunk]
            self.assertEqual(
                kinds,
                [torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Tanh, torch.nn.Linear,
                 torch.nn.ReLU, torch.nn.Linear],
            )
            for layer in trunk:
                if isinstance(layer, torch.nn.Linear):
                    self.assertTrue(torch.all(layer.bias == 0))

    def test_critic_target_starts_equal_and_moves_by_tau(self):
        agent = make_agent()
        before = copy.deepcopy(agent.critic_target.state_dict())
        agent.update(make_batch(), step=0)
        for key, target in agent.critic_target.state_dict().items():
            expected = before[key] + 0.01 * (agent.critic.state_dict()[key] - before[key])
            torch.testing.assert_close(target, expected)

    def test_act_returns_mean_in_action_range(self):
        agent = make_agent()
        action = agent.act(torch.randn(3).numpy())
        self.assertEqual(action.shape, (2,))
        self.assertTrue(((action > -1) & (action < 1)).all())


class CriticTargetTest(unittest.TestCase):
    def test_td_loss_has_no_entropy_term(self):
        agent = make_agent()
        batch = make_batch()
        critic = copy.deepcopy(agent.critic)

        torch.manual_seed(0)
        metrics = agent._update_critic(batch)

        torch.manual_seed(0)
        with torch.no_grad():
            next_actions = agent.actor(batch.next_observations).sample()
            target_q1, target_q2 = agent.critic_target(batch.next_observations, next_actions)
            target_q = batch.rewards + batch.discounts * torch.min(target_q1, target_q2)
            q1, q2 = critic(batch.observations, batch.actions)
            expected = (
                torch.nn.functional.mse_loss(q1, target_q)
                + torch.nn.functional.mse_loss(q2, target_q)
            )
        self.assertAlmostEqual(metrics["train/td_loss"], expected.item(), places=5)


class TrainingTest(unittest.TestCase):
    def test_updates_are_finite_and_temperature_starts_at_one(self):
        agent = make_agent()
        self.assertAlmostEqual(agent.log_actor_alpha.exp().item(), 1.0)
        for step in range(5):
            metrics = agent.update(make_batch(), step=step)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_checkpoint_round_trip(self):
        agent = make_agent()
        agent.update(make_batch(), step=0)
        agent._name = 7
        with tempfile.TemporaryDirectory() as directory:
            path = agent.save(Path(directory))
            loaded = load_agent(path, torch.device("cpu"))
        self.assertIsInstance(loaded, CQLExORL)
        for key, value in agent.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
