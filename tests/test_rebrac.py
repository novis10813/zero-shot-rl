"""ReBRAC buffer alignment, critic target and E2E term, checkpoint round trip.

Run from the repo root: PYTHONPATH=. python -m unittest discover -s tests
"""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from agents.base import OfflineReplayBuffer, load_agent
from agents.rebrac.agent import ReBRAC
from agents.rebrac.replay_buffer import ReBRACBatch, ReBRACReplayBuffer


def make_episode(length, offset):
    values = np.arange(length + 1, dtype=np.float32) + offset
    return {
        "observation": np.stack([values, values + 0.5], axis=1),
        "action": values[:, None],
        "reward": np.zeros((length + 1, 1), dtype=np.float32),
        "discount": np.ones((length + 1, 1), dtype=np.float32),
        "physics": values[:, None],
    }


def make_agent(**kwargs):
    defaults = dict(
        observation_length=3, action_length=2, device=torch.device("cpu"), name="rebrac",
        hidden_dimension=16, hidden_layers=3, ensemble_size=2, actor_learning_rate=1e-3,
        critic_learning_rate=1e-3, batch_size=8, discount=0.98, tau=0.005,
        policy_noise=0.0, noise_clip=0.5, policy_freq=2, actor_bc_coef=0.03,
        critic_bc_coef=0.1,
    )
    defaults.update(kwargs)
    return ReBRAC(**defaults)


def make_batch(n=8, has_next=1.0):
    g = torch.Generator().manual_seed(0)
    return ReBRACBatch(
        observations=torch.randn(n, 3, generator=g),
        actions=torch.rand(n, 2, generator=g) * 2 - 1,
        rewards=torch.rand(n, 1, generator=g),
        next_observations=torch.randn(n, 3, generator=g),
        next_actions=torch.rand(n, 2, generator=g) * 2 - 1,
        has_next_actions=torch.full((n, 1), has_next),
    )


class ReBRACBufferTests(unittest.TestCase):
    def test_next_actions_follow_subsampling_and_episode_boundaries(self):
        episodes = [make_episode(3, 0), make_episode(4, 10), make_episode(1, 20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.npz"
            np.savez(path, **{f"e{i}": np.array(e, dtype=object) for i, e in enumerate(episodes)})
            with mock.patch.object(
                OfflineReplayBuffer, "_relabel_episode", staticmethod(lambda rc, episode, task: episode)
            ):
                buffer = ReBRACReplayBuffer(
                    reward_constructor=None, dataset_path=path, discount=0.98,
                    device=torch.device("cpu"), task="walk", transitions=6,
                )
        actions = buffer.storage["actions"][:, 0].numpy()
        next_actions = buffer.storage["next_actions"][:, 0].numpy()
        has_next = buffer.storage["has_next_actions"][:, 0].numpy()
        # action values are offset + t, so the next action is the stored one plus 1,
        # except at each episode's last action (3, 14, 21)
        last = np.isin(actions, [3.0, 14.0, 21.0])
        np.testing.assert_array_equal(has_next, (~last).astype(np.float32))
        np.testing.assert_array_equal(next_actions[~last], actions[~last] + 1)
        batch = buffer.sample(4)
        self.assertEqual(batch.has_next_actions.shape, (4, 1))


class ReBRACAgentTests(unittest.TestCase):
    def test_critic_penalty_is_dropped_without_next_action(self):
        torch.manual_seed(0)
        agent = make_agent(critic_bc_coef=100.0)
        batch = make_batch(has_next=0.0)
        far = ReBRACBatch(**{**batch.__dict__, "next_actions": batch.next_actions + 5})
        torch.manual_seed(1)
        loss_a = make_agent(critic_bc_coef=100.0)._update_critic(batch)["train/td_loss"]
        torch.manual_seed(1)
        loss_b = make_agent(critic_bc_coef=100.0)._update_critic(far)["train/td_loss"]
        self.assertAlmostEqual(loss_a, loss_b, places=6)
        self.assertIsNotNone(agent)

    def test_e2e_term_raises_q_on_data(self):
        batch = make_batch()
        qs = []
        for beta in (0.0, 10.0):
            torch.manual_seed(0)
            agent = make_agent(e2e_beta_dq=beta)
            for step in range(20):
                agent._update_critic(batch)
            with torch.no_grad():
                qs.append(agent.critic(batch.observations, batch.actions).mean().item())
        self.assertGreater(qs[1], qs[0])

    def test_update_and_checkpoint_round_trip(self):
        torch.manual_seed(0)
        agent = make_agent(e2e_beta_dq=0.001)
        metrics = agent.update(make_batch(), step=0)
        self.assertIn("train/actor_loss", metrics)
        self.assertNotIn("train/actor_loss", agent.update(make_batch(), step=1))
        observation = np.zeros(3, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            agent._name = 7
            loaded = load_agent(agent.save(Path(tmp)), torch.device("cpu"))
        self.assertEqual(loaded.e2e_beta_dq, 0.001)
        np.testing.assert_array_equal(loaded.act(observation), agent.act(observation))


if __name__ == "__main__":
    unittest.main()
