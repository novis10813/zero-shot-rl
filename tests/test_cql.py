"""CQL (on SAC) updates and acts with the squashed-normal actor of agents/base.py.

Run from the repo root: PYTHONPATH=. python -m unittest discover -s tests
"""

import unittest

import numpy as np
import torch

from agents.base import Batch
from agents.cql.agent import CQL


def make_agent(lagrange):
    return CQL(
        observation_length=3, action_length=2, device=torch.device("cpu"), name="cql",
        batch_size=8, discount=0.98, critic_hidden_dimension=16, critic_hidden_layers=2,
        critic_betas=[0.9, 0.999], critic_tau=0.005, critic_learning_rate=1e-4,
        critic_target_update_frequency=2, actor_hidden_dimension=16, actor_hidden_layers=2,
        actor_betas=[0.9, 0.999], actor_learning_rate=1e-4, actor_log_std_bounds=[-5, 2],
        alpha_learning_rate=1e-4, alpha_betas=[0.9, 0.999], actor_update_frequency=1,
        init_temperature=0.1, learnable_temperature=True, activation="relu",
        action_range=[-1.0, 1.0], normalisation_samples=None, cql_n_samples=3,
        cql_lagrange=lagrange, cql_alpha=0.01, cql_target_penalty=50.0,
    )


def make_batch(n=8):
    g = torch.Generator().manual_seed(0)
    return Batch(
        observations=torch.randn(n, 3, generator=g),
        actions=torch.rand(n, 2, generator=g) * 2 - 1,
        rewards=torch.rand(n, 1, generator=g),
        next_observations=torch.randn(n, 3, generator=g),
        not_dones=torch.ones(n, 1),
        discounts=torch.ones(n, 1),
    )


class CQLTests(unittest.TestCase):
    def test_update_and_act(self):
        for lagrange in (False, True):
            torch.manual_seed(0)
            agent = make_agent(lagrange)
            for step in range(3):
                metrics = agent.update(make_batch(), step=step)
            self.assertTrue(all(np.isfinite(float(v)) for v in metrics.values()))
            observation = np.zeros(3, dtype=np.float32)
            action = agent.act(observation, sample=False)
            self.assertEqual(action.shape, (2,))
            np.testing.assert_array_equal(action, agent.act(observation, sample=False))


if __name__ == "__main__":
    unittest.main()
