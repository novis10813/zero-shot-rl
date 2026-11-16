"""Open- and closed-loop replay of zero_shot_rl.eval.paired_horizon on a short stock rollout."""

import unittest

import numpy as np

from zero_shot_rl.env.dynamics import parse_dynamics
from zero_shot_rl.env.walker import make_env
from zero_shot_rl.eval.paired_horizon import closed_loop, open_loop


def rollout(steps, seed=0):
    env = make_env(seed)
    obs = [env.reset().observation["observations"]]
    states, actions = [], np.random.default_rng(seed).uniform(-1, 1, (steps, 6))
    for a in actions:
        states.append(env.physics.get_state().copy())
        obs.append(env.step(a).observation["observations"])
    return np.stack(states), actions, np.stack(obs)


class PairedHorizonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states, cls.actions, cls.obs = rollout(40)

    def test_stock_open_loop_reproduces_rollout(self):
        out = open_loop(make_env(0), self.states[10], self.actions[10:30])
        np.testing.assert_allclose(out, self.obs[11:31], atol=1e-9)

    def test_closed_loop_with_recorded_actions_is_open_loop(self):
        recorded = iter(self.actions[5:20])
        obs, actions = closed_loop(make_env(0), self.states[5], self.obs[5], lambda o: next(recorded), 15)
        np.testing.assert_array_equal(actions, self.actions[5:20])
        np.testing.assert_allclose(obs, self.obs[6:21], atol=1e-9)

    def test_closed_loop_sees_current_observation(self):
        seen = []

        def policy(o):
            seen.append(o.copy())
            return np.zeros(6)
        obs, _ = closed_loop(make_env(0), self.states[0], self.obs[0], policy, 3)
        np.testing.assert_array_equal(seen[0], self.obs[0])
        np.testing.assert_array_equal(seen[1:], obs[:2])

    def test_shifted_dynamics_diverge(self):
        env = make_env(0, parse_dynamics({"base": {"actuator_gear": [50] * 6}}))
        out = open_loop(env, self.states[10], self.actions[10:20])
        self.assertGreater(np.abs(out - self.obs[11:21]).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
