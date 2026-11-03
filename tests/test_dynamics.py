"""Dynamics config of zero_shot_rl.env.dynamics: defaults, resets, events and validation."""

import unittest

import numpy as np

from zero_shot_rl.env.walker import make_env

STEPS = 60
ACTIONS = np.random.default_rng(0).uniform(-1, 1, (STEPS, 6))


def rollout(env, steps=STEPS):
    obs = [env.reset().observation["observations"]]
    for a in ACTIONS[:steps]:
        obs.append(env.step(a).observation["observations"])
    return np.stack(obs)


class DynamicsTests(unittest.TestCase):
    def test_empty_config_matches_stock_env(self):
        np.testing.assert_array_equal(rollout(make_env(3, {})), rollout(make_env(3)))

    def test_fields_change_trajectory(self):
        stock = rollout(make_env(3))
        for base in (
            {"geom_friction": 0.1},
            {"geom_friction": {"all": [0.1, 0.1, 0.1]}},
            {"actuator_gear": {"right_hip": 30}},
            {"dof_damping": {"left_knee": 2.0}},
            {"opt.gravity": [0, 0, -5]},
            {"body_mass": {"torso": 20.0}},
        ):
            with self.subTest(base=base):
                self.assertFalse(np.array_equal(rollout(make_env(3, {"base": base})), stock))

    def test_reset_restores_base(self):
        config = {"events": [{"step": 30, "set": {"actuator_gear": 10, "body_mass": {"torso": 20.0}}}]}
        env, stock = make_env(3, config), make_env(3)
        rollout(env), rollout(stock)
        np.testing.assert_array_equal(env.physics.model.actuator_gear[:, 0], [10] * 6)
        # the second episode starts from base again and draws the same initial state as stock
        np.testing.assert_array_equal(rollout(env)[:31], rollout(stock)[:31])
        env.reset()
        np.testing.assert_array_equal(env.physics.model.actuator_gear[:, 0], stock.physics.model.actuator_gear[:, 0])
        np.testing.assert_array_equal(env.physics.model.body_mass, stock.physics.model.body_mass)
        np.testing.assert_array_equal(env.physics.model.body_inertia, stock.physics.model.body_inertia)

    def test_event_applies_at_its_step(self):
        k = 20
        shifted = rollout(make_env(3, {"events": [{"step": k, "set": {"actuator_gear": 20}}]}))
        stock = rollout(make_env(3))
        # observation k is returned by step k-1, which still used the base dynamics
        np.testing.assert_array_equal(shifted[: k + 1], stock[: k + 1])
        self.assertFalse(np.array_equal(shifted[k + 1], stock[k + 1]))

    def test_events_stack(self):
        env = make_env(3, {"events": [
            {"step": 10, "set": {"actuator_gear": {"right_hip": 1}}},
            {"step": 2, "set": {"actuator_gear": {"left_hip": 2}}}]})
        rollout(env, 11)
        np.testing.assert_array_equal(env.physics.model.actuator_gear[[0, 3], 0], [1, 2])

    def test_mass_scales_inertia_and_keeps_state(self):
        env, stock = make_env(3, {"events": [{"step": 5, "set": {"body_mass": {"torso": 20.0}}}]}), make_env(3)
        for e in (env, stock):
            e.reset()
            for a in ACTIONS[:6]:
                e.step(a)
        model, nominal = env.physics.model, stock.physics.model
        i = 1  # torso; body 0 is the world
        np.testing.assert_allclose(model.body_inertia[i], nominal.body_inertia[i] * 20.0 / nominal.body_mass[i])
        self.assertEqual(model.body_mass[i], 20.0)
        # the constants refresh must not reset the episode state: one step after the event
        # stays within one step of motion of the unshifted episode
        self.assertLess(np.abs(env.physics.data.qpos - stock.physics.data.qpos).max(), 0.05)

    def test_invalid_configs_raise(self):
        for bad in (
            {"bases": {}},
            {"base": {"friction": 1}},
            {"base": {"actuator_gear": {"nope": 1}}},
            {"base": {"actuator_gear": [1, 2]}},
            {"base": {"opt.gravity": 9.8}},
            {"base": {"body_mass": {"torso": -1}}},
            {"events": [{"step": -1, "set": {}}]},
            {"events": [{"step": 1}]},
            {"events": [{"step": 1, "set": {"dof_damping": {"nope": 1}}}]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                make_env(0, bad)


if __name__ == "__main__":
    unittest.main()
