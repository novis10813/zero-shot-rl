"""Sampling, episode assignment and dynamics configs of zero_shot_rl.data.multi_dynamics."""

import collections
import unittest

import numpy as np

from zero_shot_rl.data.multi_dynamics import assignment, dynamics_config, nominal_friction, sample_sets
from zero_shot_rl.env.dynamics import make_shifted_env
from zero_shot_rl.env.walker import make_env

CONFIG = dict(seed=0, sets=30, friction=dict(range=[0.01, 5.0], test=[0.02, 0.1, 1.5, 3.0]),
              gear=dict(range=[0.5, 2.0], test=[0.62, 0.76, 1.3, 1.6]), margin=1.1,
              episodes=6, tasks=["stand", "walk", "run", "flip"], first_env_seed=100)


class MultiDynamicsTest(unittest.TestCase):
    def test_groups_ranges_and_excluded_test_values(self):
        friction, gear = sample_sets(CONFIG)
        nominal = nominal_friction()
        np.testing.assert_array_equal(gear[:10], 1.0)
        np.testing.assert_array_equal(friction[10:20], nominal)
        self.assertTrue((friction[[*range(10), *range(20, 30)]] != nominal).all())
        self.assertTrue((gear[10:] != 1.0).all())
        for values, key in ((friction, "friction"), (gear, "gear")):
            low, high = CONFIG[key]["range"]
            changed = values[values != (nominal if key == "friction" else 1.0)]
            self.assertTrue(((changed >= low) & (changed <= high)).all())
            for t in CONFIG[key]["test"]:
                self.assertTrue((np.abs(np.log(changed / t)) >= np.log(1.1)).all())

    def test_assignment_balances_seeds_and_tasks(self):
        episodes = assignment(CONFIG)
        self.assertEqual(len(episodes), 180)
        self.assertEqual(len({e[3] for e in episodes}), 180)
        counts = collections.Counter((e[1], e[2]) for e in episodes)
        self.assertEqual(set(counts.values()), {15})

    def test_nominal_set_is_the_stock_environment(self):
        shifted, stock = make_shifted_env(3, dynamics_config(nominal_friction(), np.ones(6))), make_env(3)
        a, b = shifted.reset().observation["observations"], stock.reset().observation["observations"]
        np.testing.assert_array_equal(a, b)
        action = np.full(6, 0.5)
        for _ in range(20):
            a = shifted.step(action).observation["observations"]
            b = stock.step(action).observation["observations"]
        np.testing.assert_array_equal(a, b)

    def test_listed_values_and_test_configs(self):
        import yaml
        from zero_shot_rl.env.dynamics import load_dynamics
        config = yaml.safe_load(open("tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml"))
        friction, gear = sample_sets(config)
        self.assertEqual(len(assignment(config)), 9 * 240)
        counts = collections.Counter((e[0], e[1], e[2]) for e in assignment(config))
        self.assertEqual(set(counts.values()), {20})
        roots = {"friction_mild": "tasks/T3.2-ood-robustness/configs", "actuator_strong": "tasks/T3.2-ood-robustness/configs",
                 "friction_high_strong": "tasks/T4.1.2-dynamic-shift-detection/configs",
                 "actuator_high_mild": "tasks/T4.1.2-dynamic-shift-detection/configs"}
        for i, v in enumerate(config["values"]):
            if v["name"] not in roots:
                continue
            base = load_dynamics(f"{roots[v['name']]}/{v['name']}.yaml")["base"]
            listed = dynamics_config(friction[i], np.full(6, gear[i]))["base"]
            if "geom_friction" in base:
                self.assertAlmostEqual(base["geom_friction"]["all"], listed["geom_friction"]["all"])
            if "actuator_gear" in base:
                np.testing.assert_allclose(list(base["actuator_gear"].values()), listed["actuator_gear"])

    def test_gear_is_a_multiplier_of_the_xml_gear(self):
        gear = dynamics_config(0.7, 0.5)["base"]["actuator_gear"]
        self.assertEqual(gear, [50.0, 25.0, 10.0, 50.0, 25.0, 10.0])


if __name__ == "__main__":
    unittest.main()
