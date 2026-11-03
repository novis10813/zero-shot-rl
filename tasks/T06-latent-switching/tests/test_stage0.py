"""Stage 0 analysis on synthetic probes, and the probe's rollout workers."""

from pathlib import Path
import unittest

import numpy as np

from analyze.stage0 import (FRACTIONS, interval_rows, paired, q_hat, signal_rows, signals,
                            switch_rows, switched_return)

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "artifacts/T04-mcfb-pessimism/zsrl-32/800000.pt"  # T05 seed 0 input


def synthetic_probe(n=50, seed=0):
    rng = np.random.default_rng(seed)
    probe = {key: [] for key in ("task", "source", "state", "q1", "q2", "p1", "p2", "c1",
                                 "c2", "nn", "G")}
    for task in ("stand", "walk"):
        for source in ("data", "policy", "pess", "visited", "visited_pess"):
            q1 = rng.normal(10, 2, n)
            values = {"q1": q1, "q2": q1 + rng.normal(0, 1, n), "p1": -rng.uniform(0, 3, n),
                      "p2": -rng.uniform(0, 3, n), "c1": rng.normal(50, 1, n),
                      "c2": rng.normal(50, 1, n), "nn": rng.uniform(0, 1, n),
                      "G": rng.normal(8, 2, n)}
            perm = rng.permutation(n)  # rows of a source need not be in state order
            probe["task"].append(np.full(n, task))
            probe["source"].append(np.full(n, source))
            probe["state"].append(perm)
            for key, value in values.items():
                probe[key].append(value)
    probe = {key: np.concatenate(value) for key, value in probe.items()}
    probe["latent_task"] = np.array(["stand", "walk"])
    probe["latent_r_mean"] = np.array([0.4, 0.2])
    return probe


class Stage0Tests(unittest.TestCase):
    def test_signals(self):
        probe = {"q1": np.array([3.0]), "q2": np.array([1.0]), "p1": np.array([-2.0]),
                 "p2": np.array([-4.0]), "c1": np.array([49.0]), "c2": np.array([53.0]),
                 "nn": np.array([0.5])}
        u = signals(probe)
        self.assertEqual(u["head_gap"][0], 2.0)
        self.assertEqual(u["penalty"][0], 3.0)
        self.assertAlmostEqual(u["const_dev"][0], 1.0)
        self.assertEqual(u["nn"][0], 0.5)

    def test_constant_correction(self):
        probe = synthetic_probe()
        probe["c1"] = probe["c2"] = np.full(len(probe["G"]), -50.0)
        shift = np.where(probe["task"] == "stand", 0.4, 0.2) * 100
        np.testing.assert_allclose(q_hat(probe), q_hat(probe, False) + shift)
        probe["c1"] = probe["c2"] = np.full(len(probe["G"]), 50.0)
        np.testing.assert_allclose(q_hat(probe), q_hat(probe, False))

    def test_head_gap_tracks_overestimation(self):
        probe = synthetic_probe()
        probe["c1"] = probe["c2"] = np.full(len(probe["G"]), 50.0)
        # make head 2 exceed G by the head gap: over = min(q1, q2) - G grows with |q1 - q2|
        probe["q2"] = probe["q1"] + np.abs(probe["q2"] - probe["q1"])
        probe["G"] = probe["q1"] - 3 * (probe["q2"] - probe["q1"])
        rows = signal_rows(0, probe)
        self.assertEqual(len(rows), 2 * 6)
        for row in rows:
            self.assertAlmostEqual(row["rho_head_gap"], 1.0)

    def test_interval(self):
        probe = synthetic_probe()
        q_corr = q_hat(probe)
        lower = q_corr + (probe["p1"] + probe["p2"]) / 2
        probe["G"] = (q_corr + lower) / 2
        for row in interval_rows(0, probe):
            self.assertEqual(row["frac_inside"], 1.0)
        probe["G"] = q_corr + 1
        for row in interval_rows(0, probe):
            self.assertEqual((row["frac_lower"], row["frac_upper"]), (1.0, 0.0))

    def test_pairs_follow_state(self):
        probe = synthetic_probe()
        i_opt, i_pess = paired(probe, "walk", "visited", "visited_pess")
        np.testing.assert_array_equal(probe["state"][i_opt], np.arange(50))
        np.testing.assert_array_equal(probe["state"][i_pess], np.arange(50))
        self.assertTrue((probe["source"][i_pess] == "visited_pess").all())

    def test_switched_return(self):
        g_opt, g_pess = np.array([1.0, 1.0, 1.0, 1.0]), np.array([5.0, 0.0, 0.0, 0.0])
        u = np.array([9.0, 1.0, 2.0, 3.0])
        self.assertEqual(switched_return(g_opt, g_pess, u, 0.0), 1.0)
        self.assertEqual(switched_return(g_opt, g_pess, u, 0.25), 2.0)
        self.assertEqual(switched_return(g_opt, g_pess, u, 1.0), 1.25)

    def test_switch_endpoints(self):
        probe = synthetic_probe()
        rows = switch_rows(0, probe)
        self.assertEqual(len(rows), 2 * 2 * len(FRACTIONS))
        for row in rows:
            for name in ("head_gap", "penalty", "const_dev", "nn", "oracle"):
                if row["fraction"] in (0.0, 1.0):
                    self.assertAlmostEqual(row[name], row["random"])
                self.assertLessEqual(row[name], row["oracle"] + 1e-12)


@unittest.skipUnless(CHECKPOINT.exists(), "needs the T05 seed 0 FB checkpoint")
class RolloutWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import probe
        cls.probe = probe
        probe._init_worker(str(CHECKPOINT))
        env = probe.make_env(0)
        env.reset()
        cls.physics = np.stack([env.physics.get_state().copy()] * 2)
        cls.z = np.ones(probe.W["agent"]._z_dimension, dtype=np.float32)

    def test_one_step_return_is_first_reward(self):
        p = self.probe
        actions = np.zeros((2, 6), dtype=np.float32)
        g = p._mc_returns(("stand", self.z, self.physics, actions, 1, 0))
        env = p.make_env(0)
        p.set_physics(env, self.physics[0])
        env.step(actions[0])
        self.assertAlmostEqual(g[0], p.W["rewards"]["stand"](env.physics))
        self.assertEqual(g[0], g[1])

    def test_returns_reproduce_with_seed(self):
        p = self.probe
        actions = np.zeros((2, 6), dtype=np.float32)
        job = ("walk", self.z, self.physics, actions, 20, 3)
        np.testing.assert_array_equal(p._mc_returns(job), p._mc_returns(job))
        self.assertNotEqual(*p._mc_returns(job))  # actor noise differs between envs

    def test_visited_states(self):
        states = self.probe._visited((self.z, 0, 980, 5))
        self.assertEqual(len(states), 4)
        self.assertEqual(states[0].shape, self.physics[0].shape)


if __name__ == "__main__":
    unittest.main()
