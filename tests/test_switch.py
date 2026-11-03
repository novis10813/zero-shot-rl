"""Decision rule and latent selection of zero_shot_rl.latent.switch."""

import unittest

import numpy as np

from zero_shot_rl.latent.switch import choose_pess, latents


class ChoosePessTests(unittest.TestCase):
    def test_fixed_conditions(self):
        q = np.array([1.0, -1.0])
        self.assertFalse(choose_pess(q, q, q, "opt").any())
        self.assertTrue(choose_pess(q, q, q, "pess").all())

    def test_kappa(self):
        q_opt, u_opt, q_pess = np.array([3.0, 3.0, 3.0]), np.array([0.0, 1.0, 2.0]), np.full(3, 2.0)
        np.testing.assert_array_equal(choose_pess(q_opt, u_opt, q_pess, 0.0), [False] * 3)
        np.testing.assert_array_equal(choose_pess(q_opt, u_opt, q_pess, 1.0), [False, False, True])
        # ties keep a_opt
        np.testing.assert_array_equal(choose_pess(q_opt, u_opt, q_pess, 0.5), [False] * 3)


class LatentsTests(unittest.TestCase):
    def test_rows(self):
        zs = {
            "task": np.array(["walk", "walk", "walk", "walk", "run"]),
            "family": np.array(["z_r", "CQL", "CQL", "T05 CQL", "z_r"]),
            "target_angle": np.array([0.0, 5.0, 7.5, 7.5, 0.0]),
            "z": np.arange(5)[:, None] * np.ones((5, 2)),
        }
        z_r, z_cql = latents(zs, "walk", 7.5)
        np.testing.assert_array_equal(z_r, [0, 0])
        np.testing.assert_array_equal(z_cql, [2, 2])


if __name__ == "__main__":
    unittest.main()
