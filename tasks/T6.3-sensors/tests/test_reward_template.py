"""Tests for the Walker reward template fit."""

import unittest

import numpy as np
from dm_control.utils import rewards

from analyze.reward_template import DMC, PARAMS, fit, formula, r2, random_starts, template, tolerance


def sample_sensors(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0.1, 1.6, n), rng.uniform(-1, 1, n), rng.uniform(-3, 4, n), rng.uniform(-40, 60, n)], 1)


class TemplateTest(unittest.TestCase):
    def test_tolerance_matches_dm_control_gaussian(self):
        x = np.linspace(-1, 3, 101)
        expected = rewards.tolerance(x, bounds=(1.2, float("inf")), margin=0.6)
        np.testing.assert_allclose(tolerance(x, 1.2, np.inf, 0.6), expected, atol=1e-12)

    def test_stand_template_is_dmc_stand(self):
        s = sample_sensors(500)
        standing = rewards.tolerance(s[:, 0], bounds=(1.2, float("inf")), margin=0.6)
        expected = (3 * standing + (1 + s[:, 1]) / 2) / 4
        np.testing.assert_allclose(template(np.array(DMC["stand"], float), s), expected, atol=1e-12)

    def test_fit_recovers_parameters(self):
        s = sample_sensors(3000, seed=1)
        p_true = np.array(DMC["walk"], float)
        p_true[[3, 6]] = [1.0, 0.6]  # lower stand height and walk speed
        y = template(p_true, s)
        p = fit(s, y, random_starts("walk", 4, np.random.default_rng(0)))
        self.assertGreater(r2(y, template(p, s)), 0.999)
        self.assertAlmostEqual(p[PARAMS.index("h0")], 1.0, places=2)
        self.assertAlmostEqual(p[PARAMS.index("v_lo")], 0.6, places=2)

    def test_formula_mentions_active_terms(self):
        row = dict(zip(PARAMS, DMC["walk"]))
        text = formula(type("Row", (), row))
        self.assertIn("T(vx", text)
        self.assertNotIn("T(ω", text)


if __name__ == "__main__":
    unittest.main()
