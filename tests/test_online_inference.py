"""Component tests for zero_shot_rl.online (OpTI-BFM belief and selectors)."""

import unittest

import torch

from zero_shot_rl.eval.online_inference import parse_rule
from zero_shot_rl.online.belief import Belief
from zero_shot_rl.online.fb import FBModel
from zero_shot_rl.online.selectors import TS, UCB


def observations(n, steps, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(steps, n, d, generator=g, dtype=torch.float64), g


class FakeModel:
    """psi(s, w) = w / |w|, so the best w for a task vector is the task vector itself."""

    d = 5

    def psi(self, obs, w):
        return torch.nn.functional.normalize(w, dim=-1).unsqueeze(2)


class TestBelief(unittest.TestCase):
    def setUp(self):
        self.n, self.d, self.lam, self.sigma = 3, 4, 0.5, 0.3
        self.belief = Belief(self.n, self.d, lam=self.lam, sigma=self.sigma)
        self.phi, g = observations(self.n, 20, self.d)
        self.r = torch.randn(20, self.n, generator=g, dtype=torch.float64)
        for phi, r in zip(self.phi, self.r):
            self.belief.update(phi, r)

    def precision(self, i):
        X = self.phi[:, i] / self.sigma
        return self.lam * torch.eye(self.d, dtype=torch.float64) + X.T @ X

    def test_mean_is_ridge_solution(self):
        for i in range(self.n):
            X, y = self.phi[:, i] / self.sigma, self.r[:, i] / self.sigma
            expected = torch.linalg.solve(self.precision(i), X.T @ y)
            torch.testing.assert_close(self.belief.mean[i], expected)

    def test_cov_norm(self):
        x = torch.randn(self.n, 7, self.d, dtype=torch.float64)
        got = self.belief.cov_norm(x)
        for i in range(self.n):
            inv = torch.linalg.inv(self.precision(i))
            expected = torch.einsum("md,de,me->m", x[i], inv, x[i]).sqrt()
            torch.testing.assert_close(got[i], expected)

    def test_transform_has_posterior_covariance(self):
        eye = torch.eye(self.d, dtype=torch.float64).expand(self.n, -1, -1)
        root = self.belief.transform(eye) - self.belief.mean.unsqueeze(1)  # rows L^{-T} e_j
        for i in range(self.n):
            torch.testing.assert_close(root[i].T @ root[i], torch.linalg.inv(self.precision(i)))

    def test_ellipsoid_samples_stay_inside(self):
        z = self.belief.sample_ellipsoid(500, 2.0, torch.Generator().manual_seed(1))
        diff = z - self.belief.mean.unsqueeze(1)
        dist = torch.einsum("nmd,nde,nme->nm", diff, self.belief.V, diff).sqrt()
        self.assertLessEqual(dist.max().item(), 2.0 + 1e-9)
        self.assertGreater(dist.max().item(), 1.8)


class TestSelectors(unittest.TestCase):
    def converge(self, selector, w_true, steps=300):
        phis, _ = observations(len(w_true), steps, FakeModel.d, seed=2)
        for phi in phis:
            selector.observe(phi, (phi * w_true).sum(1))
        return selector.choose(None)

    def test_ucb_finds_task_vector(self):
        w_true = torch.randn(4, FakeModel.d, dtype=torch.float64)
        ucb = UCB(FakeModel(), 4, beta=0.1, generator=torch.Generator().manual_seed(0))
        cos = torch.nn.functional.cosine_similarity(self.converge(ucb, w_true), w_true, dim=1)
        self.assertGreater(cos.min().item(), 0.99)

    def test_ts_samples_concentrate(self):
        w_true = torch.randn(4, FakeModel.d, dtype=torch.float64)
        ts = TS(FakeModel(), 4, sigma=0.01, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(self.converge(ts, w_true), w_true, atol=5e-3, rtol=0)

    def test_lstsq_recovers_task_vector(self):
        phi = torch.randn(100, 6, dtype=torch.float64)
        w = torch.randn(6, dtype=torch.float64)
        torch.testing.assert_close(FBModel.lstsq(None, phi, phi @ w), w)


class TestRules(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_rule("ucb:0.1@1000"), ("ucb", 0.1, 1000))
        self.assertEqual(parse_rule("ts:0.01"), ("ts", 0.01, 1))
        method, param, period = parse_rule("oracle")
        self.assertEqual((method, period), ("oracle", 1))
        for bad in ("ucb", "random:1", "lola:1"):
            with self.assertRaises(ValueError):
                parse_rule(bad)


if __name__ == "__main__":
    unittest.main()
