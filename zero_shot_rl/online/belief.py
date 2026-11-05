"""Regularised least squares over task vectors, for n independent runs at once.

Rupf et al., "Optimistic Task Inference for Behavior Foundation Models" (arXiv 2510.20264),
eq. 7-9 and the Thompson sampling variant of section 4. After observations (phi_i, r_i),

    V   = lam I + sum_i phi_i phi_i^T / sigma^2
    eta = sum_i phi_i r_i / sigma^2,    mean = V^{-1} eta

which is the posterior N(mean, V^{-1}) of Bayesian linear regression with prior N(0, I / lam)
and noise variance sigma^2. With sigma = 1, {z : ||z - mean||_V <= beta} is the confidence
ellipsoid of OpTI-BFM.

Samples use z = mean + L^{-T} xi with V = L L^T, whose covariance is V^{-1}. The authors' code
(github.com/ThomasRupf/opti-bfm, utils/blr.py) maps xi with R^{-T} for V = R^T R, whose
covariance is (R R^T)^{-1}; this module follows the paper.
"""

import torch


class Belief:
    def __init__(self, n, d, lam=1.0, sigma=1.0, device="cpu"):
        self.sigma = sigma
        eye = torch.eye(d, dtype=torch.float64, device=device)
        self.V = (lam * eye).expand(n, d, d).clone()
        self.eta = torch.zeros(n, d, dtype=torch.float64, device=device)
        self._L = None

    def update(self, phi, r):
        """phi [n, d], r [n]: one observation per run."""
        x = phi.to(self.V) / self.sigma
        self.V += x.unsqueeze(2) * x.unsqueeze(1)
        self.eta += x * (r.to(self.V) / self.sigma).unsqueeze(1)
        self._L = None

    @property
    def L(self):
        if self._L is None:
            self._L = torch.linalg.cholesky(self.V)
        return self._L

    @property
    def mean(self):
        return torch.cholesky_solve(self.eta.unsqueeze(2), self.L).squeeze(2)

    def cov_norm(self, x):
        """x [n, ..., d] -> sqrt(x^T V^{-1} x) [n, ...]."""
        n, d = x.shape[0], x.shape[-1]
        flat = x.to(self.V).reshape(n, -1, d).transpose(1, 2)  # [n, d, m]
        y = torch.linalg.solve_triangular(self.L, flat, upper=False)
        return y.norm(dim=1).reshape(x.shape[:-1])

    def transform(self, xi):
        """xi [n, m, d] -> mean + L^{-T} xi [n, m, d]."""
        y = torch.linalg.solve_triangular(
            self.L.transpose(1, 2), xi.to(self.V).transpose(1, 2), upper=True
        )
        return self.mean.unsqueeze(1) + y.transpose(1, 2)

    def sample(self, m, generator=None):
        """m draws per run from N(mean, V^{-1}) [n, m, d]."""
        n, d = self.eta.shape
        xi = torch.randn(n, m, d, generator=generator, dtype=self.V.dtype, device="cpu")
        return self.transform(xi.to(self.V.device))

    def sample_ellipsoid(self, m, radius, generator=None):
        """m uniform draws per run from {z : ||z - mean||_V <= radius} [n, m, d]."""
        n, d = self.eta.shape
        g = torch.randn(n, m, d, generator=generator, dtype=self.V.dtype, device="cpu")
        u = torch.rand(n, m, 1, generator=generator, dtype=self.V.dtype, device="cpu")
        xi = g / g.norm(dim=2, keepdim=True) * u ** (1.0 / d) * radius
        return self.transform(xi.to(self.V.device))
