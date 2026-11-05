"""Rules that choose the task vector w of n runs from the rewards seen so far.

Rupf et al. (arXiv 2510.20264), section 4 and 5. Each rule returns w [n, d]; the policy acts
at the latent sqrt(d) w / |w| (zero_shot_rl.online.fb). The caller asks for a new w when the
rule's period starts and passes every transition to `observe`.

    Oracle   fixed w of offline task inference
    Random   w uniform on the sphere at every choice
    UCB      OpTI-BFM, eq. 11-12: among `candidates` uniform draws from the ellipsoid of
             radius 2 beta around the mean (footnote 6), the w that maximises
             min_heads psi(s, w)^T mean + beta ||psi(s, w)||_{V^-1}
    TS       OpTI-BFM-TS: w drawn from N(mean, V^-1)

UCB uses sigma = 1 and TS lam = 1, as in the paper's experiments (appendix F.1).
"""

import torch

from zero_shot_rl.online.belief import Belief


class Oracle:
    def __init__(self, n, w):
        self.w = torch.as_tensor(w).double().expand(n, -1)

    def choose(self, obs):
        return self.w

    def observe(self, phi, r):
        pass


class Random:
    def __init__(self, n, d, generator):
        self.n, self.d, self.generator = n, d, generator

    def choose(self, obs):
        return torch.randn(self.n, self.d, generator=self.generator, dtype=torch.float64)

    def observe(self, phi, r):
        pass


class UCB:
    def __init__(self, model, n, beta, lam=1.0, candidates=128, generator=None, device="cpu"):
        self.model, self.beta, self.candidates, self.generator = model, beta, candidates, generator
        self.belief = Belief(n, model.d, lam=lam, sigma=1.0, device=device)

    def scores(self, obs, w):
        """obs [n, d_s], w [n, m, d] -> UCB of each candidate, pessimistic over heads [n, m]."""
        sf = self.model.psi(obs, w).double()  # [n, m, heads, d]
        mean = self.belief.mean[:, None, None, :]
        ucb = (sf * mean).sum(-1) + self.beta * self.belief.cov_norm(sf)
        return ucb.min(dim=2).values

    def choose(self, obs):
        w = self.belief.sample_ellipsoid(self.candidates, 2 * self.beta, self.generator)
        best = self.scores(obs, w).argmax(dim=1)
        return w[torch.arange(len(w)), best]

    def observe(self, phi, r):
        self.belief.update(phi, r)


class TS:
    def __init__(self, model, n, sigma, lam=1.0, generator=None, device="cpu"):
        self.generator = generator
        self.belief = Belief(n, model.d, lam=lam, sigma=sigma, device=device)

    def choose(self, obs):
        return self.belief.sample(1, self.generator)[:, 0]

    def observe(self, phi, r):
        self.belief.update(phi, r)
