"""phi, psi and the policy of a standard FB checkpoint in the form OpTI-BFM uses.

Rupf et al. (arXiv 2510.20264) appendix D.1: F(s, a, z) are the successor features of

    phi(s) = Cov_D(B)^{-1} B(s),   Cov_D(B) = E_{s ~ D}[B(s) B(s)^T]

over the training states D. A task vector w (reward r(s) = phi(s)^T w) selects the policy at
the latent sqrt(d) w / |w|, the sphere FB is trained on; psi(s, w) = F(s, pi(s, w), w) with
the deterministic action of zero_shot_rl.env.walker.policy, one row per F head.
"""

import math

import torch

from zero_shot_rl.env.walker import policy


class FBModel:
    def __init__(self, agent, observations):
        """observations [N, d_s]: the training states that define Cov_D(B)."""
        self.agent = agent
        with torch.no_grad():
            B = agent.FB.backward_representation(observations).double()
        self.cov_inv = torch.linalg.inv(B.T @ B / len(B))
        self.d = B.shape[1]

    def unit(self, w):
        return math.sqrt(self.d) * torch.nn.functional.normalize(w, dim=-1).float()

    @torch.no_grad()
    def phi(self, obs):
        """[n, d_s] -> [n, d], float64."""
        return self.agent.FB.backward_representation(obs).double() @ self.cov_inv

    @torch.no_grad()
    def policy(self, obs, w):
        """obs [n, d_s], w [n, d] -> deterministic action [n, d_a]."""
        return policy(self.agent, obs, self.unit(w))

    @torch.no_grad()
    def psi(self, obs, w):
        """obs [n, d_s], w [n, m, d] -> psi(s, w) [n, m, heads, d]."""
        n, m, d = w.shape
        o = obs.unsqueeze(1).expand(n, m, obs.shape[1]).reshape(n * m, -1)
        z = self.unit(w.reshape(n * m, d))
        heads = self.agent.FB.forward_representation(
            observation=o, action=policy(self.agent, o, z), z=z
        )
        return torch.stack(heads, 1).reshape(n, m, len(heads), d)

    def lstsq(self, phi, r):
        """Least-squares task vector of eq. 4: phi [N, d], r [N] -> [d]."""
        return torch.linalg.lstsq(phi.double().cpu(), r.double().cpu().unsqueeze(1)).solution[:, 0]
