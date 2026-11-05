"""
Distance of actions to the behaviour in the training dataset (s_i, a_i):

    joint        min_i [ |s - s_i|^2 / (d_s) + lam |a - a_i|^2 / (d_a) ]
    conditional  min over the k dataset states nearest to s of |a - a_i|^2 / d_a

Every coordinate is standardised by its dataset standard deviation and each squared norm is
divided by the dimension count, so one unit is one standard deviation per coordinate.
"""

import torch

METRICS = ["joint", "conditional"]


def add_arguments(parser):
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=50)


class Behavior:
    """Dataset (s_i, a_i) with the standardisation and the two distances."""

    def __init__(self, states, actions, lam, k):
        self.states, self.actions, self.lam, self.k = states, actions, lam, k
        self.sigma_s = states.std(0).clamp(min=1e-6)
        self.sigma_a = actions.std(0).clamp(min=1e-6)

    @classmethod
    def from_buffer(cls, buffer, lam, k, device):
        return cls(
            buffer.storage["observations"].float().to(device),
            buffer.storage["actions"].float().to(device),
            lam,
            k,
        )

    def state_term(self, s):
        """[Q, N]: mean squared standardised state difference to every dataset state."""
        d = torch.cdist(s / self.sigma_s, self.states / self.sigma_s) ** 2
        return d / s.shape[1]

    def action_term(self, a):
        d = torch.cdist(a / self.sigma_a, self.actions / self.sigma_a) ** 2
        return d / a.shape[1]

    def distances(self, state_term, a):
        """joint and conditional distance of actions `a` [Q, d_a] at the states of `state_term`."""
        action_term = self.action_term(a)
        joint = (state_term + self.lam * action_term).min(1).values
        nearest = state_term.topk(self.k, dim=1, largest=False).indices
        conditional = action_term.gather(1, nearest).min(1).values
        return joint.cpu().numpy(), conditional.cpu().numpy()
