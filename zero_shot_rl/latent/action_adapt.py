"""Action adaptation through the context-conditioned B_0 predictor of zero_shot_rl.latent.context_encoder.

For an action a of the policy at state s, adapt() looks for the action a' whose predicted next
feature under the current dynamics (context c1) matches the one of a under the stock dynamics
(context c0):

    a' = argmin_{a' in [-1, 1]^d_a} |g(s, a', c1) - g(s, a, c0)|^2

by `steps` Adam steps of rate `lr` from a' = a, clipping to [-1, 1] after every step. Each sample
is optimised independently (the loss is a sum over the batch and Adam acts per coordinate).
Nothing is trained: g and the contexts come from a trained context model.
"""

import torch


def adapt(model, s, a, c1, c0, steps, lr):
    """s [N, d_s], a [N, d_a], c1 and c0 [N or 1, d_c] -> a' [N, d_a], objective [N] at a and at a'."""
    with torch.no_grad():
        target = model.predict(s, a, c0)
    x = a.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    def objective():
        return ((model.predict(s, x, c1) - target) ** 2).sum(-1)

    with torch.no_grad():
        before = objective()
    for _ in range(steps):
        opt.zero_grad()
        objective().sum().backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)
    with torch.no_grad():
        after = objective()
    return x.detach(), before, after
