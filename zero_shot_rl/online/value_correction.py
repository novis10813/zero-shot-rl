"""Online value correction of a frozen FB checkpoint at a fixed latent z.

F, B, the actor and z stay frozen. A critic Q_phi(s, a) is trained online from the task reward
of one environment, and the action is chosen by Q_adapt among candidates around the FB action:

    direct     Q_adapt(s, a) = Q_phi(s, a)
    residual   Q_adapt(s, a) = scale * min_heads F(s, a, z)^T z + Q_phi(s, a)
    search     Q_adapt(s, a) = scale * min_heads F(s, a, z)^T z, no critic and no training

`scale` = |E_D[r B(s)]| / sqrt(d) (`reward_scale`) puts F^T z in reward units, because z is the
task vector E_D[r B(s)] normalised to the sphere of radius sqrt(d). Q_phi is two critics; the
last layer of the residual critics starts at zero, so Q_adapt starts as the FB value.

Candidates of a state: a_0 = pi(s, z), the deterministic action of
zero_shot_rl.env.walker.policy, and a_i = clip(a_0 + delta_i, -1, 1) for i = 1..n-1, where delta_i = sigma eps_i and, if
`radius` is set, delta_i is scaled down to |delta_i|_2 <= radius (radius 0 keeps only a_0). The
greedy action maximises min_critics Q_adapt over the candidates. Both critics regress

    y = r + discount * max_i min_critics Q_adapt_target(s', a'_i)

with fresh candidates a'_i at s'. Episodes have no terminal state, so y has no done mask.

The residual variant follows the discrete residual soft Q-learning of Li et al. 2023 (Residual
Q-Learning, arXiv:2306.09526, eqs. 6 and 8) over the candidate set, with the FB actor as the
prior policy pi(a|s) = N(a_0, prior_sigma^2) and the observed task reward as the add-on reward.
With score(s, a) = min_critics Q_adapt(s, a) + omega log pi(a|s), the policy is
softmax_i(score(s, a_i) / alpha), the critics regress

    y = r + discount * alpha * log sum_i exp(score_target(s', a'_i) / alpha)

and the executed action in training is a sample of that policy (the test action is the argmax
of the score). omega is omega' = omega alpha of the paper and alpha its temperature. The
log-sum-exp is over the candidates, so y holds the constant alpha log n more than a mean over
them. The direct variant keeps the greedy rule and the max target.
"""

import copy
import math

import torch

from zero_shot_rl.env.walker import policy

VARIANTS = ("direct", "residual", "search")
PRIOR_SIGMA = 0.2  # the std of the FB actor in zero_shot_rl.env.walker.policy


def reward_scale(agent, observations, rewards):
    """|E[r B(s)]| / sqrt(d) over the task inference transitions of FB.infer_z."""
    with torch.no_grad():
        B = agent.FB.backward_representation(observations)
    z = rewards.T @ B / len(rewards)
    return (z.norm() / math.sqrt(B.shape[1])).item()


def mlp(inputs, hidden, layers):
    modules, width = [], inputs
    for _ in range(layers):
        modules += [torch.nn.Linear(width, hidden), torch.nn.ReLU()]
        width = hidden
    return torch.nn.Sequential(*modules, torch.nn.Linear(width, 1))


class Buffer:
    """Transitions (s, a, r, s') of one run, sampled uniformly with replacement."""

    def __init__(self, capacity, observation_length, action_length, device):
        self.observations = torch.zeros(capacity, observation_length, device=device)
        self.actions = torch.zeros(capacity, action_length, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_observations = torch.zeros(
            capacity, observation_length, device=device
        )
        self.size = 0

    def add(self, observation, action, reward, next_observation):
        i = self.size
        self.observations[i], self.actions[i] = observation, action
        self.rewards[i], self.next_observations[i] = reward, next_observation
        self.size += 1

    def sample(self, batch_size, generator):
        i = torch.randint(self.size, (batch_size,), generator=generator).to(
            self.rewards.device
        )
        return (
            self.observations[i],
            self.actions[i],
            self.rewards[i],
            self.next_observations[i],
        )


class QAdapt:
    def __init__(
        self,
        agent,
        z,
        scale,
        variant,
        observation_length,
        action_length,
        *,
        candidates=32,
        sigma=0.2,
        radius=None,
        hidden=64,
        layers=2,
        learning_rate=1e-4,
        tau=0.01,
        discount=0.98,
        omega=1.0,
        alpha=1.0,
    ):
        """z [d] is the latent the actor and F are conditioned on. omega and alpha are the
        weight of the prior policy and the temperature of the residual variant."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.agent, self.z, self.scale, self.variant = agent, z, scale, variant
        self.candidates, self.sigma, self.radius = candidates, sigma, radius
        self.tau, self.discount = tau, discount
        self.omega, self.alpha = omega, alpha
        self.hidden, self.layers = hidden, layers
        self.critics = torch.nn.ModuleList(
            mlp(observation_length + action_length, hidden, layers) for _ in range(2)
        ).to(z.device)
        if variant == "residual":
            for critic in self.critics:
                torch.nn.init.zeros_(critic[-1].weight)
                torch.nn.init.zeros_(critic[-1].bias)
        self.targets = copy.deepcopy(self.critics).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critics.parameters(), lr=learning_rate)

    @torch.no_grad()
    def fb_value(self, obs, actions):
        """[m, d_s], [m, d_a] -> scale * min_heads F(s, a, z)^T z [m]."""
        z = self.z.expand(len(obs), -1)
        heads = self.agent.FB.forward_representation(
            observation=obs, action=actions, z=z
        )
        return self.scale * torch.stack([(F * z).sum(-1) for F in heads]).min(0).values

    def q(self, obs, actions, target=False):
        """[m, d_s], [m, d_a] -> Q_adapt of each critic [2, m]."""
        x = torch.cat([obs, actions], dim=-1)
        q = torch.stack(
            [c(x).squeeze(-1) for c in (self.targets if target else self.critics)]
        )
        if self.variant == "search":
            return self.fb_value(obs, actions).expand(2, -1)
        return q + self.fb_value(obs, actions) if self.variant == "residual" else q

    @torch.no_grad()
    def propose(self, obs, generator):
        """[b, d_s] -> candidates [b, n, d_a]; candidate 0 is the FB action."""
        a0 = policy(self.agent, obs, self.z.expand(len(obs), -1))
        noise = torch.randn(
            len(obs), self.candidates - 1, a0.shape[1], generator=generator
        )
        delta = self.sigma * noise.to(a0.device)
        if self.radius is not None:
            norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            delta = delta * (self.radius / norm.clamp_min(1e-8)).clamp(max=1.0)
        actions = torch.clamp(a0[:, None, :] + delta, -1.0, 1.0)
        return torch.cat([a0[:, None, :], actions], dim=1)

    @torch.no_grad()
    def values(self, obs, candidates, target=False):
        """[b, d_s], [b, n, d_a] -> min_critics Q_adapt [b, n]."""
        b, n, _ = candidates.shape
        o = obs.unsqueeze(1).expand(b, n, obs.shape[1]).reshape(b * n, -1)
        return (
            self.q(o, candidates.reshape(b * n, -1), target).min(0).values.reshape(b, n)
        )

    def log_prior(self, candidates):
        """[b, n, d_a] -> log pi(a|s) of the candidates [b, n], pi = N(a_0, PRIOR_SIGMA^2)."""
        prior = torch.distributions.Normal(candidates[:, :1], PRIOR_SIGMA)
        return prior.log_prob(candidates).sum(-1)

    @torch.no_grad()
    def scores(self, obs, candidates, target=False):
        """[b, d_s], [b, n, d_a] -> what the rule ranks the candidates by [b, n]: Q_adapt, and
        for the residual variant also omega log pi(a|s)."""
        values = self.values(obs, candidates, target)
        if self.variant == "residual":
            values = values + self.omega * self.log_prior(candidates)
        return values

    @torch.no_grad()
    def act(self, obs, generator, explore=0.0, sample=False):
        """[b, d_s] -> the action [b, d_a] and a dict of [b] tensors: its candidate `index`,
        its `value` Q_adapt, `shift` = |a - a_0|, `gain` = Q_adapt(s, a) - Q_adapt(s, a_0) and
        `at_fb` = 1 if the action is a_0 (index 0) else 0.
        The action is the candidate of the highest score, or with probability `explore` a
        uniform one. With `sample` the residual variant draws it from softmax(score / alpha)."""
        candidates = self.propose(obs, generator)
        values = self.values(obs, candidates)
        scores = self.scores(obs, candidates)
        index = scores.argmax(1)
        if sample and self.variant == "residual":
            probs = torch.softmax(scores / self.alpha, dim=1).cpu()
            index = torch.multinomial(probs, 1, generator=generator)[:, 0].to(
                index.device
            )
        elif explore > 0:
            uniform = torch.randint(self.candidates, (len(obs),), generator=generator)
            random = torch.rand(len(obs), generator=generator) < explore
            index = torch.where(
                random.to(index.device), uniform.to(index.device), index
            )
        rows = torch.arange(len(obs), device=index.device)
        action = candidates[rows, index]
        info = {
            "index": index,
            "value": values[rows, index],
            "shift": (action - candidates[:, 0]).norm(dim=1),
            "gain": values[rows, index] - values[:, 0],
            "at_fb": (index == 0).float(),
        }
        return action, info

    @torch.no_grad()
    def next_value(self, next_obs, generator):
        """[b, d_s] -> the value of the next state in the target y [b]: the max over fresh
        candidates of the target critics, or for the residual variant their soft maximum
        alpha log sum_i exp(score_i / alpha)."""
        candidates = self.propose(next_obs, generator)
        if self.variant == "residual":
            scores = self.scores(next_obs, candidates, target=True)
            return self.alpha * torch.logsumexp(scores / self.alpha, dim=1)
        return self.values(next_obs, candidates, target=True).max(1).values

    def update(self, batch, generator):
        """One gradient step of both critics on (s, a, r, s'); returns the TD loss."""
        obs, actions, rewards, next_obs = batch
        y = rewards + self.discount * self.next_value(next_obs, generator)
        loss = ((self.q(obs, actions) - y) ** 2).mean(1).sum()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for p, p_target in zip(
                self.critics.parameters(), self.targets.parameters()
            ):
                p_target.lerp_(p, self.tau)
        return loss.item()

    def save(self, path):
        """The critics and what is needed to rebuild Q_adapt on the same FB checkpoint."""
        torch.save(
            {
                "variant": self.variant,
                "scale": self.scale,
                "z": self.z.cpu(),
                "candidates": self.candidates,
                "sigma": self.sigma,
                "radius": self.radius,
                "hidden": self.hidden,
                "layers": self.layers,
                "omega": self.omega,
                "alpha": self.alpha,
                "critics": self.critics.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, agent, observation_length, action_length):
        """The QAdapt saved at `path`, on `agent` (the FB checkpoint it was trained on)."""
        state = torch.load(path, map_location="cpu")
        model = cls(
            agent,
            state["z"].to(next(agent.parameters()).device),
            state["scale"],
            state["variant"],
            observation_length,
            action_length,
            candidates=state["candidates"],
            sigma=state["sigma"],
            radius=state["radius"],
            hidden=state["hidden"],
            layers=state["layers"],
            omega=state.get("omega", 1.0),
            alpha=state.get("alpha", 1.0),
        )
        model.critics.load_state_dict(state["critics"])
        return model
