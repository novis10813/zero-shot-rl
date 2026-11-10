"""Online value correction of a frozen FB checkpoint at a fixed latent z.

F, B, the actor and z stay frozen. A critic Q_phi(s, a) is trained online from the task reward
of one environment, and the action is chosen by Q_adapt among candidates around the FB action:

    direct     Q_adapt(s, a) = Q_phi(s, a)
    residual   Q_adapt(s, a) = scale * min_heads F(s, a, z)^T z + Q_phi(s, a)

`scale` = |E_D[r B(s)]| / sqrt(d) (`reward_scale`) puts F^T z in reward units, because z is the
task vector E_D[r B(s)] normalised to the sphere of radius sqrt(d). Q_phi is two critics; the
last layer of the residual critics starts at zero, so Q_adapt starts as the FB value.

Candidates of a state: a_0 = pi(s, z), the deterministic action of
zero_shot_rl.env.walker.policy, and a_i = clip(a_0 + sigma eps_i, -1, 1) for i = 1..n-1. The
greedy action maximises min_critics Q_adapt over the candidates. Both critics regress

    y = r + discount * max_i min_critics Q_adapt_target(s', a'_i)

with fresh candidates a'_i at s'. Episodes have no terminal state, so y has no done mask.
"""

import copy
import math

import torch

from zero_shot_rl.env.walker import policy

VARIANTS = ("direct", "residual")


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
        self.next_observations = torch.zeros(capacity, observation_length, device=device)
        self.size = 0

    def add(self, observation, action, reward, next_observation):
        i = self.size
        self.observations[i], self.actions[i] = observation, action
        self.rewards[i], self.next_observations[i] = reward, next_observation
        self.size += 1

    def sample(self, batch_size, generator):
        i = torch.randint(self.size, (batch_size,), generator=generator).to(self.rewards.device)
        return self.observations[i], self.actions[i], self.rewards[i], self.next_observations[i]


class QAdapt:
    def __init__(self, agent, z, scale, variant, observation_length, action_length, *,
                 candidates=32, sigma=0.2, hidden=1024, layers=2, learning_rate=1e-4,
                 tau=0.01, discount=0.98):
        """z [d] is the latent the actor and F are conditioned on."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.agent, self.z, self.scale, self.variant = agent, z, scale, variant
        self.candidates, self.sigma, self.tau, self.discount = candidates, sigma, tau, discount
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
        heads = self.agent.FB.forward_representation(observation=obs, action=actions, z=z)
        return self.scale * torch.stack([(F * z).sum(-1) for F in heads]).min(0).values

    def q(self, obs, actions, target=False):
        """[m, d_s], [m, d_a] -> Q_adapt of each critic [2, m]."""
        x = torch.cat([obs, actions], dim=-1)
        q = torch.stack([c(x).squeeze(-1) for c in (self.targets if target else self.critics)])
        return q + self.fb_value(obs, actions) if self.variant == "residual" else q

    @torch.no_grad()
    def propose(self, obs, generator):
        """[b, d_s] -> candidates [b, n, d_a]; candidate 0 is the FB action."""
        a0 = policy(self.agent, obs, self.z.expand(len(obs), -1))
        noise = torch.randn(len(obs), self.candidates - 1, a0.shape[1], generator=generator)
        noisy = (a0.unsqueeze(1) + self.sigma * noise.to(a0.device)).clamp(-1.0, 1.0)
        return torch.cat([a0.unsqueeze(1), noisy], dim=1)

    @torch.no_grad()
    def values(self, obs, candidates, target=False):
        """[b, d_s], [b, n, d_a] -> min_critics Q_adapt [b, n]."""
        b, n, _ = candidates.shape
        o = obs.unsqueeze(1).expand(b, n, obs.shape[1]).reshape(b * n, -1)
        return self.q(o, candidates.reshape(b * n, -1), target).min(0).values.reshape(b, n)

    @torch.no_grad()
    def act(self, obs, generator, explore=0.0):
        """[b, d_s] -> the action [b, d_a], its candidate index [b] and its value [b]. The
        action is the greedy candidate, or with probability `explore` a uniform one."""
        candidates = self.propose(obs, generator)
        values = self.values(obs, candidates)
        index = values.argmax(1)
        if explore > 0:
            uniform = torch.randint(self.candidates, (len(obs),), generator=generator)
            random = torch.rand(len(obs), generator=generator) < explore
            index = torch.where(random.to(index.device), uniform.to(index.device), index)
        rows = torch.arange(len(obs), device=index.device)
        return candidates[rows, index], index, values[rows, index]

    def update(self, batch, generator):
        """One gradient step of both critics on (s, a, r, s'); returns the TD loss."""
        obs, actions, rewards, next_obs = batch
        target = self.values(next_obs, self.propose(next_obs, generator), target=True)
        y = rewards + self.discount * target.max(1).values
        loss = ((self.q(obs, actions) - y) ** 2).mean(1).sum()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for p, p_target in zip(self.critics.parameters(), self.targets.parameters()):
                p_target.lerp_(p, self.tau)
        return loss.item()

    def save(self, path):
        """The critics and what is needed to rebuild Q_adapt on the same FB checkpoint."""
        torch.save({"variant": self.variant, "scale": self.scale, "z": self.z.cpu(),
                    "candidates": self.candidates, "sigma": self.sigma,
                    "critics": self.critics.state_dict()}, path)
