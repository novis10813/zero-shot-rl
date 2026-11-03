"""CQL as implemented in the ExORL benchmark (denisyarats/exorl, agent/cql.py).

Jeen et al. (2024) report CQL with "the same implementation and hyperparameters as is used
on the ExORL benchmark". agents/cql differs from that implementation (no LayerNorm, entropy
term in the critic target, other temperature and target updates), so this module ports it
line by line:
    networks  Linear -> LayerNorm -> Tanh -> Linear -> ReLU -> Linear, orthogonal init
    actor     mu = tanh(mu), std = exp(clamp(log_std, -10, 2)), SquashedNormal(mu, std)
    critic    y = r + discount * min_m Q'_m(s', a'), a' ~ pi(s'), no entropy term
              L_Q = sum_m E_B[(Q_m(s, a) - y)^2] + alpha * (E_B[logsumexp_j Q_m(s, a_j)]
              summed over m - E_B[Q_1(s, a) + Q_2(s, a)]), a_j over n_samples uniform
              actions, pi(s) samples, pi(s') samples and the dataset action
    actor     L_pi = E_B[temperature * log pi(a~|s) - min_m Q_m(s, a~)], a~ ~ pi(s), with the
              per-dimension log_prob of ExORL (not summed over action dimensions)
    temperature  learned towards target entropy -action_length, initial value 1
Critic, actor and the Polyak target update (tau) run on every step, in that order. The
discount of a transition is the dataset's episode discount times gamma, as in ExORL.
"""

from typing import Dict

import numpy as np
import torch

from agents.base import AbstractAgent, Batch
from agents.utils import SquashedNormal


def _weight_init(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def _trunk(input_dimension: int, output_dimension: int, hidden_dimension: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dimension, hidden_dimension),
        torch.nn.LayerNorm(hidden_dimension),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dimension, hidden_dimension),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_dimension, output_dimension),
    )


class ExORLActor(torch.nn.Module):
    def __init__(self, observation_length, action_length, hidden_dimension):
        super().__init__()
        self.policy = _trunk(observation_length, 2 * action_length, hidden_dimension)
        self.apply(_weight_init)

    def forward(self, observations: torch.Tensor) -> SquashedNormal:
        mu, log_std = self.policy(observations).chunk(2, dim=-1)
        mu = torch.tanh(mu)
        std = log_std.clamp(-10, 2).exp()
        return SquashedNormal(mu, std)


class ExORLCritic(torch.nn.Module):
    def __init__(self, observation_length, action_length, hidden_dimension):
        super().__init__()
        self.q1_net = _trunk(observation_length + action_length, 1, hidden_dimension)
        self.q2_net = _trunk(observation_length + action_length, 1, hidden_dimension)
        self.apply(_weight_init)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([observations, actions], dim=-1)
        return self.q1_net(x), self.q2_net(x)


class CQLExORL(AbstractAgent):
    def __init__(
        self,
        observation_length: int,
        action_length: int,
        device: torch.device,
        name: str,
        hidden_dimension: int,
        learning_rate: float,
        batch_size: int,
        tau: float,
        n_samples: int,
        alpha: float,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )
        self.device = device
        self.batch_size = batch_size
        self.tau = tau
        self.n_samples = n_samples
        self.alpha = alpha
        self.target_entropy = -action_length

        self.actor = ExORLActor(observation_length, action_length, hidden_dimension).to(device)
        self.critic = ExORLCritic(observation_length, action_length, hidden_dimension).to(
            device
        )
        self.critic_target = ExORLCritic(
            observation_length, action_length, hidden_dimension
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_actor_alpha = torch.nn.Parameter(torch.zeros(1, device=device))

        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.actor_alpha_optimiser = torch.optim.Adam(
            [self.log_actor_alpha], lr=learning_rate
        )

    @torch.no_grad()
    def act(self, observation: np.ndarray, step: int = None, sample: bool = False) -> np.ndarray:
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        dist = self.actor(observation)
        action = dist.sample() if sample else dist.mean
        return action.cpu().numpy()[0]

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        metrics = self._update_critic(batch)
        metrics.update(self._update_actor(batch))
        self._soft_update()
        return metrics

    def _repeated_critic_apply(self, observations: torch.Tensor, actions: torch.Tensor):
        """observations [batch, obs], actions [n_samples, batch, act] -> two [n, batch, 1]."""
        n_samples, batch_size = actions.shape[:2]
        repeated = observations.unsqueeze(0).expand(n_samples, -1, -1)
        q1, q2 = self.critic(
            repeated.reshape(n_samples * batch_size, -1),
            actions.reshape(n_samples * batch_size, -1),
        )
        return q1.reshape(n_samples, batch_size, 1), q2.reshape(n_samples, batch_size, 1)

    def _update_critic(self, batch: Batch) -> Dict[str, float]:
        observations, actions = batch.observations, batch.actions
        next_observations = batch.next_observations

        with torch.no_grad():
            next_actions = self.actor(next_observations).sample()
            target_q1, target_q2 = self.critic_target(next_observations, next_actions)
            target_q = batch.rewards + batch.discounts * torch.min(target_q1, target_q2)

        q1, q2 = self.critic(observations, actions)
        td_loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(
            q2, target_q
        )

        with torch.no_grad():
            random_actions = torch.empty(
                (self.n_samples, q1.shape[0], actions.shape[-1]), device=self.device
            ).uniform_(-1, 1)
            sampled_actions = self.actor(observations).sample((self.n_samples,))
            next_sampled_actions = self.actor(next_observations).sample((self.n_samples,))

        rand_q1, rand_q2 = self._repeated_critic_apply(observations, random_actions)
        sampled_q1, sampled_q2 = self._repeated_critic_apply(observations, sampled_actions)
        next_sampled_q1, next_sampled_q2 = self._repeated_critic_apply(
            observations, next_sampled_actions
        )
        cat_q1 = torch.cat([rand_q1, sampled_q1, next_sampled_q1, q1.unsqueeze(0)], dim=0)
        cat_q2 = torch.cat([rand_q2, sampled_q2, next_sampled_q2, q2.unsqueeze(0)], dim=0)
        cql_logsumexp = (
            torch.logsumexp(cat_q1, dim=0).mean() + torch.logsumexp(cat_q2, dim=0).mean()
        )
        cql_penalty = cql_logsumexp - (q1 + q2).mean()
        critic_loss = td_loss + self.alpha * cql_penalty

        self.critic_optimiser.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimiser.step()

        return {
            "train/critic_loss": critic_loss.item(),
            "train/td_loss": td_loss.item(),
            "train/Q1": q1.mean().item(),
            "train/cql_penalty": cql_penalty.item(),
        }

    def _update_actor(self, batch: Batch) -> Dict[str, float]:
        dist = self.actor(batch.observations)
        actions = dist.rsample()
        log_pi = dist.log_prob(actions)

        alpha_loss = -(self.log_actor_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.actor_alpha_optimiser.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.actor_alpha_optimiser.step()
        temperature = self.log_actor_alpha.exp().detach()

        q1, q2 = self.critic(batch.observations, actions)
        actor_loss = (temperature * log_pi - torch.min(q1, q2)).mean()
        self.actor_optimiser.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimiser.step()

        return {
            "train/actor_loss": actor_loss.item(),
            "train/actor_entropy": -log_pi.mean().item(),
            "train/temperature": temperature.item(),
        }

    def _soft_update(self) -> None:
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.lerp_(p, self.tau)

    def load(self, filepath):
        raise NotImplementedError("use agents.base.load_agent")
