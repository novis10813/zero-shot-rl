"""ReBRAC (Tarasov et al., 2023, arXiv 2305.09836) and ReBRAC-E2E, single-task offline RL.

Follows the CORL ReBRAC implementation:
    critic  y = r + gamma * (min_m Q'_m(s', a~') - beta2 * ||a~' - a'_D||^2),
            a~' = clip(pi'(s') + clip(N(0, noise), -noise_clip, noise_clip), -1, 1)
            L_Q = sum_m E_B[(Q_m(s, a) - y)^2]
    actor   L_pi = E_B[beta1 * ||pi(s) - a||^2 - min_m Q_m(s, pi(s)) / E_B|min_m Q_m(s, pi(s))|],
            updated with the soft target updates every policy_freq steps
Critic hidden layers are Linear -> ReLU -> LayerNorm, actor hidden layers Linear -> ReLU,
actor output tanh.

ReBRAC-E2E (End-to-end Offline Distribution RL, the practical method, eq. 10) subtracts
beta_dq * E_B[mean_m Q_m(s, a)] from L_Q. e2e_beta_dq = 0 is plain ReBRAC.

DMC episodes end by time limit only, so no transition is terminal. a'_D is missing for an
episode's final transition, whose beta2 term is dropped (has_next_actions = 0).
"""

from typing import Dict

import numpy as np
import torch

from agents.base import AbstractAgent
from agents.rebrac.replay_buffer import ReBRACBatch


def _mlp(input_dimension, output_dimension, hidden_dimension, hidden_layers, layernorm,
         last_layer_scale):
    layers = []
    width = input_dimension
    for _ in range(hidden_layers):
        layers += [torch.nn.Linear(width, hidden_dimension), torch.nn.ReLU()]
        if layernorm:
            layers.append(torch.nn.LayerNorm(hidden_dimension))
        width = hidden_dimension
    last = torch.nn.Linear(width, output_dimension)
    torch.nn.init.uniform_(last.weight, -last_layer_scale, last_layer_scale)
    torch.nn.init.uniform_(last.bias, -last_layer_scale, last_layer_scale)
    return torch.nn.Sequential(*layers, last)


class ReBRACActor(torch.nn.Module):
    def __init__(self, observation_length, action_length, hidden_dimension, hidden_layers):
        super().__init__()
        self.trunk = _mlp(observation_length, action_length, hidden_dimension, hidden_layers,
                          layernorm=False, last_layer_scale=1e-3)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.trunk(observations))


class ReBRACCritic(torch.nn.Module):
    """Ensemble of independent Q networks; forward returns [ensemble, batch, 1]."""

    def __init__(self, observation_length, action_length, hidden_dimension, hidden_layers,
                 ensemble_size):
        super().__init__()
        self.qs = torch.nn.ModuleList(
            _mlp(observation_length + action_length, 1, hidden_dimension, hidden_layers,
                 layernorm=True, last_layer_scale=3e-3)
            for _ in range(ensemble_size)
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observations, actions], dim=-1)
        return torch.stack([q(x) for q in self.qs])


class ReBRAC(AbstractAgent):
    def __init__(
        self,
        observation_length: int,
        action_length: int,
        device: torch.device,
        name: str,
        hidden_dimension: int,
        hidden_layers: int,
        ensemble_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        batch_size: int,
        discount: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_freq: int,
        actor_bc_coef: float,
        critic_bc_coef: float,
        e2e_beta_dq: float = 0.0,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )
        self.device = device
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.actor_bc_coef = actor_bc_coef
        self.critic_bc_coef = critic_bc_coef
        self.e2e_beta_dq = e2e_beta_dq

        self.actor = ReBRACActor(
            observation_length, action_length, hidden_dimension, hidden_layers
        ).to(device)
        self.actor_target = ReBRACActor(
            observation_length, action_length, hidden_dimension, hidden_layers
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = ReBRACCritic(
            observation_length, action_length, hidden_dimension, hidden_layers, ensemble_size
        ).to(device)
        self.critic_target = ReBRACCritic(
            observation_length, action_length, hidden_dimension, hidden_layers, ensemble_size
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimiser = torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )

    @torch.no_grad()
    def act(self, observation: np.ndarray, step: int = None, sample: bool = False) -> np.ndarray:
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return self.actor(observation).cpu().numpy()[0]

    def update(self, batch: ReBRACBatch, step: int) -> Dict[str, float]:
        metrics = self._update_critic(batch)
        if step % self.policy_freq == 0:
            metrics.update(self._update_actor(batch))
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        return metrics

    def _update_critic(self, batch: ReBRACBatch) -> Dict[str, float]:
        with torch.no_grad():
            noise = (torch.randn_like(batch.actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(batch.next_observations) + noise).clamp(-1, 1)
            bc_penalty = ((next_actions - batch.next_actions) ** 2).sum(-1, keepdim=True)
            next_q = self.critic_target(batch.next_observations, next_actions).min(0).values
            next_q = next_q - self.critic_bc_coef * batch.has_next_actions * bc_penalty
            target_q = batch.rewards + self.discount * next_q

        q = self.critic(batch.observations, batch.actions)
        td_loss = ((q - target_q.unsqueeze(0)) ** 2).mean(dim=(1, 2)).sum()
        e2e_term = q.mean()
        critic_loss = td_loss - self.e2e_beta_dq * e2e_term

        self.critic_optimiser.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimiser.step()

        return {
            "train/critic_loss": critic_loss.item(),
            "train/td_loss": td_loss.item(),
            "train/Q": e2e_term.item(),
            "train/critic_bc_penalty": bc_penalty.mean().item(),
        }

    def _update_actor(self, batch: ReBRACBatch) -> Dict[str, float]:
        actions = self.actor(batch.observations)
        bc_penalty = ((actions - batch.actions) ** 2).sum(-1, keepdim=True)
        q = self.critic(batch.observations, actions).min(0).values
        lmbda = 1.0 / q.abs().mean().detach()
        actor_loss = (self.actor_bc_coef * bc_penalty - lmbda * q).mean()

        self.actor_optimiser.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimiser.step()

        return {
            "train/actor_loss": actor_loss.item(),
            "train/actor_bc_penalty": bc_penalty.mean().item(),
        }

    def load(self, filepath):
        raise NotImplementedError("use agents.base.load_agent")

    def _soft_update(self, network: torch.nn.Module, target: torch.nn.Module) -> None:
        with torch.no_grad():
            for p, p_target in zip(network.parameters(), target.parameters()):
                p_target.lerp_(p, self.tau)
