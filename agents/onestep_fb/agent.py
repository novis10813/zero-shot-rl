"""One-step forward-backward representations (Zheng et al., arXiv:2602.11399).

Ported from github.com/chongyi-zheng/onestep-fb (agents/onestep_fb.py). F(s, a) and
B(s, a) do not depend on z: the TD target uses the dataset's next action, so
F(s, a)^T z_r approximates (1 - discount) Q^beta_r. A z-conditioned actor maximises
F(s, pi(s, z))^T z.
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from agents.base import AbstractAgent


class OneStepFBMLP(torch.nn.Module):
    """MLP of the official code: Linear -> [LayerNorm] -> tanh, then Linear -> ReLU
    for the remaining hidden layers. Weights are Glorot-uniform, biases zero."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        output_dimension: Optional[int],
        layer_norm: bool,
        output_init_scale: float = 1.0,
    ):
        super().__init__()
        layers = [torch.nn.Linear(input_dimension, hidden_dimension)]
        if layer_norm:
            layers.append(torch.nn.LayerNorm(hidden_dimension, eps=1e-6))
        layers.append(torch.nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers += [torch.nn.Linear(hidden_dimension, hidden_dimension), torch.nn.ReLU()]
        self.trunk = torch.nn.Sequential(*layers)
        for module in self.trunk:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

        self.head = None
        if output_dimension is not None:
            self.head = torch.nn.Linear(hidden_dimension, output_dimension)
            # flax variance_scaling(scale, "fan_avg", "uniform") == Glorot gain sqrt(scale)
            torch.nn.init.xavier_uniform_(self.head.weight, gain=math.sqrt(output_init_scale))
            torch.nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        return x if self.head is None else self.head(x)


class OneStepFB(AbstractAgent):
    """One-step FB agent with a deterministic (constant-std Gaussian mode) actor."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        device: torch.device,
        name: str,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        forward_layer_norm: bool,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        backward_layer_norm: bool,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        actor_layer_norm: bool,
        actor_final_init_scale: float,
        learning_rate: float,
        batch_size: int,
        discount: float,
        tau: float,
        orthonormalisation_coefficient: float,
        z_mix_ratio: float,
        bc_coefficient: float,
        reward_temperature: float,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )
        obs_action = observation_length + action_length

        # two forward heads (official num_ensembles=2), aggregated by their mean
        self.forward_representations = torch.nn.ModuleList(
            [
                OneStepFBMLP(
                    obs_action,
                    forward_hidden_dimension,
                    forward_hidden_layers,
                    z_dimension,
                    forward_layer_norm,
                )
                for _ in range(2)
            ]
        )
        self.backward_representation = OneStepFBMLP(
            obs_action,
            backward_hidden_dimension,
            backward_hidden_layers,
            z_dimension,
            backward_layer_norm,
        )
        self.forward_representations_target = torch.nn.ModuleList(
            [
                OneStepFBMLP(
                    obs_action,
                    forward_hidden_dimension,
                    forward_hidden_layers,
                    z_dimension,
                    forward_layer_norm,
                )
                for _ in range(2)
            ]
        )
        self.backward_representation_target = OneStepFBMLP(
            obs_action,
            backward_hidden_dimension,
            backward_hidden_layers,
            z_dimension,
            backward_layer_norm,
        )
        self.forward_representations_target.load_state_dict(
            self.forward_representations.state_dict()
        )
        self.backward_representation_target.load_state_dict(
            self.backward_representation.state_dict()
        )
        for param in list(self.forward_representations_target.parameters()) + list(
            self.backward_representation_target.parameters()
        ):
            param.requires_grad_(False)

        self.actor = OneStepFBMLP(
            observation_length + z_dimension,
            actor_hidden_dimension,
            actor_hidden_layers,
            action_length,
            actor_layer_norm,
            output_init_scale=actor_final_init_scale,
        )

        self.to(device)

        # one Adam over all trained parameters, as in the official code
        self.optimizer = torch.optim.Adam(
            list(self.forward_representations.parameters())
            + list(self.backward_representation.parameters())
            + list(self.actor.parameters()),
            lr=learning_rate,
        )

        self._device = device
        self._z_dimension = z_dimension
        self.batch_size = batch_size
        self._discount = discount
        self._tau = tau
        self._orthonormalisation_coefficient = orthonormalisation_coefficient
        self._z_mix_ratio = z_mix_ratio
        self._bc_coefficient = bc_coefficient
        self._reward_temperature = reward_temperature

    def forward_representation(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """F(s, a) of both heads, shape [2, N, z_dimension]."""
        x = torch.cat([observation, action], dim=-1)
        return torch.stack([F(x) for F in self.forward_representations])

    def backward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """B(s, a), shape [N, z_dimension]."""
        return self.backward_representation(torch.cat([observation, action], dim=-1))

    def policy(self, observation: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Mean action of the actor clipped to [-1, 1] (eval temperature 0)."""
        return self.actor(torch.cat([observation, z], dim=-1)).clamp(-1.0, 1.0)

    def predict_q(
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Mean over heads of F(s, a)^T z, shape [N]."""
        return torch.einsum(
            "eid,id->ei", self.forward_representation(observation, action), z
        ).mean(0)

    def act(
        self,
        observation: np.ndarray,
        task: np.ndarray,
        step: Optional[int] = None,
        sample: bool = False,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Mean action for one observation. sample=True adds the actor's unit std."""
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            mean = self.actor(torch.cat([observation, z], dim=-1))
            action = mean + torch.randn_like(mean) if sample else mean

        return action.clamp(-1.0, 1.0).cpu().numpy()[0], None

    def sample_z(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Gaussian zs and permuted batch B(s, a), both scaled to norm sqrt(d), mixed
        with probability z_mix_ratio for the Gaussian."""
        size = observations.shape[0]
        scale = math.sqrt(self._z_dimension)
        gaussian = torch.randn((size, self._z_dimension), device=self._device)
        gaussian = scale * torch.nn.functional.normalize(gaussian, dim=1)
        with torch.no_grad():
            backward = self.backward(observations, actions)
        backward = backward[torch.randperm(size, device=self._device)]
        backward = scale * torch.nn.functional.normalize(backward, dim=1)
        use_gaussian = torch.rand((size, 1), device=self._device) < self._z_mix_ratio

        return torch.where(use_gaussian, gaussian, backward)

    def update(self, batch, step: int) -> Dict[str, float]:
        """One gradient step on the one-step FB loss (Eq. 7-9) and the actor loss."""
        observations = batch.observations
        actions = batch.actions
        size = observations.shape[0]
        identity = torch.eye(size, device=self._device)
        off_diagonal = 1 - identity

        # targets are moved towards the pre-update weights, as in the official code
        self.soft_update()

        with torch.no_grad():
            target_next_forward = torch.stack(
                [
                    F(torch.cat([batch.next_observations, batch.next_actions], dim=-1))
                    for F in self.forward_representations_target
                ]
            )
            target_backward = self.backward_representation_target(
                torch.cat([observations, actions], dim=-1)
            )
            target_measures = torch.einsum(
                "eid,kd->eik", target_next_forward, target_backward
            ).mean(0)

        forward = self.forward_representation(observations, actions)
        backward = self.backward(observations, actions)
        measures = torch.einsum("eid,kd->eik", forward, backward)

        td_error = (measures - self._discount * target_measures.unsqueeze(0)) * off_diagonal
        off_diag_loss = (0.5 * (td_error**2).sum(-1) / (size - 1)).mean()
        diag_loss = -(1 - self._discount) * torch.diagonal(measures, dim1=1, dim2=2).mean()
        repr_loss = off_diag_loss + diag_loss

        covariance = backward @ backward.T
        ortho_diag_loss = -torch.diagonal(covariance).mean()
        ortho_off_diag_loss = (
            0.5 * ((covariance * off_diagonal) ** 2).sum(-1) / (size - 1)
        ).mean()
        ortho_loss = ortho_diag_loss + ortho_off_diag_loss
        fb_loss = repr_loss + self._orthonormalisation_coefficient * ortho_loss

        # actor: F is held fixed, gradients reach only the actor through the action
        zs = self.sample_z(observations, actions)
        mean = self.actor(torch.cat([observations, zs], dim=-1))
        for param in self.forward_representations.parameters():
            param.requires_grad_(False)
        q = self.predict_q(observations, zs, mean.clamp(-1.0, 1.0))
        for param in self.forward_representations.parameters():
            param.requires_grad_(True)
        q_loss = -q.mean() / q.abs().mean().detach()
        # log N(actions; mean, I) up to a constant
        bc_loss = 0.5 * ((actions - mean) ** 2).sum(-1).mean()
        actor_loss = q_loss + self._bc_coefficient * bc_loss

        self.optimizer.zero_grad(set_to_none=True)
        (fb_loss + actor_loss).backward()
        self.optimizer.step()

        return {
            "train/fb_loss": fb_loss.item(),
            "train/repr_off_diag_loss": off_diag_loss.item(),
            "train/repr_diag_loss": diag_loss.item(),
            "train/ortho_loss": ortho_loss.item(),
            "train/succ_measure_mean": measures.mean().item(),
            "train/actor_loss": actor_loss.item(),
            "train/q_mean": q.mean().item(),
            "train/bc_mse": (2 * bc_loss / actions.shape[-1]).item(),
        }

    def soft_update(self) -> None:
        with torch.no_grad():
            for online, target in (
                (self.forward_representations, self.forward_representations_target),
                (self.backward_representation, self.backward_representation_target),
            ):
                for param, target_param in zip(online.parameters(), target.parameters()):
                    target_param.mul_(1 - self._tau).add_(self._tau * param)

    def infer_z(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> np.ndarray:
        """z = sqrt(d) normalize(E[w r B(s, a)]) with w = softmax(reward_temperature r)
        over the inference transitions (official infer_latent)."""
        with torch.no_grad():
            backward = self.backward(observations, actions)
            rewards = rewards.reshape(-1, 1)
            weights = torch.softmax(self._reward_temperature * rewards, dim=0)
            z = (weights * rewards * backward).mean(0, keepdim=True)
            z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)

        return z.squeeze(0).cpu().numpy()

    def load(self, filepath: Path):
        """Loads model."""
        pass
