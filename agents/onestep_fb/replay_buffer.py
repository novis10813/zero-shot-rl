"""Replay buffer for one-step FB: adds the dataset's next action to each transition."""

import dataclasses
from typing import Dict, Optional, Tuple

import torch

from agents.base import Batch
from agents.fb.inference_buffer import FBInferenceReplayBuffer


@dataclasses.dataclass
class OneStepBatch(Batch):
    next_actions: Optional[torch.Tensor] = None


class OneStepFBReplayBuffer(FBInferenceReplayBuffer):
    """
    FB buffer whose transitions also carry the next dataset action. Transition t of an
    episode is (obs[t], action[t + 1], obs[t + 1]), so its next action is action[t + 2];
    the last transition of each episode has none and is never sampled for training.
    """

    def sample(self, batch_size: int) -> OneStepBatch:
        """Samples transitions that have a next dataset action."""
        batch_indices = self.valid_indices[
            torch.randint(0, len(self.valid_indices), (batch_size,), device=self.device)
        ]

        return OneStepBatch(
            observations=self.storage["observations"][batch_indices],
            actions=self.storage["actions"][batch_indices],
            rewards=self.storage["rewards"][batch_indices],
            next_observations=self.storage["next_observations"][batch_indices],
            next_actions=self.storage["next_actions"][batch_indices],
            discounts=self.storage["discounts"][batch_indices],
            not_dones=self.storage["not_dones"][batch_indices],
        )

    def sample_task_inference_transitions(
        self,
        inference_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Samples (s, a) pairs and their rewards for every task. The reward of a
        transition is computed from the physics after taking a in s.
        Returns:
            observations: [inference_steps, observation_length]
            actions: [inference_steps, action_length]
            rewards: task -> [inference_steps, 1]
        """
        batch_indices = torch.randint(
            0, len(self.storage["observations"]), (inference_steps,)
        )
        actions = self.storage["actions"][batch_indices]
        _, rewards = self.reward_constructor.process_timesteps(
            batch_physics=self.storage["physics"][batch_indices],
            batch_actions=actions.detach().cpu().numpy(),
        )

        return self.storage["observations"][batch_indices], actions, rewards
