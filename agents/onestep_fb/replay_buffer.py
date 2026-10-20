"""Replay buffer for one-step FB: adds the dataset's next action to each transition."""

import dataclasses
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from agents.base import Batch
from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor


@dataclasses.dataclass
class OneStepBatch(Batch):
    next_actions: Optional[torch.Tensor] = None


class OneStepFBReplayBuffer(FBReplayBuffer):
    """
    FB buffer whose transitions also carry the next dataset action. Transition t of an
    episode is (obs[t], action[t + 1], obs[t + 1]), so its next action is action[t + 2];
    the last transition of each episode has none and is never sampled for training.
    """

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        dataset_path: Path,
        discount: float,
        transitions: int,
        device: torch.device,
        relabel: bool = False,
        task: str = None,
    ):
        super().__init__(
            reward_constructor=reward_constructor,
            dataset_path=dataset_path,
            discount=discount,
            transitions=transitions,
            device=device,
            relabel=relabel,
            task=task,
        )

        # rebuild the concatenated next actions and repeat the parent's subsampling
        episodes = dict(np.load(dataset_path, allow_pickle=True))
        actions, next_actions, has_next = [], [], []
        for _, episode in episodes.items():
            episode_actions = episode.item()["action"]
            actions.append(episode_actions[1:])
            next_actions.append(np.concatenate([episode_actions[2:], episode_actions[-1:]]))
            valid = np.ones(len(episode_actions) - 1, dtype=bool)
            valid[-1] = False
            has_next.append(valid)
        dataset_length = sum(len(a) for a in actions)
        rng = np.random.default_rng(42)
        if transitions is None:
            sample_indices = rng.choice(dataset_length, dataset_length, replace=False)
        else:
            sample_indices = rng.choice(dataset_length, transitions, replace=False)

        # the parent stored action[t + 1] for the same indices: check the alignment
        stored = self.storage["actions"].cpu().numpy()
        if not np.array_equal(np.concatenate(actions)[sample_indices], stored):
            raise RuntimeError("Next-action indices do not match the stored transitions.")

        self.storage["next_actions"] = torch.as_tensor(
            np.concatenate(next_actions)[sample_indices], device=self.device
        )
        self.valid_indices = torch.as_tensor(
            np.flatnonzero(np.concatenate(has_next)[sample_indices]), device=self.device
        )

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
