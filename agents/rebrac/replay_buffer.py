"""Single-task offline buffer with the next dataset action of each transition (ReBRAC)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agents.base import OfflineReplayBuffer
from rewards import RewardFunctionConstructor


@dataclass
class ReBRACBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    next_actions: torch.Tensor
    has_next_actions: torch.Tensor  # [batch, 1], 0 for an episode's final transition


class ReBRACReplayBuffer(OfflineReplayBuffer):
    """OfflineReplayBuffer plus next_actions and has_next_actions.

    A transition uses observation[t] and action[t + 1]. Its next dataset action is
    action[t + 2], which does not exist for an episode's final transition: that
    transition stores its own action and has_next_actions = 0. The seed-42
    subsampling of the parent is repeated and checked against the stored actions,
    as in agents/fb/inference_buffer.py.
    """

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        dataset_path: Path,
        discount: float,
        device: torch.device,
        task: str,
        transitions: int = None,
    ):
        super().__init__(
            reward_constructor=reward_constructor,
            dataset_path=dataset_path,
            discount=discount,
            device=device,
            task=task,
            relabel=True,
            transitions=transitions,
        )

        episodes = dict(np.load(dataset_path, allow_pickle=True))
        actions, next_actions, has_next = [], [], []
        for _, episode in episodes.items():
            episode_actions = episode.item()["action"]
            actions.append(episode_actions[1:])
            next_actions.append(
                np.concatenate([episode_actions[2:], episode_actions[-1:]])
            )
            valid = np.ones(len(episode_actions) - 1, dtype=np.float32)
            valid[-1] = 0.0
            has_next.append(valid)
        dataset_length = sum(len(a) for a in actions)
        rng = np.random.default_rng(42)
        if transitions is None:
            sample_indices = rng.choice(dataset_length, dataset_length, replace=False)
        else:
            sample_indices = rng.choice(dataset_length, transitions, replace=False)

        stored = self.storage["actions"].cpu().numpy()
        if not np.array_equal(np.concatenate(actions)[sample_indices], stored):
            raise RuntimeError("Next-action indices do not match the stored transitions.")

        self.storage["next_actions"] = torch.as_tensor(
            np.concatenate(next_actions)[sample_indices], device=self.device
        )
        self.storage["has_next_actions"] = torch.as_tensor(
            np.concatenate(has_next)[sample_indices], device=self.device
        ).unsqueeze(-1)

    def sample(self, batch_size: int) -> ReBRACBatch:
        idx = torch.randint(0, len(self.storage["observations"]), (batch_size,))
        return ReBRACBatch(
            observations=self.storage["observations"][idx],
            actions=self.storage["actions"][idx],
            rewards=self.storage["rewards"][idx],
            next_observations=self.storage["next_observations"][idx],
            next_actions=self.storage["next_actions"][idx],
            has_next_actions=self.storage["has_next_actions"][idx],
        )
