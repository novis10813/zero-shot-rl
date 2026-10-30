"""FB inference data with aligned next dataset actions for reward corrections."""

from pathlib import Path

import numpy as np
import torch

from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor


class FBInferenceReplayBuffer(FBReplayBuffer):
    """Add next actions and their valid indices without changing FB sampling APIs.

    A transition uses observation[t] and action[t + 1]. Its next dataset action is
    action[t + 2], which does not exist for an episode's final transition.
    Inference callers select valid_indices explicitly. The final next-action
    placeholder and seed-42 subsampling preserve the historical T05 inference set.
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

