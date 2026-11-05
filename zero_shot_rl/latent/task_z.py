"""
Task latents of FB checkpoints, inferred as in OfflineRLWorkspace.eval.

Per task, z is inferred from --z_inference_steps reward-labelled transitions of the same
--dataset_transitions subsample of the dataset (same --seed for every checkpoint), so every
checkpoint sees the same inference transitions. Shared by the entries that compare the
actions of several checkpoints, which take --checkpoints LABEL path [path ...] once per
algorithm; a checkpoint's index counts within its label.
"""

from pathlib import Path

import torch

from agents.base import load_agent
from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor
from zero_shot_rl.utils import set_seed_everywhere

TASKS = ["stand", "walk", "run", "flip"]


def add_arguments(parser):
    parser.add_argument("--checkpoints", nargs="+", action="append", required=True,
                        help="LABEL path [path ...]; repeat per algorithm")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_set(args, device):
    """The dataset buffer and the task inference transitions, identical for every checkpoint."""
    set_seed_everywhere(args.seed)
    reward_constructor = RewardFunctionConstructor(
        domain_name="walker", task_names=TASKS, seed=args.seed, device=device
    )
    buffer = FBReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=Path(args.dataset_path),
        transitions=args.dataset_transitions,
        relabel=False,
        task=None,
        device=device,
        discount=args.discount,
        action_condition=None,
    )
    observations_z, rewards_z = buffer.sample_task_inference_transitions(
        inference_steps=args.z_inference_steps
    )
    return buffer, observations_z, rewards_z


def reward_functions(args, device):
    """task -> reward function of the simulator physics."""
    return RewardFunctionConstructor(
        domain_name="walker", task_names=TASKS, seed=args.seed, device=device
    ).reward_functions


def load_with_zs(checkpoint, observations_z, rewards_z, device):
    """The agent and its z per task (tensor [d_z] on `device`)."""
    agent = load_agent(Path(checkpoint), device)
    agent.eval()
    zs = {t: agent.infer_z(observations_z, rewards_z[t]) for t in TASKS}
    return agent, {t: torch.as_tensor(z, dtype=torch.float32, device=device) for t, z in zs.items()}


def load_checkpoints(args, observations_z, rewards_z, device):
    """(label, index, agent, zs) for every path of --checkpoints."""
    return [
        (label, index, *load_with_zs(path, observations_z, rewards_z, device))
        for label, *paths in args.checkpoints
        for index, path in enumerate(paths)
    ]
