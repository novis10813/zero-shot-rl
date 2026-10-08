"""
Evaluates a saved FB / CFB checkpoint (a .pt file written by AbstractAgent.save) on
ExORL tasks with the same protocol as OfflineRLWorkspace.eval.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
from loguru import logger

from agents.base import load_agent
from agents.fb.replay_buffer import FBReplayBuffer
from agents.workspaces import OfflineRLWorkspace
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("domain_name", type=str)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--eval_tasks", nargs="+", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--discount", type=float, default=0.98)
parser.add_argument("--dataset_transitions", type=int, default=100000)
parser.add_argument("--z_inference_steps", type=int, default=10000)
parser.add_argument("--eval_rollouts", type=int, default=10)
parser.add_argument("--std_dev_eval", type=float, default=0.05)
args = parser.parse_args()

if args.domain_name == "point_mass_maze":
    raise NotImplementedError("Goal-state z inference is not supported here.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed_everywhere(args.seed)

reward_constructor = RewardFunctionConstructor(
    domain_name=args.domain_name,
    task_names=args.eval_tasks,
    seed=args.seed,
    device=device,
)

replay_buffer = FBReplayBuffer(
    reward_constructor=reward_constructor,
    dataset_path=Path(args.dataset_path),
    transitions=args.dataset_transitions,
    relabel=False,
    task=None,
    device=device,
    discount=args.discount,
    action_condition=None,
)

agent = load_agent(Path(args.checkpoint), device)

workspace = OfflineRLWorkspace(
    reward_constructor=reward_constructor,
    learning_steps=0,
    model_dir=Path(args.checkpoint).parent,
    eval_frequency=1,
    eval_rollouts=args.eval_rollouts,
    z_inference_steps=args.z_inference_steps,
    train_std=None,
    eval_std=args.std_dev_eval,
    wandb_logging=False,
    device=device,
)
(
    workspace.observations_z,
    workspace.rewards_z,
) = replay_buffer.sample_task_inference_transitions(
    inference_steps=args.z_inference_steps
)

metrics = workspace.eval(agent=agent, tasks=args.eval_tasks)
logger.info(
    f"Checkpoint {args.checkpoint}: "
    + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
)
