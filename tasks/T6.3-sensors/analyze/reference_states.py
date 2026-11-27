"""Export the reference states used by the reward analyses: B(s), 4 sensors and true task rewards.

The reference set is the one `zero_shot_rl.finetune.anfis_z` uses (100,000 dataset transitions,
10,000 states, seed 42). Storing B(s) lets `analyze.reward_template` recompute implied rewards
r_z(s) = B(s)^T Sigma_B^{-1} z for any z without the simulator.

Output `data/inputs/reference_states.npz`: `backward` [N, d_z] (float32), `sensors` [N, 4],
`sensor_names`, `rewards` [N, tasks], `tasks`.

Usage (repo root):
    PYTHONPATH=.:tasks/T6.3-sensors MUJOCO_GL=egl uv run python -m analyze.reference_states \\
        --checkpoint <fb.pt> --dataset_path datasets/walker/rnd/dataset.npz \\
        --out tasks/T6.3-sensors/data/inputs/reference_states.npz
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from agents.base import load_agent
from zero_shot_rl.env.sensors import DOMAIN_TASKS
from zero_shot_rl.finetune.anfis_z import collect_reference_set
from zero_shot_rl.provenance import no_clobber

SENSORS = ("torso_height", "torso_upright", "horizontal_velocity", "angmomentum")


def main(argv: Sequence[str] = None) -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--out", type=Path, default=Path("tasks/T6.3-sensors/data/inputs/reference_states.npz"))
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--reference_seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    tasks = list(DOMAIN_TASKS["walker"])
    observations, features, rewards, _ = collect_reference_set(
        dataset_path=Path(args.dataset_path),
        domain="walker",
        sensor_names=list(SENSORS),
        tasks=tasks,
        dataset_transitions=args.dataset_transitions,
        z_inference_steps=args.z_inference_steps,
        seed=args.reference_seed,
        discount=args.discount,
        device=device,
    )
    agent = load_agent(Path(args.checkpoint), device)
    with torch.no_grad():
        backward = agent.FB.backward_representation(observations).detach().cpu().numpy().astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(args.out) as tmp:
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                backward=backward,
                sensors=features.cpu().numpy(),
                sensor_names=np.array(SENSORS),
                rewards=np.stack([rewards[t].cpu().numpy()[:, 0] for t in tasks], axis=1),
                tasks=np.array(tasks),
            )
    print(f"wrote {args.out}: backward {backward.shape}")


if __name__ == "__main__":
    main()
