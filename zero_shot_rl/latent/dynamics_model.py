"""Backward-compatibility shim; canonical module is zero_shot_rl.train.dynamics_model."""

from zero_shot_rl.train.dynamics_model import (
    TARGETS,
    DynamicsModel,
    exorl_statistics,
    fit,
    flatten,
    load_dynamics_model,
    load_rollouts,
    main,
    split_episodes,
)

__all__ = [
    "TARGETS",
    "DynamicsModel",
    "exorl_statistics",
    "fit",
    "flatten",
    "load_dynamics_model",
    "load_rollouts",
    "main",
    "split_episodes",
]

if __name__ == "__main__":
    main()
