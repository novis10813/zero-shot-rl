"""Backward-compatibility shim; canonical module is zero_shot_rl.train.context_encoder."""

from zero_shot_rl.train.context_encoder import (
    VAL_SET_MOD,
    VAL_STOCK_SEED,
    ContextModel,
    Sampler,
    backward_features,
    load_context_model,
    load_episodes,
    losses,
    main,
    mlp,
)

__all__ = [
    "VAL_SET_MOD",
    "VAL_STOCK_SEED",
    "ContextModel",
    "Sampler",
    "backward_features",
    "load_context_model",
    "load_episodes",
    "losses",
    "main",
    "mlp",
]

if __name__ == "__main__":
    main()
