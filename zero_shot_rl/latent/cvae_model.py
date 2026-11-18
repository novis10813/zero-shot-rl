"""Backward-compatibility shim; canonical module is zero_shot_rl.train.cvae_model."""

from zero_shot_rl.train.cvae_model import (
    CVAE,
    CVAEEnsemble,
    fit,
    icp_threshold,
    is_ood,
    load_cvae_ensemble,
    main,
    mlp,
)

__all__ = [
    "CVAE",
    "CVAEEnsemble",
    "fit",
    "icp_threshold",
    "is_ood",
    "load_cvae_ensemble",
    "main",
    "mlp",
]

if __name__ == "__main__":
    main()
