"""Backward-compatibility shim; canonical module is zero_shot_rl.finetune.online_inference."""

from zero_shot_rl.finetune.online_inference import (
    FIELDS,
    PREDICTION_STEPS,
    SNAPSHOTS,
    main,
    make_selector,
    parse_rule,
    run_task,
)

__all__ = [
    "FIELDS",
    "PREDICTION_STEPS",
    "SNAPSHOTS",
    "main",
    "make_selector",
    "parse_rule",
    "run_task",
]

if __name__ == "__main__":
    main()
