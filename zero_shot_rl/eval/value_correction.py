"""Backward-compatibility shim; canonical module is zero_shot_rl.finetune.value_correction."""

from zero_shot_rl.finetune.value_correction import (
    EVAL_FIELDS,
    TRAIN_FIELDS,
    VALUE_STEPS,
    eval_points,
    evaluate,
    frozen_rows,
    main,
    run,
    write_csv,
)

__all__ = [
    "EVAL_FIELDS",
    "TRAIN_FIELDS",
    "VALUE_STEPS",
    "eval_points",
    "evaluate",
    "frozen_rows",
    "main",
    "run",
    "write_csv",
]

if __name__ == "__main__":
    main()
