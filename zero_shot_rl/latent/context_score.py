"""Backward-compatibility shim; canonical module is zero_shot_rl.eval.context_score."""

from zero_shot_rl.eval.context_score import STARTS, contexts, main, prediction_errors

__all__ = ["STARTS", "contexts", "main", "prediction_errors"]

if __name__ == "__main__":
    main()
