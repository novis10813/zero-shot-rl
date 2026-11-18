"""Backward-compatibility shim; canonical module is zero_shot_rl.eval.dynamics_score."""

from zero_shot_rl.eval.dynamics_score import main, score_file

__all__ = ["main", "score_file"]

if __name__ == "__main__":
    main()
