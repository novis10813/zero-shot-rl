"""Backward-compatibility shim; canonical module is zero_shot_rl.eval.cvae_score."""

from zero_shot_rl.eval.cvae_score import CHUNK, main, score_file

__all__ = ["CHUNK", "main", "score_file"]

if __name__ == "__main__":
    main()
