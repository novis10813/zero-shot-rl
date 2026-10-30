"""Format of the per-step behavior traces written by zero_shot_rl.rollout.trace.

numpy only, so analysis code can use it without torch or MuJoCo.
"""

import numpy as np

EPISODES = 20  # environment seeds 0..EPISODES-1
STEPS = 1000
DT = 0.025  # walker control timestep, seconds
TRACE_KEYS = ("reward", "speed", "height", "upright", "angular_momentum")


def validate_trace(trace, data, rows):
    """`trace` replays latent rows `rows` of `data` (an infer NPZ)."""
    if not np.array_equal(trace["rows"], rows):
        raise ValueError("Trace rows differ from saved latent selection")
    if not np.array_equal(trace["episodes"], np.arange(EPISODES)):
        raise ValueError("Unexpected initialization seeds")
    if float(trace["dt"]) != DT:
        raise ValueError("Unexpected control timestep")
    if not np.array_equal(trace["z"], data["z"][rows]):
        raise ValueError("Trace latents differ from saved inference")
    for key in (*TRACE_KEYS, "angle"):
        shape = (len(rows), EPISODES, STEPS + (key == "angle"))
        if trace[key].shape != shape or not np.isfinite(trace[key]).all():
            raise ValueError(f"Invalid {key} trace")
    if np.any((trace["reward"] < 0) | (trace["reward"] > 1)):
        raise ValueError("Task reward outside [0,1]")
    if np.any(np.abs(trace["upright"]) > 1 + 1e-6):
        raise ValueError("Invalid uprightness")
    if np.any(np.abs(np.diff(np.unwrap(trace["angle"], axis=-1), axis=-1)) > np.pi / 2):
        raise ValueError("Angle sampling too sparse for reliable turns")
