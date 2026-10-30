"""Checks and row selection for latent NPZ files written by zero_shot_rl.latent.infer.

numpy only, so analysis code can use it without torch or MuJoCo.
"""

import numpy as np


def validate(data, source, tasks, z_dimension):
    n = len(data["task"])
    if data["z"].shape != (n, z_dimension):
        raise ValueError(f"{source}: expected {z_dimension}-dimensional latents")
    for key in ("family", "strength", "target_angle", "angle", "converged", "multi"):
        if data[key].shape != (n,):
            raise ValueError(f"{source}: malformed {key}")
    if set(data["task"]) != set(tasks):
        raise ValueError(f"{source}: unexpected tasks")
    if not np.isfinite(data["z"]).all():
        raise ValueError(f"{source}: nonfinite latent")
    if not np.allclose(np.linalg.norm(data["z"], axis=1), np.sqrt(z_dimension), atol=2e-6,
                       rtol=0):
        raise ValueError(f"{source}: latent norm differs from sqrt({z_dimension})")


def family_rows(data, families):
    """Every row of the given families, in file order."""
    return [row for row, family in enumerate(data["family"]) if family in families]


def matched_rows(data, tasks, families, target_angle):
    """Exactly one converged row per task and family, at `target_angle` degrees from
    z_r (z_r itself has no target). Ordered by task, then family."""
    rows = []
    for task in tasks:
        for family in families:
            mask = (data["task"] == task) & (data["family"] == family)
            if family != "z_r":
                mask &= data["target_angle"] == target_angle
            matches = np.flatnonzero(mask)
            if len(matches) != 1 or not data["converged"][matches[0]]:
                raise ValueError(f"Expected one converged {task}/{family} latent")
            rows.append(int(matches[0]))
    return rows
