"""Float64 latent geometry for analysis of saved latents."""

import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError("Expected a finite, nonzero vector")
    return vector / norm


def cosine(a, b):
    return float(np.clip(unit(a) @ unit(b), -1.0, 1.0))


def angle_deg(a, b):
    return float(np.degrees(np.arccos(cosine(a, b))))


def tangent(z, baseline):
    """Unit component orthogonal to the raw-reward baseline, not z - baseline."""
    u = unit(baseline)
    return unit(unit(z) - cosine(z, baseline) * u)
