"""Geometry and damped fixed-point search for implicit-reward latents.

Reward definitions and constant-reward fastpaths belong to the callers. Numerical
operation order, warm starts and failure results preserve the original experiments.
"""

import math

import numpy as np
import torch


def normalized(z_hat, d):
    return math.sqrt(d) * z_hat / z_hat.norm()


def angle(a, b):
    cos = torch.dot(a, b) / (a.norm() * b.norm())
    return math.degrees(math.acos(float(cos.clamp(-1.0, 1.0))))


class LatentSolver:
    """Solver for a caller-provided dz(family, strength, z_hat) correction."""

    def z_hat_r(self, task, idx=None):
        r = self.rewards[task] if idx is None else self.rewards[task][idx]
        B = self.B if idx is None else self.B[idx]
        return (r * B).mean(0)

    def solve(self, family, strength, z_hat_r, init=None):
        """Damped fixed point for z-dependent families (started from `init`, default
        z_hat_r)."""
        z_hat = (z_hat_r if init is None else init).clone()
        for it in range(1, self.args.max_iters + 1):
            target = z_hat_r + self.dz(family, strength, z_hat)
            new = (1 - self.args.eta) * z_hat + self.args.eta * target
            residual = float((new - z_hat).norm() / z_hat_r.norm())
            z_hat = new
            if residual < self.args.tol:
                return z_hat, True, it
        return z_hat, False, self.args.max_iters

    def angle_grid(self, family, sign, z_hat_r):
        """Angle to z_r on a log grid of strengths, walked upwards with warm starts;
        stops at the first strength whose fixed point does not converge."""
        z_r = normalized(z_hat_r, self.d)
        scale = float(z_hat_r.norm() / self.dz(family, 1.0, z_hat_r).norm())
        grid, z_hat = [(0.0, 0.0, z_hat_r)], z_hat_r
        for s in scale * np.logspace(-3, 3, 25):
            z_hat, converged, _ = self.solve(family, sign * s, z_hat_r, init=z_hat)
            if not converged:
                break
            grid.append((s, angle(normalized(z_hat, self.d), z_r), z_hat))
        return grid

    def match_angle(self, family, sign, target, z_hat_r, grid):
        """Strength (with the given sign) whose solved z is `target` degrees from z_r:
        bisection inside the first grid interval that crosses it. None if unreachable."""
        z_r = normalized(z_hat_r, self.d)
        for (lo, th_lo, z_lo), (hi, th_hi, _) in zip(grid, grid[1:]):
            if th_lo < target <= th_hi:
                break
        else:
            return None
        for _ in range(self.args.bisect_iters):
            mid = 0.5 * (lo + hi)
            z_hat, converged, _ = self.solve(family, sign * mid, z_hat_r, init=z_lo)
            if not converged:
                return "no fixed point"
            if angle(normalized(z_hat, self.d), z_r) < target:
                lo, z_lo = mid, z_hat
            else:
                hi = mid
        return sign * 0.5 * (lo + hi), z_lo

