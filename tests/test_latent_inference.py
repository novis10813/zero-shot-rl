"""Standard FB inference buffer, latent solver and entry-module imports.

Run from the repo root: PYTHONPATH=. python -m unittest discover -s tests
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from agents.fb.inference_buffer import FBInferenceReplayBuffer
from agents.fb.replay_buffer import FBReplayBuffer
from zero_shot_rl.latent.implicit import FBImplicitZ
from zero_shot_rl.latent.solver import LatentSolver, angle, normalized


REPO = Path(__file__).resolve().parents[1]


def make_episode(length, offset):
    values = np.arange(length + 1, dtype=np.float32) + offset
    return {
        "observation": np.stack([values, values + 0.5], axis=1),
        "action": values[:, None],
        "reward": np.zeros((length + 1, 1), dtype=np.float32),
        "discount": np.ones((length + 1, 1), dtype=np.float32),
        "physics": values[:, None],
    }


class InferenceBufferTests(unittest.TestCase):
    def test_next_actions_follow_subsampling_and_episode_boundaries(self):
        episodes = [make_episode(3, 0), make_episode(4, 10), make_episode(1, 20)]
        actions = np.concatenate([e["action"][1:] for e in episodes])
        next_actions = np.concatenate([
            np.concatenate([e["action"][2:], e["action"][-1:]]) for e in episodes
        ])
        valid = np.array([True, True, False, True, True, True, False, False])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.npz"
            np.savez(path, **{f"episode_{i}": e for i, e in enumerate(episodes)})
            for transitions in (None, 5):
                with self.subTest(transitions=transitions):
                    kwargs = dict(reward_constructor=None, dataset_path=path, discount=0.98,
                                  transitions=transitions, device=torch.device("cpu"))
                    np.random.seed(7)
                    torch.manual_seed(7)
                    plain = FBReplayBuffer(**kwargs)
                    numpy_after = np.random.get_state()
                    torch_after = torch.get_rng_state()
                    np.random.seed(7)
                    torch.manual_seed(7)
                    enriched = FBInferenceReplayBuffer(**kwargs)
                    for key, value in plain.storage.items():
                        np.testing.assert_array_equal(enriched.storage[key], value)
                    actual_numpy = np.random.get_state()
                    self.assertEqual(actual_numpy[0], numpy_after[0])
                    np.testing.assert_array_equal(actual_numpy[1], numpy_after[1])
                    self.assertEqual(actual_numpy[2:], numpy_after[2:])
                    self.assertTrue(torch.equal(torch.get_rng_state(), torch_after))
                    indices = np.random.default_rng(42).choice(
                        len(actions), len(actions) if transitions is None else transitions,
                        replace=False,
                    )
                    np.testing.assert_array_equal(enriched.storage["actions"], actions[indices])
                    np.testing.assert_array_equal(enriched.storage["next_actions"],
                                                  next_actions[indices])
                    np.testing.assert_array_equal(enriched.valid_indices,
                                                  np.flatnonzero(valid[indices]))
                    self.assertIs(FBInferenceReplayBuffer.sample, FBReplayBuffer.sample)
                    self.assertIs(FBInferenceReplayBuffer.sample_task_inference_transitions,
                                  FBReplayBuffer.sample_task_inference_transitions)


class ToySolver(LatentSolver):
    def __init__(self, **overrides):
        settings = dict(eta=0.3, tol=1e-6, max_iters=300, bisect_iters=10)
        settings.update(overrides)
        self.args = SimpleNamespace(**settings)
        self.d = 2
        self.B = torch.tensor([[1., 2.], [3., 4.]])
        self.rewards = {"task": torch.tensor([[2.], [5.]])}
        self.calls = []

    def dz(self, family, strength, z_hat):
        if family == "linear":
            return strength * torch.tensor([0., 1.])
        return strength * torch.tensor([0., 1.]) + 0.2 * z_hat

    def solve(self, family, strength, z_hat_r, init=None):
        self.calls.append((strength, init))
        if family == "linear" or strength == 0:
            return z_hat_r + self.dz(family, strength, z_hat_r), True, 0
        return super().solve(family, strength, z_hat_r, init)


class SolverTests(unittest.TestCase):
    def test_geometry_and_reward_projection(self):
        solver = ToySolver()
        vector = torch.tensor([3., 4.])
        self.assertTrue(torch.equal(normalized(vector, 2), 2 ** 0.5 * vector / vector.norm()))
        self.assertEqual(angle(torch.tensor([1., 0.]), torch.tensor([0., 1.])), 90.)
        self.assertTrue(torch.equal(solver.z_hat_r("task"), torch.tensor([8.5, 12.])))
        self.assertTrue(torch.equal(solver.z_hat_r("task", torch.tensor([1])),
                                    torch.tensor([15., 20.])))

    def test_damped_iteration_matches_original_arithmetic_and_counts(self):
        baseline = torch.tensor([1., 0.])
        for max_iters in (1, 300):
            for init in (None, torch.tensor([2., 3.])):
                with self.subTest(max_iters=max_iters, init=init):
                    solver = ToySolver(max_iters=max_iters)
                    expected = (baseline if init is None else init).clone()
                    converged = False
                    for iteration in range(1, max_iters + 1):
                        target = baseline + solver.dz("feedback", 0.5, expected)
                        updated = (1 - solver.args.eta) * expected + solver.args.eta * target
                        residual = float((updated - expected).norm() / baseline.norm())
                        expected = updated
                        if residual < solver.args.tol:
                            converged = True
                            break
                    result, actual_converged, actual_iters = solver.solve(
                        "feedback", 0.5, baseline, init=init
                    )
                    self.assertTrue(torch.equal(result, expected))
                    self.assertEqual((actual_converged, actual_iters), (converged, iteration))
        self.assertTrue(torch.equal(baseline, torch.tensor([1., 0.])))

    def test_angle_grid_warm_starts_and_signed_bisection(self):
        baseline = torch.tensor([1., 0.])
        for sign in (-1, 1):
            solver = ToySolver()
            grid = solver.angle_grid("linear", sign, baseline)
            self.assertEqual(len(grid), 26)
            np.testing.assert_array_equal([g[0] for g in grid[1:]], np.logspace(-3, 3, 25))
            for index, (strength, init) in enumerate(solver.calls):
                self.assertEqual(strength, sign * grid[index + 1][0])
                self.assertIs(init, grid[index][2])
            strength, warm_start = solver.match_angle("linear", sign, 10., baseline, grid)
            self.assertEqual(np.sign(strength), sign)
            self.assertLess(abs(angle(warm_start, baseline) - 10.), 0.02)
            self.assertIsNone(solver.match_angle("linear", sign, 100., baseline, grid))

    def test_nonconvergence_stops_grid_and_is_reported_in_bisection(self):
        solver = ToySolver(max_iters=1, tol=0.)
        baseline = torch.tensor([1., 0.])
        grid = solver.angle_grid("feedback", 1, baseline)
        self.assertEqual(len(grid), 1)
        self.assertEqual(solver.match_angle("feedback", 1, 10., baseline,
                                           [(0., 0., baseline), (1., 45., baseline)]),
                         "no fixed point")

    def test_standard_fb_keeps_zero_and_constant_reward_fastpaths(self):
        self.assertTrue(issubclass(FBImplicitZ, LatentSolver))
        solver = FBImplicitZ.__new__(FBImplicitZ)
        solver.u_1 = torch.tensor([0., 2.])
        solver.args = SimpleNamespace(eta=0.3, tol=1e-3, max_iters=300)
        baseline = torch.tensor([1., 0.])
        for strength in (0., 0.5):
            result, converged, iterations = solver.solve("E2E", strength, baseline)
            self.assertTrue(torch.equal(result, baseline + strength * solver.u_1))
            self.assertEqual((converged, iterations), (True, 0))


class ImportTests(unittest.TestCase):
    def test_entrypoints_import(self):
        script = """
import importlib
for name in ('zero_shot_rl.latent.infer', 'zero_shot_rl.rollout.evaluate',
             'zero_shot_rl.rollout.video', 'zero_shot_rl.rollout.trace'):
    importlib.import_module(name)
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO)
        result = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=environment,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
