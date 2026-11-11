"""Component test for the trace comparison of zero_shot_rl.eval.fb_trajectories."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from zero_shot_rl.eval.fb_trajectories import compare_with_trace


class CompareWithTraceTest(unittest.TestCase):
    def test_difference_on_shared_episodes_only(self):
        data = {"task": np.array(["walk", "walk", "run"]), "family": np.array(["CQL", "z_r", "z_r"])}
        reference = np.random.default_rng(0).random((2, 3, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            np.savez(path, rows=np.array([1, 2]), reward=reference)
            rewards = np.zeros((2, 5, 4))
            rewards[0, :3], rewards[1, :3] = reference[0], reference[1]
            rewards[:, 3:] = 99.0
            self.assertEqual(compare_with_trace(path, data, ["walk", "run"], [1, 2], rewards), 0.0)
            rewards[1, 2, 3] += 0.5
            self.assertAlmostEqual(compare_with_trace(path, data, ["walk", "run"], [1, 2], rewards), 0.5)


if __name__ == "__main__":
    unittest.main()
