"""Task-specific metrics, pairing, physical sign and trace validation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze.task_behavior import (analyze, behavior_rows, report_table, rotation_metrics,
                                   trace_name, validate_trace)
from common import INPUTS, ROOT, SOURCES, csv_text
from zero_shot_rl.eval.env import make_env
from zero_shot_rl.eval.trace import publish_npz, torso_angle
from zero_shot_rl.eval.trace_format import DT, EPISODES, STEPS, TRACE_KEYS


def wrap(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))


class TaskBehaviorTests(unittest.TestCase):
    def test_complete_turns_and_direction(self):
        for turns, expected in ((0, (0, 0)), (0.9, (0, 0)), (1, (1, 0)),
                                (2.3, (2, 0)), (-1, (0, 1)), (-2.3, (0, 2))):
            angles = 0.4 + np.linspace(0, turns * 2 * np.pi, 1001)
            p, n, net = rotation_metrics(wrap(angles))
            self.assertEqual((p, n), expected)
            self.assertAlmostEqual(net, turns)

    def test_partial_reversals_and_boundary_chatter(self):
        # Repeated almost-full out-and-back movement never completes a turn.
        angles = np.tile(np.r_[np.linspace(0, 1.9 * np.pi, 50),
                                np.linspace(1.9 * np.pi, 0, 50)], 10)
        self.assertEqual(rotation_metrics(wrap(angles))[:2], (0, 0))
        angles = np.r_[np.linspace(0, 2 * np.pi + 0.01, 100),
                       2 * np.pi + np.sin(np.arange(100)) * 0.01]
        self.assertEqual(rotation_metrics(wrap(angles))[:2], (1, 0))
        angles = np.r_[np.linspace(0, 2 * np.pi, 100), np.linspace(2 * np.pi, 0, 100)]
        self.assertEqual(rotation_metrics(wrap(angles)), (1, 1, 0.0))
        for invalid in ([0], [0, np.nan], [[0, 1], [2, 3]]):
            with self.assertRaises(ValueError):
                rotation_metrics(invalid)

    def test_torso_angle_sign_matches_positive_y_rotation(self):
        env = make_env(0)
        try:
            env.reset()
            self.assertAlmostEqual(env.control_timestep(), DT)
            for pitch in (-0.3, 0, 0.3):
                with env.physics.reset_context():
                    env.physics.named.data.qpos["rooty"] = pitch
                    env.physics.data.qvel[:] = 0
                    env.physics.named.data.qvel["rooty"] = 1
                self.assertAlmostEqual(torso_angle(env.physics), pitch)
                self.assertGreater(env.physics.angmomentum(), 0)
        finally:
            env.close()

    def test_selection_and_rejection(self):
        for _, source in SOURCES:
            with np.load(INPUTS / source, allow_pickle=False) as archive:
                data = dict(archive)
            rows = behavior_rows(data)
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(data["family"][r] == "z_r" for r in rows), 4)
            bad = {k: v.copy() for k, v in data.items()}
            bad["converged"][rows[1]] = False
            with self.assertRaises(ValueError):
                behavior_rows(bad)

    def test_saved_trace_tables_and_report(self):
        tables = analyze(INPUTS)
        for name, rows in tables.items():
            self.assertEqual((ROOT / "data" / name).read_text(), csv_text(rows))
        self.assertIn(report_table(tables), (ROOT / "REPORT.md").read_text())
        episodes = tables["behavior_fb_episodes.csv"]
        for seed, source in SOURCES:
            with np.load(INPUTS / source, allow_pickle=False) as archive:
                data = dict(archive)
            with np.load(INPUTS / trace_name(seed), allow_pickle=False) as archive:
                trace = dict(archive)
            for index, row in enumerate(behavior_rows(data)):
                group = [r for r in episodes if r["training_seed"] == seed and r["source_row"] == row]
                self.assertEqual(len(group), EPISODES)
                self.assertEqual([r["episode"] for r in group], list(range(EPISODES)))
                for episode, record in enumerate(group):
                    self.assertAlmostEqual(record["return"], trace["reward"][index, episode].sum())
                    self.assertAlmostEqual(record["speed"], trace["speed"][index, episode].mean())
                    self.assertAlmostEqual(record["net_displacement_estimate"],
                                           trace["speed"][index, episode].sum() * DT)
                    self.assertAlmostEqual(record["net_displacement_estimate"],
                                           record["speed"] * STEPS * DT)
                    self.assertAlmostEqual(record["walk_speed_mae"],
                                           np.abs(trace["speed"][index, episode] - 1).mean())
                    unwrapped = np.unwrap(trace["angle"][index, episode])
                    self.assertAlmostEqual(record["net_turns"],
                                           (unwrapped[-1] - unwrapped[0]) / (2 * np.pi))

    def test_synthetic_trace_aggregation_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for seed, source in SOURCES:
                with np.load(INPUTS / source, allow_pickle=False) as archive:
                    data = dict(archive)
                np.savez(directory / source, **data)
                rows = behavior_rows(data)
                shape = (24, EPISODES, STEPS)
                trace = {key: np.zeros(shape) for key in TRACE_KEYS}
                trace.update(rows=np.array(rows), episodes=np.arange(EPISODES), dt=np.array(DT),
                             z=data["z"][rows], angle=np.zeros((24, EPISODES, STEPS + 1)))
                trace["reward"][:] = 0.2 + seed * 0.1
                trace["speed"][:] = 1
                trace["height"][:] = 1.2
                trace["upright"][:] = 1
                for i, row in enumerate(rows):
                    if data["family"][row] != "z_r":
                        trace["reward"][i] += 0.05
                        trace["speed"][i] += 0.1
                        trace["angle"][i] = wrap(np.linspace(0, 2 * np.pi, STEPS + 1))
                validate_trace(trace, data)
                for key, value in (("reward", 1.1), ("upright", 1.1), ("speed", np.nan)):
                    broken = dict(trace, **{key: trace[key].copy()})
                    broken[key][0, 0, 0] = value
                    with self.assertRaises(ValueError):
                        validate_trace(broken, data)
                broken = dict(trace, angle=trace["angle"].copy())
                broken["angle"][0, 0, 1] = 2.0
                with self.assertRaises(ValueError):
                    validate_trace(broken, data)
                broken = dict(trace, angle=np.zeros((24, EPISODES, STEPS)))
                with self.assertRaises(ValueError):
                    validate_trace(broken, data)
                publish_npz(directory / trace_name(seed), trace)
                with self.assertRaises(FileExistsError):
                    publish_npz(directory / trace_name(seed), dict(test=np.array(1)))
            tables = analyze(directory)
            self.assertEqual([len(rows) for rows in tables.values()], [1440, 72, 24])
            for row in tables["behavior_fb_task_summary.csv"]:
                baseline = row["family"] == "z_r"
                self.assertAlmostEqual(row["return_mean"], 300 if baseline else 350)
                self.assertAlmostEqual(row["delta_return_mean"], 0 if baseline else 50)
                self.assertAlmostEqual(row["net_displacement_estimate_mean"], 25 if baseline else 27.5)
                self.assertAlmostEqual(row["delta_net_displacement_estimate_mean"], 0 if baseline else 2.5)
                self.assertAlmostEqual(row["positive_turns_mean"], 0 if baseline else 1)
                self.assertAlmostEqual(row["walk_speed_mae_mean"], 0 if baseline else 0.1)
            for row in tables["behavior_fb_seed_summary.csv"]:
                self.assertAlmostEqual(row["delta_return_se"], 0)
            table = report_table(tables)
            self.assertIn("正向完整圈數", table)
            self.assertIn("距離／速度", table)
            self.assertEqual(len(table.splitlines()), 26)


if __name__ == "__main__":
    unittest.main()
