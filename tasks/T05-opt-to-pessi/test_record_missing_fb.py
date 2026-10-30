"""Video supplement planning and encoder checks, without environment rollouts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_q1_fb import METHODS, ROOT, SOURCES
from zero_shot_rl.latent.archive import family_rows
from zero_shot_rl.rollout.video import record, verify_video, video_name


class VideoTests(unittest.TestCase):
    def test_saved_latent_coverage_and_names(self):
        for (_, source), count in zip(SOURCES, (100, 92, 99)):
            with np.load(ROOT / "data" / source, allow_pickle=False) as data:
                rows = family_rows(data, ("z_r", *METHODS))
                self.assertEqual(len(rows), count)
                names = [video_name(data, row) for row in rows]
                self.assertEqual(len(set(names)), count)
                self.assertTrue(all("AX" not in name and "C1" not in name and "C2" not in name
                                    for name in names))
                for task in ("stand", "walk", "run", "flip"):
                    for family in ("CQL", "ReBRAC", "ReBRAC+E2E", "E2E", "E2E-bi"):
                        self.assertEqual(sum(data["task"][r] == task and data["family"][r] == family
                                             and data["target_angle"][r] == 10 for r in rows), 1)

    def test_encoder_validation(self):
        good = dict(codec_name="h264", width=320, height=240,
                    nb_read_frames="500", r_frame_rate="30/1")
        with patch("zero_shot_rl.rollout.video.subprocess.run") as run:
            run.return_value.stdout = json.dumps(dict(streams=[good]))
            verify_video(Path("example.mp4"))
            for key, value in (("codec_name", "wrong"), ("width", 1), ("height", 1),
                               ("nb_read_frames", "499"), ("r_frame_rate", "15/1")):
                stream = dict(good, **{key: value})
                run.return_value.stdout = json.dumps(dict(streams=[stream]))
                with self.assertRaises(ValueError):
                    verify_video(Path("example.mp4"))

    def test_existing_video_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.mp4"
            path.write_bytes(b"preserve original")
            with patch("zero_shot_rl.rollout.video.render_episode") as render_episode:
                with self.assertRaises(FileExistsError):
                    record(None, None, path)
                render_episode.assert_not_called()
            self.assertEqual(path.read_bytes(), b"preserve original")


if __name__ == "__main__":
    unittest.main()
