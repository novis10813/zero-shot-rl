"""Tests for T6.4-fuzzy-add-rule summary aggregation."""

import unittest
from pathlib import Path

from analyze.summarize import build_summary_df, render_markdown


class SummarizeTest(unittest.TestCase):
    def test_summary_matches_stored_outputs(self):
        df = build_summary_df()
        self.assertEqual(len(df), 30)  # 6 conditions x (4 envs + 4-env mean)
        data_dir = Path("tasks/T6.4-fuzzy-add-rule/data")
        self.assertEqual(df.to_csv(index=False).encode("utf-8"), (data_dir / "summary.csv").read_bytes())
        self.assertEqual(render_markdown(df).encode("utf-8"), (data_dir / "summary.md").read_bytes())


if __name__ == "__main__":
    unittest.main()
