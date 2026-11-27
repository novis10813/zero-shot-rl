"""Tests for the T6.3 sep-CMA-ES versus T6.2 comparison table."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze.sep_vs_full import ENVS, SEEDS, summarize, to_markdown


class SepVsFullTest(unittest.TestCase):
    def test_pairs_seeds_and_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cond, offset in (("a", 0.0), ("b", 100.0)):
                for env in ENVS:
                    for s in (*SEEDS, "ft1030"):  # ft1030 must be ignored
                        d = root / cond / env / s
                        d.mkdir(parents=True)
                        extra = 1e6 if s == "ft1030" else 0.0
                        pd.DataFrame({
                            "step": [0, 40000],
                            "stand": [offset, offset + 10 + extra],
                            "walk": [offset, offset + 10 + extra],
                            "run": [offset, offset + 10 + extra],
                            "flip": [offset, offset + 10 + extra],
                        }).to_csv(d / "table6.csv", index=False)
            df = summarize((("A", root, "a"), ("B", root, "b")))
            self.assertEqual(len(df), 2 * len(ENVS) + 2)
            mean = df[df.env == "4-env mean"].set_index("condition")
            self.assertAlmostEqual(mean.loc["A", "final_iqm"], 10.0)
            self.assertAlmostEqual(mean.loc["B", "select_iqm"], 110.0)
            self.assertIn("| 4-env mean | B |", to_markdown(df))


if __name__ == "__main__":
    unittest.main()
