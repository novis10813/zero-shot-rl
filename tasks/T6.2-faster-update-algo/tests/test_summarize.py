"""Tests for T6.2-faster-update-algo summary aggregation and markdown table output."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze.summarize import summarize, to_markdown


class T62SummarizeTest(unittest.TestCase):
    def test_summarize_and_markdown_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for env, offset in (
                ("friction_mild", 0.0),
                ("friction_high_mild", 10.0),
                ("actuator_mild", 20.0),
                ("actuator_high_mild", 30.0),
            ):
                for s in (1000, 1010):
                    d = root / "cmaes_z" / env / f"ft{s}"
                    d.mkdir(parents=True, exist_ok=True)
                    df = pd.DataFrame({
                        "step": [0, 40000, 80000],
                        "stand": [100 + offset, 500 + offset, 450 + offset],
                        "walk": [100 + offset, 400 + offset, 380 + offset],
                        "run": [50 + offset, 200 + offset, 190 + offset],
                        "flip": [80 + offset, 300 + offset, 280 + offset],
                    })
                    df.to_csv(d / "table6.csv", index=False)

            # Also add an incomplete condition (1 seed only) that must be excluded when min_seeds=2.
            inc = root / "rela_z" / "friction_mild" / "ft1000"
            inc.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "step": [0, 40000, 80000],
                "stand": [100, 900, 900],
                "walk": [100, 900, 900],
                "run": [50, 500, 500],
                "flip": [80, 600, 600],
            }).to_csv(inc / "table6.csv", index=False)

            out = summarize(root, t61_summary=None, min_seeds=2)
            self.assertEqual(len(out), 5)  # 4 envs + mean_of_4_dynamics; incomplete rela_z excluded
            self.assertNotIn("rela_z", set(out["condition"]))
            fm = out[out["dynamics"] == "friction_mild"].iloc[0]
            # At step 40000: sorted tasks [200, 300, 400, 500] -> IQM = mean(300, 400) = 350.0
            self.assertAlmostEqual(float(fm["select_iqm"]), 350.0, places=5)
            # At step 80000: sorted tasks [190, 280, 380, 450] -> IQM = mean(280, 380) = 330.0
            self.assertAlmostEqual(float(fm["final_step_iqm"]), 330.0, places=5)

            md = to_markdown(out)
            header = md.splitlines()[0]
            self.assertEqual(
                header,
                "| OOD env | algo | searching space | select IQM | final step IQM |",
            )
            self.assertIn("| 4-env mean | CMA-ES (full-cov) | z | 365.0 | 345.0 |", md)


if __name__ == "__main__":
    unittest.main()
