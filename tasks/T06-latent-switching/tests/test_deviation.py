"""Join of action deviations with logged returns (analyze.deviation)."""

import csv
from pathlib import Path
import tempfile
import unittest

from analyze import deviation


def write(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class DeviationTest(unittest.TestCase):
    def test_each_policy_gets_its_own_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write(tmp / "dev.csv", deviation.FIELDS[1:8], [
                {"kind": "candidate", "name": "z_r 0", "dev_data": 10, "dev_opt": 0},
                {"kind": "candidate", "name": "T05 z_r", "dev_data": 11, "dev_opt": 1},
                {"kind": "switch", "name": "switch 0.01", "kappa": 0.01, "dev_data": 12,
                 "dev_opt": 2, "pess_frac": 0.8},
                {"kind": "random", "name": "random 0.5", "p": 0.5, "dev_data": 13,
                 "dev_opt": 3, "pess_frac": 0.5},
            ])
            write(tmp / "rollouts.csv", ["family", "target_angle", "episode", "return"], [
                {"family": f, "target_angle": a, "episode": e, "return": r + e}
                for f, a, r in (("z_r", "0", 100), ("T05 z_r", "nan", 200))
                for e in range(2)
            ])
            write(tmp / "switch.csv", ["condition", "kappa", "return"], [
                {"condition": "opt", "kappa": "nan", "return": 1},
                {"condition": "switch", "kappa": "0.01", "return": 300},
                {"condition": "switch", "kappa": "0.01", "return": 302},
            ])
            write(tmp / "random.csv", ["condition", "p", "return"], [
                {"condition": "random", "p": "0.5", "return": 400},
                {"condition": "random", "p": "0.75", "return": 1},
            ])
            rows = deviation.seed_rows(0, tmp / "dev.csv", tmp / "rollouts.csv",
                                       tmp / "switch.csv", tmp / "random.csv")
            write(tmp / "final.csv", ["condition", "return"], [
                {"condition": "opt", "return": 500}, {"condition": "opt", "return": 502},
                *({"condition": c, "return": 1} for c in ("pess", "random", "switch")),
            ])
            with open(tmp / "dev.csv", "a", newline="") as f:
                csv.writer(f).writerow(["candidate", "CQL 7.5", "", "", "", 9, 4])
                csv.writer(f).writerow(["switch", "switch 0", 0, "", 0.4, 9, 5])
            write(tmp / "switch.csv", ["condition", "kappa", "return"],
                  [{"condition": "switch", "kappa": k, "return": 1} for k in ("0.01", "0")])
            with open(tmp / "rollouts.csv", "a", newline="") as f:
                csv.writer(f).writerow(["CQL", "7.5", 0, 1])
            final = deviation.seed_rows(0, tmp / "dev.csv", tmp / "rollouts.csv",
                                        tmp / "switch.csv", tmp / "random.csv",
                                        tmp / "final.csv")
        got = {r["name"]: (r["return"], r["episodes"]) for r in rows}
        self.assertEqual(got, {"z_r 0": (100.5, 2), "T05 z_r": (200.5, 2),
                               "switch 0.01": (301.0, 2), "random 0.5": (400.0, 1)})
        pure = [r for r in final if r["kind"] == "final" and r["name"] == "pure FB"]
        self.assertEqual([(r["return"], r["dev_opt"]) for r in pure], [(501.0, "0")])


if __name__ == "__main__":
    unittest.main()
