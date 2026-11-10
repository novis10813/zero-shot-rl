"""ICP rows and fit summary of analyze.detect on known samples."""

import unittest

import numpy as np

from analyze.detect import CONFIGS, fit_rows, icp_rows, render


def data(shift):
    rng = np.random.default_rng(0)
    out = {"val_member": rng.uniform(0, 1, size=(3, 1000)).astype(np.float32),
           "min_rollout": rng.uniform(0, 1, size=(4, 50)).astype(np.float32)}
    for c in CONFIGS:
        out[f"min_{c}"] = out["min_rollout"] + shift
    return out


class DetectTest(unittest.TestCase):
    def test_icp_rows_use_pooled_threshold_and_min_rule(self):
        rows = icp_rows("run", data(10.0))
        self.assertEqual(len(rows), 2 * len(CONFIGS))
        first = rows[0]
        self.assertAlmostEqual(first["thr"], 0.95, delta=0.01)
        self.assertEqual(first["tpr"], 1.0)
        self.assertLess(first["fpr"], 0.1)
        self.assertEqual(icp_rows("run", data(-10.0))[0]["tpr"], 0.0)

    def test_fit_rows_average_members_and_handle_missing_sigma(self):
        from analyze import detect
        fit = {r: [{"best_epoch": 10, "val_rec": 1.0, "val_kl": 0.0, "active_dims": 0, "sigma": None},
                   {"best_epoch": 20, "val_rec": 3.0, "val_kl": 2.0, "active_dims": 2, "sigma": None}]
               for r in detect.RUNS}
        fit["learn_sigma"][0]["sigma"], fit["learn_sigma"][1]["sigma"] = 0.1, 0.3
        rows = {r["run"]: r for r in fit_rows(fit)}
        self.assertEqual(rows["beta1"]["best_epoch"], 15)
        self.assertEqual(rows["beta1"]["sigma"], "")
        self.assertAlmostEqual(rows["learn_sigma"]["sigma"], 0.2)
        self.assertIn("beta1,15,2,1,1,\n", render(detect.FIT, [rows["beta1"]]))


if __name__ == "__main__":
    unittest.main()
