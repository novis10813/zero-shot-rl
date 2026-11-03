"""Selection of the stored T05 rows appended by cql_sweep."""

import unittest

import numpy as np

from cql_sweep import stored_rows


class StoredRowsTests(unittest.TestCase):
    def test_rows(self):
        stored = {
            "task": np.array(["walk", "walk", "walk", "walk", "run", "walk"]),
            "family": np.array(["z_r", "CQL", "CQL", "ReBRAC", "z_r", "CQL"]),
            "target_angle": np.array([np.nan, 5.0, 10.0, 5.0, np.nan, 20.0]),
        }
        self.assertEqual(stored_rows(stored, ["walk"], [5, 10]), [0, 1, 2])
        self.assertEqual(stored_rows(stored, ["run", "walk"], [20]), [4, 0, 5])


if __name__ == "__main__":
    unittest.main()
