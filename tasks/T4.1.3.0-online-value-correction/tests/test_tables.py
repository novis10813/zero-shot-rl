"""The tables of REPORT.md are rebuilt from data/inputs/."""

import unittest

from analyze import tables


class TestTables(unittest.TestCase):
    def test_tables_are_reproduced(self):
        self.assertEqual(tables.OUT.read_text(), tables.build())

    def test_half_up_rounds_halves_up(self):
        self.assertEqual([tables.half_up(x) for x in (612.5, 609.5, 539.2, 427.4)], [613, 610, 539, 427])


if __name__ == "__main__":
    unittest.main()
