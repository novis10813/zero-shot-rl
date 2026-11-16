"""The history window of zero_shot_rl.eval.adapt_episode on a known trajectory."""

import unittest

import numpy as np

from zero_shot_rl.eval.adapt_episode import window


class WindowTest(unittest.TestCase):
    def test_window_holds_the_h_transitions_before_the_step(self):
        states = np.arange(6, dtype=np.float32).reshape(1, 6, 1) ** 2  # s_0 .. s_5
        actions = np.arange(5, dtype=np.float32).reshape(1, 5, 1)
        s, a, d = window(states, actions, t=4, h=3)  # transitions 1, 2, 3
        np.testing.assert_array_equal(s[0, :, 0], [1, 4, 9])
        np.testing.assert_array_equal(a[0, :, 0], [1, 2, 3])
        np.testing.assert_array_equal(d[0, :, 0], [3, 5, 7])


if __name__ == "__main__":
    unittest.main()
