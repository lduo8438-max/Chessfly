import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from chessfly.male_cns_brain import _intersect_body_ids


class MaleCNSBrainTests(unittest.TestCase):
    def test_intersection_preserves_requested_rows(self):
        rows, local = _intersect_body_ids(
            np.array([30, 10, 99, 20]), np.array([10, 20, 30, 40])
        )
        np.testing.assert_array_equal(rows, [0, 1, 3])
        np.testing.assert_array_equal(local, [2, 0, 1])


if __name__ == "__main__":
    unittest.main()
