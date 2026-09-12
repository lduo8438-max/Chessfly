import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly.social_video import _evaluation


class SocialVideoTests(unittest.TestCase):
    def test_centipawn_formatting(self):
        self.assertEqual(_evaluation(36), "+0.36")
        self.assertEqual(_evaluation(-125), "-1.25")
        self.assertEqual(_evaluation(100000), "MATE")


if __name__ == "__main__":
    unittest.main()
