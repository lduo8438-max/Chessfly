import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly.benchmark import POSITION_LINES, fixed_positions


class BenchmarkTests(unittest.TestCase):
    def test_all_frozen_positions_are_live_and_legal(self):
        boards = fixed_positions()
        self.assertEqual(len(boards), len(POSITION_LINES))
        self.assertTrue(all(not board.is_game_over() for board in boards))
        self.assertTrue(all(any(board.legal_moves) for board in boards))


if __name__ == "__main__":
    unittest.main()
