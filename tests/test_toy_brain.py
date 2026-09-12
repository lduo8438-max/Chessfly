import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess

from chessfly.toy_brain import ToyChessBrain


class ToyChessBrainTests(unittest.TestCase):
    def test_decision_runs_through_lif_and_is_legal(self):
        board = chess.Board()
        decision, frames = ToyChessBrain().decide(board)
        self.assertEqual(len(frames), 3)
        self.assertTrue(any(spike >= 768 for frame in frames for spike in frame))
        self.assertIn(chess.Move.from_uci(decision.selected_uci), board.legal_moves)


if __name__ == "__main__":
    unittest.main()

