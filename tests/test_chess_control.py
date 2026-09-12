import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess

from chessfly.chess_control import (
    CHANNELS,
    ChessMoveDecoder,
    assign_readout_populations,
)


class ChessMoveDecoderTests(unittest.TestCase):
    def setUp(self):
        self.populations = {channel: (index,) for index, channel in enumerate(CHANNELS)}
        self.decoder = ChessMoveDecoder(self.populations)

    def test_assignment_is_balanced_and_stable(self):
        first = assign_readout_populations(range(100))
        second = assign_readout_populations(reversed(range(100)))
        self.assertEqual(first, second)
        sizes = [len(values) for values in first.values()]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_decoder_always_selects_legal_move(self):
        board = chess.Board()
        decision = self.decoder.decode(board, ((0, 8, 16, 24),), 100.0)
        self.assertIn(chess.Move.from_uci(decision.selected_uci), board.legal_moves)

    def test_silent_decision_is_explicit_and_deterministic(self):
        board = chess.Board()
        decision = self.decoder.decode(board, ((),), 100.0)
        self.assertTrue(decision.silent)
        self.assertTrue(decision.tie_break)
        self.assertEqual(decision.selected_uci, "a2a3")

    def test_promotion_tie_prefers_queen(self):
        board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        decision = self.decoder.decode(board, ((),), 100.0)
        promotion_moves = [move for move in board.legal_moves if move.from_square == chess.A7]
        self.assertTrue(promotion_moves)
        self.assertEqual(decision.selected_uci, "a7a8q")

    def test_game_over_returns_no_move(self):
        board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
        decision = self.decoder.decode(board, ((),), 100.0)
        self.assertIsNone(decision.selected_uci)
        self.assertEqual(decision.reason, "game_over")


if __name__ == "__main__":
    unittest.main()

