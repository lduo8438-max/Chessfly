import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess

from chessfly.vision import FRAME_SIZE, render_board_stimulus


class VisionTests(unittest.TestCase):
    def test_stimulus_has_fixed_rgb_geometry(self):
        image = render_board_stimulus(chess.Board())
        self.assertEqual(image.size, FRAME_SIZE)
        self.assertEqual(image.mode, "RGB")

    def test_perspective_and_last_move_change_pixels(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        white = render_board_stimulus(board, chess.WHITE)
        black = render_board_stimulus(board, chess.BLACK)
        highlighted = render_board_stimulus(board, chess.WHITE, move)
        self.assertNotEqual(white.tobytes(), black.tobytes())
        self.assertNotEqual(white.tobytes(), highlighted.tobytes())

    def test_render_is_deterministic(self):
        board = chess.Board()
        self.assertEqual(
            render_board_stimulus(board).tobytes(),
            render_board_stimulus(board).tobytes(),
        )


if __name__ == "__main__":
    unittest.main()
