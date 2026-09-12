import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess

from chessfly import board_art
from chessfly.board_art import render_display_board
from chessfly.vision import render_board_stimulus


class BoardArtTests(unittest.TestCase):
    def test_board_is_square_and_deterministic(self):
        board = chess.Board()
        first = render_display_board(board, size=256)
        second = render_display_board(board, size=256)
        self.assertEqual(first.size, (256, 256))
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_size_must_stay_legible(self):
        with self.assertRaises(ValueError):
            render_display_board(chess.Board(), size=32)

    def test_last_move_is_highlighted(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        board.push(move)
        plain = render_display_board(board, size=256)
        marked = render_display_board(board, size=256, last_move=move)
        self.assertNotEqual(plain.tobytes(), marked.tobytes())

    def test_perspective_flips_the_board(self):
        board = chess.Board()
        white = render_display_board(board, size=256)
        black = render_display_board(board, size=256, perspective=chess.BLACK)
        self.assertNotEqual(white.tobytes(), black.tobytes())

    def test_letters_are_used_when_no_chess_font_exists(self):
        original = board_art.GLYPH_FONTS
        board_art.GLYPH_FONTS = ()
        try:
            self.assertIsNone(board_art.piece_font(40))
            image = render_display_board(chess.Board(), size=256)
        finally:
            board_art.GLYPH_FONTS = original
        self.assertEqual(image.size, (256, 256))

    def test_display_art_never_replaces_the_neural_stimulus(self):
        board = chess.Board()
        stimulus = render_board_stimulus(board)
        self.assertEqual(stimulus.size, (320, 180))
        self.assertNotEqual(stimulus.size, render_display_board(board, size=256).size)


if __name__ == "__main__":
    unittest.main()
