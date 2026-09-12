"""Deterministic 320x180 chess stimulus rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess
from PIL import Image, ImageDraw, ImageFont


FRAME_SIZE = (320, 180)
BOARD_PIXELS = 176
CELL_PIXELS = 22
BOARD_ORIGIN = (72, 2)

PIECE_LABELS = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def render_board_stimulus(
    board: chess.Board,
    perspective: chess.Color = chess.WHITE,
    last_move: Optional[chess.Move] = None,
) -> Image.Image:
    """Render the exact RGB stimulus intended for visual-neuron mapping."""
    image = Image.new("RGB", FRAME_SIZE, "#05080d")
    draw = ImageDraw.Draw(image)
    x0, y0 = BOARD_ORIGIN
    accent = "#62e8ff" if board.turn == chess.WHITE else "#ffb454"
    draw.rectangle(
        (x0 - 2, y0 - 2, x0 + BOARD_PIXELS + 1, y0 + BOARD_PIXELS + 1),
        outline=accent,
        width=2,
    )
    font = _piece_font()

    for display_rank in range(8):
        for display_file in range(8):
            if perspective == chess.WHITE:
                file_index = display_file
                rank_index = 7 - display_rank
            else:
                file_index = 7 - display_file
                rank_index = display_rank
            square = chess.square(file_index, rank_index)
            left = x0 + display_file * CELL_PIXELS
            top = y0 + display_rank * CELL_PIXELS
            light = (display_file + display_rank) % 2 == 0
            fill = "#aeb9b7" if light else "#243b42"
            draw.rectangle(
                (left, top, left + CELL_PIXELS - 1, top + CELL_PIXELS - 1),
                fill=fill,
            )
            if last_move is not None and square in (
                last_move.from_square,
                last_move.to_square,
            ):
                draw.rectangle(
                    (left + 1, top + 1, left + CELL_PIXELS - 2, top + CELL_PIXELS - 2),
                    outline="#d9ff66",
                    width=2,
                )
            piece = board.piece_at(square)
            if piece is not None:
                _draw_piece(draw, left, top, piece, font)
    return image


def save_board_stimulus(
    board: chess.Board,
    path: Path,
    perspective: chess.Color = chess.WHITE,
    last_move: Optional[chess.Move] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    render_board_stimulus(board, perspective, last_move).save(path, format="PNG")


def _draw_piece(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    piece: chess.Piece,
    font: ImageFont.ImageFont,
) -> None:
    margin = 3
    bounds = (
        left + margin,
        top + margin,
        left + CELL_PIXELS - margin - 1,
        top + CELL_PIXELS - margin - 1,
    )
    if piece.color == chess.WHITE:
        disk, outline, text = "#f2eee2", "#121820", "#111820"
    else:
        disk, outline, text = "#111820", "#d6e2df", "#eef8f5"
    draw.ellipse(bounds, fill=disk, outline=outline, width=1)
    label = PIECE_LABELS[piece.piece_type]
    text_box = draw.textbbox((0, 0), label, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.text(
        (
            left + (CELL_PIXELS - width) / 2,
            top + (CELL_PIXELS - height) / 2 - text_box[1],
        ),
        label,
        fill=text,
        font=font,
    )


def _piece_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except OSError:
        return ImageFont.load_default()

