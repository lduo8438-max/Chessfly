"""Display-only chessboard art with Staunton piece glyphs.

This module exists so a video can show a readable board.  It is presentation
only: the neural stimulus stays in :mod:`chessfly.vision` and is never changed
to suit what a render should look like.  Pieces are drawn from a system Unicode
chess font rather than any third-party piece artwork.
"""

from __future__ import annotations

from typing import Optional, Sequence

import chess
from PIL import Image, ImageDraw, ImageFont


LIGHT_SQUARE = (201, 212, 209)
DARK_SQUARE = (58, 94, 104)
BORDER = (46, 83, 93)
LAST_MOVE = (217, 255, 102)
COORDINATE = (126, 150, 156)
WHITE_PIECE = (244, 241, 232)
WHITE_EDGE = (13, 19, 25)
BLACK_PIECE = (18, 25, 31)
BLACK_EDGE = (198, 214, 210)

PIECE_GLYPHS = {
    chess.PAWN: "♟",
    chess.KNIGHT: "♞",
    chess.BISHOP: "♝",
    chess.ROOK: "♜",
    chess.QUEEN: "♛",
    chess.KING: "♚",
}
PIECE_LABELS = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}
GLYPH_FONTS: Sequence[str] = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)
LABEL_FONTS: Sequence[str] = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "DejaVuSans-Bold.ttf",
)


def piece_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Return the first system font that actually carries the chess glyphs."""
    for path in GLYPH_FONTS:
        try:
            font = ImageFont.truetype(path, size)
        except OSError:
            continue
        if font.getmask(PIECE_GLYPHS[chess.KING]).getbbox() is not None:
            return font
    return None


def _label_font(size: int) -> ImageFont.ImageFont:
    for path in LABEL_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_display_board(
    board: chess.Board,
    size: int = 520,
    perspective: chess.Color = chess.WHITE,
    last_move: Optional[chess.Move] = None,
    coordinates: bool = True,
    light_square: tuple[int, int, int] = LIGHT_SQUARE,
    dark_square: tuple[int, int, int] = DARK_SQUARE,
    border_colour: tuple[int, int, int] = BORDER,
    highlight_colour: tuple[int, int, int] = LAST_MOVE,
) -> Image.Image:
    """Draw a readable board; falls back to letters where no chess font exists."""
    if size < 64:
        raise ValueError("display board must be at least 64 pixels wide")
    LIGHT, DARK = light_square, dark_square
    EDGE, MARK = border_colour, highlight_colour
    cell = size // 8
    span = cell * 8
    image = Image.new("RGB", (span, span), DARK)
    draw = ImageDraw.Draw(image, "RGBA")
    glyphs = piece_font(int(cell * 0.82))
    fallback = None if glyphs is not None else _label_font(int(cell * 0.5))
    coordinate_font = _label_font(max(8, int(cell * 0.22)))

    for display_rank in range(8):
        for display_file in range(8):
            if perspective == chess.WHITE:
                file_index, rank_index = display_file, 7 - display_rank
            else:
                file_index, rank_index = 7 - display_file, display_rank
            square = chess.square(file_index, rank_index)
            left = display_file * cell
            top = display_rank * cell
            light = (display_file + display_rank) % 2 == 0
            draw.rectangle(
                (left, top, left + cell - 1, top + cell - 1),
                fill=LIGHT if light else DARK,
            )
            if last_move is not None and square in (
                last_move.from_square,
                last_move.to_square,
            ):
                draw.rectangle(
                    (left, top, left + cell - 1, top + cell - 1),
                    fill=MARK + (62,),
                )
                draw.rectangle(
                    (left + 1, top + 1, left + cell - 2, top + cell - 2),
                    outline=MARK,
                    width=max(1, cell // 22),
                )
            if coordinates:
                if display_rank == 7:
                    draw.text(
                        (left + cell - max(3, cell // 14), top + cell - max(3, cell // 14)),
                        chess.FILE_NAMES[file_index],
                        font=coordinate_font,
                        fill=DARK if light else LIGHT,
                        anchor="rs",
                    )
                if display_file == 0:
                    draw.text(
                        (left + max(3, cell // 14), top + max(2, cell // 16)),
                        chess.RANK_NAMES[rank_index],
                        font=coordinate_font,
                        fill=DARK if light else LIGHT,
                        anchor="lt",
                    )
            piece = board.piece_at(square)
            if piece is not None:
                _draw_piece(draw, left, top, cell, piece, glyphs, fallback)

    draw.rectangle((0, 0, span - 1, span - 1), outline=EDGE, width=max(1, cell // 26))
    return image


def _draw_piece(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    cell: int,
    piece: chess.Piece,
    glyphs: Optional[ImageFont.FreeTypeFont],
    fallback: Optional[ImageFont.ImageFont],
) -> None:
    fill, edge = (
        (WHITE_PIECE, WHITE_EDGE)
        if piece.color == chess.WHITE
        else (BLACK_PIECE, BLACK_EDGE)
    )
    centre = (left + cell / 2, top + cell / 2)
    if glyphs is not None:
        draw.text(
            (centre[0], centre[1] + cell * 0.04),
            PIECE_GLYPHS[piece.piece_type],
            font=glyphs,
            fill=fill,
            anchor="mm",
            stroke_width=max(1, round(cell / 28)),
            stroke_fill=edge,
        )
        return
    margin = max(2, cell // 8)
    draw.ellipse(
        (left + margin, top + margin, left + cell - margin, top + cell - margin),
        fill=fill,
        outline=edge,
        width=max(1, cell // 22),
    )
    draw.text(
        centre,
        PIECE_LABELS[piece.piece_type],
        font=fallback,
        fill=edge,
        anchor="mm",
    )
