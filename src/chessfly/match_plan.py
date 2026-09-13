"""Turn a recorded run into a frame-accurate animation plan for the 3D scene.

All chess reasoning happens here, where python-chess is available: piece
identity, captures, castling, en passant and promotion are resolved into flat
events with frame numbers.  The Blender script only applies keyframes, and both
the 3D board and the overlay panel step on the same pacing curve, so they cannot
drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import chess
import numpy as np

from .match_video import _pacing


PLAN_SCHEMA_VERSION = 2
ESTABLISHING_FRAMES = 45
TRAVEL_SHARE = 0.45
MINIMUM_TRAVEL_FRAMES = 5

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def build_plan(
    run_dir: Path,
    duration_seconds: float = 60.0,
    fps: int = 30,
    highlight_plies: Sequence[int] = (),
) -> dict:
    """Resolve every ply into piece moves, captures and promotions with frames.

    `highlight_plies` are one-based Chessfly plies that get a "think" shot: the
    camera turns to the brain and the recorded decision plays out in full.
    """
    if duration_seconds <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    run_dir = Path(run_dir)
    records = [
        json.loads(line)
        for line in (run_dir / "moves.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("the run holds no recorded plies to animate")
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    total_frames = max(1, round(duration_seconds * fps))
    by_ply = {int(record["ply"]): index for index, record in enumerate(records)}
    highlights = sorted(set(int(ply) for ply in highlight_plies))
    for ply in highlights:
        if ply not in by_ply:
            raise ValueError(f"highlight ply {ply} is not in this run")
        if records[by_ply[ply]]["actor"] != "chessfly":
            raise ValueError(f"highlight ply {ply} is a Stockfish move, not a decision")
    weights = _pacing(len(records), [by_ply[ply] for ply in highlights])
    bounds = np.concatenate(([0.0], weights)) * total_frames

    board = chess.Board()
    occupancy: dict[int, str] = {}
    pieces = []
    for square, piece in board.piece_map().items():
        identifier = f"p{len(pieces):02d}"
        occupancy[square] = identifier
        pieces.append(
            {
                "id": identifier,
                "kind": PIECE_NAMES[piece.piece_type],
                "color": "white" if piece.color == chess.WHITE else "black",
                "square": chess.square_name(square),
            }
        )

    events: list[dict] = []
    plies: list[dict] = []
    for index, record in enumerate(records):
        start = int(round(bounds[index]))
        end = int(round(bounds[index + 1]))
        travel = max(MINIMUM_TRAVEL_FRAMES, int(round((end - start) * TRAVEL_SHARE)))
        landing = min(end, start + travel)
        move = chess.Move.from_uci(str(record["move_uci"]))
        if move not in board.legal_moves:
            raise ValueError(f"recorded move {move.uci()} is illegal at ply {record['ply']}")

        captured_square = None
        if board.is_en_passant(move):
            direction = -8 if board.turn == chess.WHITE else 8
            captured_square = move.to_square + direction
        elif board.piece_at(move.to_square) is not None:
            captured_square = move.to_square
        if captured_square is not None:
            events.append(
                {
                    "kind": "capture",
                    "frame": landing,
                    "piece": occupancy.pop(captured_square),
                }
            )

        mover = occupancy.pop(move.from_square)
        events.append(
            {
                "kind": "move",
                "piece": mover,
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "lift_frame": start,
                "land_frame": landing,
            }
        )
        occupancy[move.to_square] = mover

        if board.is_castling(move):
            back_rank = 0 if board.turn == chess.WHITE else 7
            king_side = chess.square_file(move.to_square) > 4
            rook_from = chess.square(7 if king_side else 0, back_rank)
            rook_to = chess.square(5 if king_side else 3, back_rank)
            rook = occupancy.pop(rook_from)
            events.append(
                {
                    "kind": "move",
                    "piece": rook,
                    "from": chess.square_name(rook_from),
                    "to": chess.square_name(rook_to),
                    "lift_frame": start,
                    "land_frame": landing,
                }
            )
            occupancy[rook_to] = rook

        if move.promotion is not None:
            events.append(
                {
                    "kind": "promote",
                    "frame": landing,
                    "piece": mover,
                    "to": PIECE_NAMES[move.promotion],
                }
            )

        plies.append(
            {
                "ply": int(record["ply"]),
                "actor": str(record["actor"]),
                "move_uci": move.uci(),
                "move_san": str(record["move_san"]),
                "start_frame": start,
                "land_frame": landing,
                "end_frame": end,
                "think": int(record["ply"]) in highlights,
            }
        )
        board.push(move)

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_run": str(run_dir.resolve()),
        "fps": int(fps),
        "duration_seconds": float(duration_seconds),
        "total_frames": total_frames,
        "result": manifest["result"],
        "termination": manifest["termination"],
        "stockfish_elo": int(manifest["stockfish"]["elo"]),
        "chessfly_mode": str(manifest["mode"]),
        "headline_spikes": _headline_spikes(run_dir),
        "highlight_plies": highlights,
        "shots": build_shots(plies, total_frames),
        "pieces": pieces,
        "events": events,
        "plies": plies,
    }


def build_shots(plies: Sequence[dict], total_frames: int) -> list[dict]:
    """Cut the timeline into camera shots, merging neighbours that share a shot.

    An establishing overview opens the film, highlighted decisions turn to the
    brain, the final ply holds on the result, and everything else watches the
    board.
    """
    spans: list[dict] = []

    def add(name: str, start: int, end: int, ply: int | None) -> None:
        if end <= start:
            return
        if spans and spans[-1]["name"] == name and name != "think":
            spans[-1]["end_frame"] = end
            return
        spans.append({"name": name, "start_frame": start, "end_frame": end, "ply": ply})

    for index, ply in enumerate(plies):
        start, end = int(ply["start_frame"]), int(ply["end_frame"])
        if index == len(plies) - 1:
            name = "result"
        elif ply["think"]:
            name = "think"
        else:
            name = "board"
        if index == 0:
            cut = min(end, start + ESTABLISHING_FRAMES)
            add("overview", start, cut, None)
            start = cut
        add(name, start, end, int(ply["ply"]) if name == "think" else None)
    if spans:
        spans[-1]["end_frame"] = total_frames
    return spans


def _headline_spikes(run_dir: Path) -> int:
    """Read the first recorded decision so the 3D monitor quotes this run."""
    artifacts = sorted((run_dir / "neural").glob("ply-*/runtime.json"))
    if not artifacts:
        return 0
    return int(json.loads(artifacts[0].read_text(encoding="utf-8"))["total_spikes"])


def write_plan(
    run_dir: Path,
    output: Path,
    duration_seconds: float = 60.0,
    fps: int = 30,
    highlight_plies: Sequence[int] = (),
) -> Path:
    plan = build_plan(
        run_dir,
        duration_seconds=duration_seconds,
        fps=fps,
        highlight_plies=highlight_plies,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
