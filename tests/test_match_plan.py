import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess

from chessfly.match_plan import build_plan


def write_run(root: Path, moves: list[str]) -> Path:
    """Lay out the minimum run directory that an animation plan needs."""
    run_dir = Path(root)
    run_dir.mkdir(parents=True, exist_ok=True)
    board = chess.Board()
    lines = []
    for index, uci in enumerate(moves, start=1):
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        board.push(move)
        lines.append(
            json.dumps(
                {
                    "ply": index,
                    "actor": "chessfly" if index % 2 else "stockfish",
                    "move_uci": uci,
                    "move_san": san,
                }
            )
        )
    (run_dir / "moves.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "result": "*",
                "termination": "MAX_PLIES",
                "mode": "male-cns-subgraph-v1",
                "stockfish": {"elo": 1320},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


class MatchPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def _plan(self, moves, duration=60.0, fps=30):
        return build_plan(write_run(self.root / "run", moves), duration, fps)

    def test_opening_position_is_fully_described(self):
        plan = self._plan(["e2e4", "e7e5"])
        self.assertEqual(len(plan["pieces"]), 32)
        self.assertEqual(len({piece["id"] for piece in plan["pieces"]}), 32)
        squares = {piece["square"] for piece in plan["pieces"]}
        self.assertIn("e1", squares)
        self.assertEqual(sum(p["kind"] == "pawn" for p in plan["pieces"]), 16)

    def test_frames_cover_the_timeline_without_gaps(self):
        plan = self._plan(["e2e4", "e7e5", "g1f3", "b8c6"], duration=20.0, fps=30)
        plies = plan["plies"]
        self.assertEqual(plies[0]["start_frame"], 0)
        self.assertEqual(plies[-1]["end_frame"], plan["total_frames"])
        for earlier, later in zip(plies, plies[1:]):
            self.assertEqual(earlier["end_frame"], later["start_frame"])
        for ply in plies:
            self.assertLess(ply["start_frame"], ply["land_frame"])
            self.assertLessEqual(ply["land_frame"], ply["end_frame"])

    def test_a_capture_removes_the_piece_that_stood_there(self):
        plan = self._plan(["e2e4", "d7d5", "e4d5"])
        captures = [event for event in plan["events"] if event["kind"] == "capture"]
        self.assertEqual(len(captures), 1)
        black_d_pawn = next(
            piece["id"] for piece in plan["pieces"] if piece["square"] == "d7"
        )
        self.assertEqual(captures[0]["piece"], black_d_pawn)

    def test_castling_also_moves_the_rook(self):
        plan = self._plan(
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1"]
        )
        rook = next(piece["id"] for piece in plan["pieces"] if piece["square"] == "h1")
        king = next(piece["id"] for piece in plan["pieces"] if piece["square"] == "e1")
        moves = [event for event in plan["events"] if event["kind"] == "move"]
        rook_move = next(event for event in moves if event["piece"] == rook)
        king_move = next(event for event in moves if event["piece"] == king)
        self.assertEqual((rook_move["from"], rook_move["to"]), ("h1", "f1"))
        self.assertEqual((king_move["from"], king_move["to"]), ("e1", "g1"))
        self.assertEqual(rook_move["land_frame"], king_move["land_frame"])

    def test_en_passant_removes_the_pawn_beside_the_target(self):
        plan = self._plan(["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"])
        captured = next(
            event["piece"] for event in plan["events"] if event["kind"] == "capture"
        )
        black_d_pawn = next(
            piece["id"] for piece in plan["pieces"] if piece["square"] == "d7"
        )
        self.assertEqual(captured, black_d_pawn)

    def test_promotions_are_recorded_for_both_sides(self):
        plan = self._plan(
            [
                "h2h4", "a7a5",
                "h4h5", "a5a4",
                "h5h6", "a4a3",
                "h6g7", "a3b2",
                "g7h8q", "b2a1q",
            ]
        )
        promotions = [event for event in plan["events"] if event["kind"] == "promote"]
        self.assertEqual([event["to"] for event in promotions], ["queen", "queen"])
        white_h_pawn = next(
            piece["id"] for piece in plan["pieces"] if piece["square"] == "h2"
        )
        self.assertEqual(promotions[0]["piece"], white_h_pawn)

    def test_an_illegal_recorded_move_is_refused(self):
        run_dir = write_run(self.root / "bad", ["e2e4"])
        (run_dir / "moves.jsonl").write_text(
            json.dumps(
                {"ply": 1, "actor": "chessfly", "move_uci": "e2e5", "move_san": "e5"}
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            build_plan(run_dir, 20.0, 30)

    def test_an_empty_run_is_refused(self):
        run_dir = write_run(self.root / "empty", ["e2e4"])
        (run_dir / "moves.jsonl").write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_plan(run_dir, 20.0, 30)

    def test_highlights_must_be_neural_decisions_in_the_run(self):
        run_dir = write_run(self.root / "highlights", ["e2e4", "e7e5", "g1f3", "b8c6"])
        with self.assertRaises(ValueError):
            build_plan(run_dir, 20.0, 30, highlight_plies=[2])
        with self.assertRaises(ValueError):
            build_plan(run_dir, 20.0, 30, highlight_plies=[9])

    def test_shots_cover_the_timeline_and_turn_to_the_brain_on_highlights(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "d2d3", "d7d6"]
        plan = build_plan(
            write_run(self.root / "shots", moves), 30.0, 30, highlight_plies=[3, 7]
        )
        shots = plan["shots"]
        self.assertEqual(shots[0]["name"], "overview")
        self.assertEqual(shots[0]["start_frame"], 0)
        self.assertEqual(shots[-1]["name"], "result")
        self.assertEqual(shots[-1]["end_frame"], plan["total_frames"])
        for earlier, later in zip(shots, shots[1:]):
            self.assertEqual(earlier["end_frame"], later["start_frame"])
        self.assertEqual([s["ply"] for s in shots if s["name"] == "think"], [3, 7])
        think = {p["ply"] for p in plan["plies"] if p["think"]}
        self.assertEqual(think, {3, 7})

    def test_highlighted_plies_are_given_more_time(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "d2d3", "d7d6", "c2c3", "g8f6"]
        plan = build_plan(write_run(self.root / "time", moves), 30.0, 30, highlight_plies=[5])
        spans = {p["ply"]: p["end_frame"] - p["start_frame"] for p in plan["plies"]}
        self.assertGreater(spans[5], 3 * spans[3])

    def test_every_event_names_a_real_piece(self):
        plan = self._plan(["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"])
        identifiers = {piece["id"] for piece in plan["pieces"]}
        for event in plan["events"]:
            self.assertIn(event["piece"], identifiers)


if __name__ == "__main__":
    unittest.main()
