import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly.social_video import (
    VideoRun,
    _decision_summary,
    _evaluation,
    _outcome_lines,
    _repeat_count,
)


def _run(moves, **manifest):
    defaults = {
        "result": "*",
        "termination": "MAX_PLIES",
        "chessfly_color": "white",
    }
    defaults.update(manifest)
    return VideoRun(None, defaults, tuple(moves), (), (), ())


def _move(ply, actor="chessfly", uci="a2a3", san="a3", tie_break=True, spikes=()):
    return {
        "ply": ply,
        "actor": actor,
        "move_uci": uci,
        "move_san": san,
        "tie_break": tie_break,
        "output_spikes": list(spikes),
    }


class SocialVideoTests(unittest.TestCase):
    def test_centipawn_formatting(self):
        self.assertEqual(_evaluation(36), "+0.36")
        self.assertEqual(_evaluation(-125), "-1.25")
        self.assertEqual(_evaluation(100000), "MATE")
        self.assertEqual(_evaluation(-100000), "MATE")

    def test_forced_mates_read_as_mate_in_n_not_pawns(self):
        self.assertEqual(_evaluation(-99991), "-M9")
        self.assertEqual(_evaluation(99999), "M1")
        self.assertEqual(_evaluation(-99001), "-M999")
        self.assertEqual(_evaluation(-98999), "-989.99")

    def test_a_capped_run_is_still_called_a_demo(self):
        run = _run([_move(1), _move(2, actor="stockfish")])
        self.assertEqual(_outcome_lines(run), ("2-PLY DEMO", "STOPPED AT THE PLY CAP"))

    def test_a_loss_is_named_as_a_loss(self):
        run = _run(
            [_move(index) for index in range(1, 67)],
            result="0-1",
            termination="CHECKMATE",
        )
        self.assertEqual(
            _outcome_lines(run), ("66-PLY FULL GAME", "LOST BY CHECKMATE ON MOVE 33")
        )

    def test_a_win_and_a_draw_read_correctly(self):
        won = _run(
            [_move(index) for index in range(1, 6)],
            result="1-0",
            termination="CHECKMATE",
        )
        self.assertEqual(_outcome_lines(won)[1], "WON BY CHECKMATE ON MOVE 3")
        black = _run(
            [_move(index) for index in range(1, 6)],
            result="0-1",
            termination="CHECKMATE",
            chessfly_color="black",
        )
        self.assertEqual(_outcome_lines(black)[1], "WON BY CHECKMATE ON MOVE 3")
        drawn = _run(
            [_move(index) for index in range(1, 9)],
            result="1/2-1/2",
            termination="THREEFOLD_REPETITION",
        )
        self.assertEqual(
            _outcome_lines(drawn)[1], "DRAWN BY THREEFOLD REPETITION ON MOVE 4"
        )

    def test_decision_summary_counts_only_neural_decisions(self):
        run = _run(
            [
                _move(1, spikes=(10, 11)),
                _move(2, actor="stockfish", tie_break=False),
                _move(3, tie_break=False, spikes=(12,)),
                _move(4, actor="stockfish", tie_break=False),
            ]
        )
        decisions, tie_percent, mean_spikes = _decision_summary(run)
        self.assertEqual(decisions, 2)
        self.assertAlmostEqual(tie_percent, 50.0)
        self.assertAlmostEqual(mean_spikes, 1.5)

    def test_decision_summary_handles_a_run_with_no_neural_moves(self):
        run = _run([_move(1, actor="stockfish", tie_break=False)])
        self.assertEqual(_decision_summary(run), (0, 0.0, 0.0))

    def test_replayed_tie_broken_moves_are_counted(self):
        moves = [
            _move(1, uci="a1a2", san="Ra2"),
            _move(2, actor="stockfish", uci="e7e6", san="e6", tie_break=False),
            _move(3, uci="a2a1", san="Ra1"),
            _move(4, actor="stockfish", uci="b8c6", san="Nc6", tie_break=False),
            _move(5, uci="a1a2", san="Ra2"),
        ]
        run = _run(moves)
        self.assertEqual(_repeat_count(run, 0), 1)
        self.assertEqual(_repeat_count(run, 4), 2)

    def test_engine_and_spike_driven_moves_are_never_marked(self):
        moves = [
            _move(1, uci="a1a2", san="Ra2", tie_break=False),
            _move(2, actor="stockfish", uci="a7a6", san="a6", tie_break=False),
            _move(3, uci="a1a2", san="Ra2", tie_break=False),
            _move(4, actor="stockfish", uci="a7a6", san="a6", tie_break=False),
        ]
        run = _run(moves)
        self.assertEqual(_repeat_count(run, 2), 0)
        self.assertEqual(_repeat_count(run, 3), 0)


if __name__ == "__main__":
    unittest.main()
