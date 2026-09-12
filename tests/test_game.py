import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import chess.engine
import numpy as np

from chessfly.chess_control import CHANNELS, NeuralDecision
from chessfly.game import play_game
from chessfly.stockfish import StockfishConfig


SILENT_RATES = {channel: 0.0 for channel in CHANNELS}


class FakeStockfish:
    """A scripted opponent with the surface that play_game relies on."""

    def __init__(self, moves=None, evaluation=0):
        self.config = StockfishConfig()
        self.restarts = 0
        self.scripted = list(moves or [])
        self.evaluation = evaluation
        self.played = 0

    def evaluate_cp(self, board, pov):
        return self.evaluation

    def play(self, board):
        if self.scripted:
            move = chess.Move.from_uci(self.scripted[self.played % len(self.scripted)])
        else:
            move = sorted(board.legal_moves, key=lambda value: value.uci())[0]
        self.played += 1
        return move

    def restart(self):
        self.restarts += 1


class CountingBrain:
    """A stateful stand-in whose choices depend on how many decisions it has made."""

    mode = "fake-counting-brain"
    display_name = "Fake counting brain"
    window_ms = 1.0
    episode_seed = 7

    def __init__(self, scripted=None):
        self.counter = 0
        self.scripted = list(scripted or [])
        self.output_neuron_indices = frozenset({0, 1, 2})
        self.exported = []

    def decide(self, board):
        self.counter += 1
        legal = sorted(move.uci() for move in board.legal_moves)
        if self.scripted:
            chosen = self.scripted[(self.counter - 1) % len(self.scripted)]
        else:
            chosen = legal[self.counter % len(legal)]
        scores = {uci: (1.0 if uci == chosen else 0.0) for uci in legal}
        decision = NeuralDecision(chosen, scores, SILENT_RATES, False, False, "selected")
        return decision, [(0,), (1,)]

    def describe(self):
        return {"mode": self.mode, "decisions": self.counter}

    def state_dict(self):
        return {"counter": np.asarray(self.counter, dtype=np.int64)}

    def load_state_dict(self, state):
        self.counter = int(np.asarray(state["counter"]).reshape(()))

    def export_decision_artifact(self, run_dir, ply):
        target = Path(run_dir) / "neural" / f"ply-{ply:03d}"
        target.mkdir(parents=True, exist_ok=False)
        (target / "runtime.json").write_text(json.dumps({"ply": ply}), encoding="utf-8")
        self.exported.append(ply)


class CrashingStockfish(FakeStockfish):
    """Fails hard once a chosen ply is reached, standing in for a killed process."""

    def __init__(self, crash_at_ply, **kwargs):
        super().__init__(**kwargs)
        self.crash_at_ply = crash_at_ply
        self.calls = 0

    def evaluate_cp(self, board, pov):
        self.calls += 1
        if len(board.move_stack) >= self.crash_at_ply:
            raise KeyboardInterrupt("simulated interruption")
        return self.evaluation


class FlakyStockfish(FakeStockfish):
    """Raises engine errors a fixed number of times before recovering."""

    def __init__(self, failures, **kwargs):
        super().__init__(**kwargs)
        self.failures = failures

    def evaluate_cp(self, board, pov):
        if self.failures:
            self.failures -= 1
            raise chess.engine.EngineError("simulated engine failure")
        return self.evaluation


class GameCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def _lines(self, run_dir):
        return (run_dir / "moves.jsonl").read_text(encoding="utf-8").splitlines()

    def test_every_ply_is_flushed_before_the_game_ends(self):
        run_dir = self.root / "incremental"
        brain = CountingBrain()
        seen = []

        class ObservingStockfish(FakeStockfish):
            def play(inner, board):
                seen.append(len((run_dir / "moves.jsonl").read_text().splitlines()))
                return FakeStockfish.play(inner, board)

        result = play_game(brain, ObservingStockfish(), run_dir, max_plies=8)
        self.assertEqual(result.plies, 8)
        # Stockfish moves on even plies, so each call must already see the odd ply.
        self.assertEqual(seen, [1, 3, 5, 7])
        self.assertEqual(len(self._lines(run_dir)), 8)

    def test_progress_and_manifest_track_the_live_position(self):
        run_dir = self.root / "progress"
        play_game(CountingBrain(), FakeStockfish(), run_dir, max_plies=6)
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        board = chess.Board()
        for uci in progress["move_stack"]:
            board.push(chess.Move.from_uci(uci))
        self.assertEqual(progress["fen"], board.fen())
        self.assertEqual(progress["plies_completed"], 6)
        self.assertEqual(manifest["plies"], 6)
        self.assertEqual(manifest["termination"], "MAX_PLIES")
        self.assertTrue(progress["neural_checkpoint"])
        self.assertTrue((run_dir / "checkpoint.npz").is_file())

    def test_interrupted_run_resumes_to_the_same_game(self):
        reference_dir = self.root / "reference"
        reference = play_game(CountingBrain(), FakeStockfish(), reference_dir, max_plies=12)

        run_dir = self.root / "resumed"
        with self.assertRaises(KeyboardInterrupt):
            play_game(
                CountingBrain(), CrashingStockfish(crash_at_ply=5), run_dir, max_plies=12
            )
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["termination"], "IN_PROGRESS")
        # The crash landed inside ply 5, so only the four durable plies survive.
        self.assertEqual(progress["plies_completed"], 4)

        resumed = play_game(
            CountingBrain(), FakeStockfish(), run_dir, max_plies=12, resume=True
        )
        self.assertEqual(resumed.resumed_from_ply, 5)
        self.assertEqual(resumed.plies, reference.plies)
        self.assertEqual(
            (run_dir / "game.pgn").read_text(encoding="utf-8"),
            (reference_dir / "game.pgn").read_text(encoding="utf-8"),
        )
        self.assertEqual(self._lines(run_dir), self._lines(reference_dir))

    def test_resume_restores_brain_state_rather_than_resetting_it(self):
        run_dir = self.root / "brain-state"
        with self.assertRaises(KeyboardInterrupt):
            play_game(
                CountingBrain(), CrashingStockfish(crash_at_ply=5), run_dir, max_plies=12
            )
        brain = CountingBrain()
        play_game(brain, FakeStockfish(), run_dir, max_plies=12, resume=True)
        # Plies 1 and 3 were durable brain decisions; it must not restart from zero.
        self.assertEqual(brain.counter, 6)

    def test_resume_truncates_records_written_past_the_checkpoint(self):
        run_dir = self.root / "torn-write"
        with self.assertRaises(KeyboardInterrupt):
            play_game(
                CountingBrain(), CrashingStockfish(crash_at_ply=5), run_dir, max_plies=12
            )
        with (run_dir / "moves.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ply": 6, "actor": "torn"}) + "\n")
        play_game(CountingBrain(), FakeStockfish(), run_dir, max_plies=12, resume=True)
        records = [json.loads(line) for line in self._lines(run_dir)]
        self.assertEqual([record["ply"] for record in records], list(range(1, 13)))
        self.assertNotIn("torn", [record["actor"] for record in records])

    def test_resume_rejects_a_different_controller(self):
        run_dir = self.root / "mismatch"
        with self.assertRaises(KeyboardInterrupt):
            play_game(
                CountingBrain(), CrashingStockfish(crash_at_ply=3), run_dir, max_plies=12
            )
        other = CountingBrain()
        other.mode = "some-other-brain"
        with self.assertRaises(ValueError):
            play_game(other, FakeStockfish(), run_dir, max_plies=12, resume=True)

    def test_resume_rejects_a_run_that_reached_a_rule_termination(self):
        run_dir = self.root / "finished"
        finished = play_game(
            CountingBrain(scripted=["f2f3", "g2g4"]),
            FakeStockfish(moves=["e7e5", "d8h4"]),
            run_dir,
            max_plies=40,
        )
        self.assertEqual(finished.termination, "CHECKMATE")
        with self.assertRaises(ValueError):
            play_game(CountingBrain(), FakeStockfish(), run_dir, max_plies=60, resume=True)

    def test_a_capped_run_continues_under_a_larger_cap(self):
        run_dir = self.root / "extended"
        capped = play_game(CountingBrain(), FakeStockfish(), run_dir, max_plies=6)
        self.assertEqual(capped.termination, "MAX_PLIES")
        with self.assertRaises(ValueError):
            play_game(CountingBrain(), FakeStockfish(), run_dir, max_plies=6, resume=True)
        extended = play_game(
            CountingBrain(), FakeStockfish(), run_dir, max_plies=10, resume=True
        )
        self.assertEqual(extended.plies, 10)
        self.assertEqual(extended.resumed_from_ply, 7)
        records = [json.loads(line) for line in self._lines(run_dir)]
        self.assertEqual([record["ply"] for record in records], list(range(1, 11)))

    def test_stale_artifacts_are_replaced_when_a_ply_is_recomputed(self):
        run_dir = self.root / "stale"
        with self.assertRaises(KeyboardInterrupt):
            play_game(
                CountingBrain(), CrashingStockfish(crash_at_ply=5), run_dir, max_plies=12
            )
        # The brain exported ply 5 before the crash, but that ply was never durable.
        stale = run_dir / "neural" / "ply-005"
        self.assertTrue(stale.is_dir())
        (stale / "partial.txt").write_text("truncated", encoding="utf-8")
        play_game(CountingBrain(), FakeStockfish(), run_dir, max_plies=12, resume=True)
        self.assertFalse((stale / "partial.txt").exists())
        self.assertTrue((stale / "decision.json").is_file())

    def test_each_neural_decision_is_recorded(self):
        run_dir = self.root / "decisions"
        play_game(CountingBrain(), FakeStockfish(), run_dir, max_plies=6)
        for ply in (1, 3, 5):
            payload = json.loads(
                (run_dir / "neural" / f"ply-{ply:03d}" / "decision.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["ply"], ply)
            self.assertIn(payload["selected_uci"], payload["move_scores"])
            self.assertEqual(payload["window_ms"], 1.0)
            self.assertEqual(payload["episode_seed"], 7)
            self.assertEqual(payload["readout_spike_count"], 2)
            self.assertEqual(len(payload["channel_rates_hz"]), len(CHANNELS))
        self.assertFalse((run_dir / "neural" / "ply-002").exists())

    def test_engine_failures_restart_stockfish_instead_of_ending_the_game(self):
        run_dir = self.root / "flaky"
        stockfish = FlakyStockfish(failures=2)
        result = play_game(
            CountingBrain(), stockfish, run_dir, max_plies=4, engine_retries=3
        )
        self.assertEqual(stockfish.restarts, 2)
        self.assertEqual(result.engine_restarts, 2)
        self.assertEqual(result.plies, 4)

    def test_engine_failures_beyond_the_retry_budget_stop_the_run(self):
        run_dir = self.root / "dead-engine"
        with self.assertRaises(RuntimeError):
            play_game(
                CountingBrain(),
                FlakyStockfish(failures=10),
                run_dir,
                max_plies=4,
                engine_retries=2,
            )
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["termination"], "IN_PROGRESS")

    def test_checkmate_is_reported_as_a_rule_termination(self):
        run_dir = self.root / "mate"
        result = play_game(
            CountingBrain(scripted=["f2f3", "g2g4"]),
            FakeStockfish(moves=["e7e5", "d8h4"]),
            run_dir,
            max_plies=40,
        )
        self.assertEqual(result.termination, "CHECKMATE")
        self.assertEqual(result.result, "0-1")
        self.assertEqual(result.plies, 4)

    def test_a_repeating_decision_loop_ends_as_a_claimed_draw(self):
        run_dir = self.root / "repetition"
        result = play_game(
            CountingBrain(scripted=["b1c3", "c3b1"]),
            FakeStockfish(moves=["b8c6", "c6b8"]),
            run_dir,
            max_plies=40,
        )
        self.assertEqual(result.termination, "THREEFOLD_REPETITION")
        self.assertEqual(result.result, "1/2-1/2")
        self.assertLess(result.plies, 40)

    def test_a_fresh_run_refuses_to_overwrite_an_existing_directory(self):
        run_dir = self.root / "occupied"
        run_dir.mkdir()
        with self.assertRaises(FileExistsError):
            play_game(CountingBrain(), FakeStockfish(), run_dir, max_plies=2)


if __name__ == "__main__":
    unittest.main()
