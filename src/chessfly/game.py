"""Run a logged Chessfly-versus-Stockfish game."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol

import chess
import chess.pgn

from .stockfish import StockfishOpponent
from .reward import stockfish_shaped_reward

from .chess_control import NeuralDecision


class ChessBrain(Protocol):
    mode: str
    display_name: str
    output_neuron_indices: frozenset[int]

    def decide(
        self, board: chess.Board
    ) -> tuple[NeuralDecision, list[tuple[int, ...]]]: ...

    def describe(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class MoveRecord:
    ply: int
    actor: str
    move_uci: str
    move_san: str
    fen_before: str
    fen_after: str
    evaluation_before_cp: int
    evaluation_after_cp: int
    shaped_reward: float
    reward_valence: str
    neural_score: Optional[float]
    silent_decision: bool
    tie_break: bool
    output_spikes: tuple[int, ...]


@dataclass(frozen=True)
class GameResult:
    result: str
    termination: str
    plies: int
    run_dir: Path


def play_game(
    brain: ChessBrain,
    stockfish: StockfishOpponent,
    run_dir: Path,
    chessfly_color: chess.Color = chess.WHITE,
    max_plies: int = 40,
) -> GameResult:
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    run_dir.mkdir(parents=True, exist_ok=False)
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Chessfly versus Stockfish"
    game.headers["White"] = brain.display_name if chessfly_color else "Stockfish"
    game.headers["Black"] = "Stockfish" if chessfly_color else brain.display_name
    game.headers["ChessflyMode"] = brain.mode
    node = game
    records = []

    for ply in range(1, max_plies + 1):
        if board.is_game_over(claim_draw=True):
            break
        fen_before = board.fen()
        evaluation_before = stockfish.evaluate_cp(board, chessfly_color)
        if board.turn == chessfly_color:
            decision, frames = brain.decide(board)
            if decision.selected_uci is None:
                break
            move = chess.Move.from_uci(decision.selected_uci)
            neural_score = decision.move_scores[decision.selected_uci]
            silent = decision.silent
            tie_break = decision.tie_break
            output_spikes = tuple(
                spike
                for frame in frames
                for spike in frame
                if spike in brain.output_neuron_indices
            )
            exporter = getattr(brain, "export_decision_artifact", None)
            if exporter is not None:
                exporter(run_dir, ply)
            actor = "chessfly"
        else:
            move = stockfish.play(board)
            neural_score = None
            silent = False
            tie_break = False
            output_spikes = ()
            actor = "stockfish"

        san = board.san(move)
        node = node.add_variation(move)
        board.push(move)
        evaluation = stockfish.evaluate_cp(board, chessfly_color)
        terminal_result = None
        if board.is_game_over(claim_draw=True):
            outcome_now = board.outcome(claim_draw=True)
            if outcome_now is not None and outcome_now.winner is None:
                terminal_result = "draw"
            elif outcome_now is not None and outcome_now.winner == chessfly_color:
                terminal_result = "win"
            else:
                terminal_result = "loss"
        reward = (
            stockfish_shaped_reward(evaluation_before, evaluation, terminal_result)
            if actor == "chessfly"
            else stockfish_shaped_reward(evaluation, evaluation)
        )
        records.append(
            MoveRecord(
                ply,
                actor,
                move.uci(),
                san,
                fen_before,
                board.fen(),
                evaluation_before,
                evaluation,
                reward.value,
                reward.valence,
                neural_score,
                silent,
                tie_break,
                output_spikes,
            )
        )

    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else "*"
    termination = outcome.termination.name if outcome else "MAX_PLIES"
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    (run_dir / "game.pgn").write_text(str(game), encoding="utf-8")
    with (run_dir / "moves.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "mode": brain.mode,
        "brain": brain.describe(),
        "result": result,
        "termination": termination,
        "plies": len(records),
        "chessfly_color": "white" if chessfly_color else "black",
        "stockfish": asdict(stockfish.config),
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return GameResult(result, termination, len(records), run_dir)
