"""Run a logged Chessfly-versus-Stockfish game with per-ply checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

import chess
import chess.engine
import chess.pgn
import numpy as np

from .stockfish import StockfishOpponent
from .reward import stockfish_shaped_reward

from .chess_control import NeuralDecision


PROGRESS_NAME = "progress.json"
CHECKPOINT_NAME = "checkpoint.npz"
MOVES_NAME = "moves.jsonl"
PGN_NAME = "game.pgn"
MANIFEST_NAME = "run.json"
PROGRESS_SCHEMA_VERSION = 1
IN_PROGRESS = "IN_PROGRESS"


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
    engine_restarts: int = 0
    resumed_from_ply: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez(handle, **state)
    os.replace(temporary, path)


def _call_engine(
    stockfish: StockfishOpponent,
    description: str,
    call: Callable[[], object],
    retries: int,
    log: Callable[[str], None],
) -> object:
    """Run one engine call, restarting a failed Stockfish process up to `retries` times."""
    failure: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return call()
        except (chess.engine.EngineError, OSError) as exc:
            failure = exc
            if attempt == retries:
                break
            log(
                f"stockfish {description} failed ({type(exc).__name__}: {exc}); "
                f"restarting engine [{attempt + 1}/{retries}]"
            )
            try:
                stockfish.restart()
            except Exception as restart_error:  # pragma: no cover - defensive
                failure = restart_error
                break
    raise RuntimeError(
        f"Stockfish {description} failed after {retries} restart attempts"
    ) from failure


def _new_pgn_game(brain: ChessBrain, chessfly_color: chess.Color) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = "Chessfly versus Stockfish"
    game.headers["White"] = brain.display_name if chessfly_color else "Stockfish"
    game.headers["Black"] = "Stockfish" if chessfly_color else brain.display_name
    game.headers["ChessflyMode"] = brain.mode
    return game


def _replay(
    brain: ChessBrain, chessfly_color: chess.Color, move_stack: list[str]
) -> tuple[chess.Board, chess.pgn.Game, chess.pgn.GameNode]:
    board = chess.Board()
    game = _new_pgn_game(brain, chessfly_color)
    node: chess.pgn.GameNode = game
    for uci in move_stack:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"checkpointed move {uci} is illegal in the replayed line")
        node = node.add_variation(move)
        board.push(move)
    return board, game, node


def _read_progress(run_dir: Path) -> dict:
    path = run_dir / PROGRESS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; cannot resume this run")
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise ValueError("unsupported progress schema; refusing to resume")
    return progress


def _truncate_moves(run_dir: Path, keep: int) -> None:
    """Drop move records written after the last durable checkpoint."""
    path = run_dir / MOVES_NAME
    if not path.is_file():
        if keep:
            raise FileNotFoundError(f"{path} is missing but {keep} records were logged")
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < keep:
        raise ValueError(
            f"{path} holds {len(lines)} records but the checkpoint expects {keep}"
        )
    if len(lines) > keep:
        _atomic_write_text(path, "".join(f"{line}\n" for line in lines[:keep]))


def _clear_ply_artifacts(run_dir: Path, ply: int) -> None:
    """Remove a partially written neural artifact directory before recomputing a ply."""
    target = run_dir / "neural" / f"ply-{ply:03d}"
    if target.exists():
        shutil.rmtree(target)


def _write_decision_record(
    run_dir: Path,
    ply: int,
    brain: ChessBrain,
    board_fen: str,
    decision: NeuralDecision,
    readout_spikes: tuple[int, ...],
) -> None:
    """Persist the decoder-level inputs, outputs, and parameters for one decision."""
    target = run_dir / "neural" / f"ply-{ply:03d}"
    target.mkdir(parents=True, exist_ok=True)
    scores = dict(decision.move_scores)
    best = max(scores.values()) if scores else None
    payload = {
        "ply": ply,
        "fen_before": board_fen,
        "selected_uci": decision.selected_uci,
        "reason": decision.reason,
        "silent": decision.silent,
        "tie_break": decision.tie_break,
        "readout_spike_count": len(readout_spikes),
        "readout_spike_neurons": sorted(set(int(value) for value in readout_spikes)),
        "legal_move_count": len(scores),
        "tied_move_count": (
            sum(1 for value in scores.values() if value == best) if scores else 0
        ),
        "best_score": best,
        "channel_rates_hz": {
            channel: float(value) for channel, value in decision.channel_rates_hz.items()
        },
        "move_scores": {move: float(value) for move, value in scores.items()},
        "window_ms": getattr(brain, "window_ms", None),
        "episode_seed": getattr(brain, "episode_seed", None),
    }
    _atomic_write_text(
        target / "decision.json", json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _manifest(
    brain: ChessBrain,
    stockfish: StockfishOpponent,
    chessfly_color: chess.Color,
    result: str,
    termination: str,
    plies: int,
) -> dict:
    return {
        "schema_version": 1,
        "mode": brain.mode,
        "brain": brain.describe(),
        "result": result,
        "termination": termination,
        "plies": plies,
        "chessfly_color": "white" if chessfly_color else "black",
        "stockfish": asdict(stockfish.config),
        # Record what was actually sent to the engine: with a skill level set,
        # the Elo field is carried but never applied.
        "stockfish_options": getattr(stockfish, "applied_options", None),
    }


def _checkpoint(
    run_dir: Path,
    brain: ChessBrain,
    stockfish: StockfishOpponent,
    board: chess.Board,
    game: chess.pgn.Game,
    chessfly_color: chess.Color,
    records_written: int,
    max_plies: int,
    result: str,
    termination: str,
    engine_restarts: int,
) -> None:
    """Write PGN, manifest, neural state, and progress so the run can be resumed."""
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    _atomic_write_text(run_dir / PGN_NAME, str(game) + "\n")
    _atomic_write_text(
        run_dir / MANIFEST_NAME,
        json.dumps(
            _manifest(
                brain, stockfish, chessfly_color, result, termination, records_written
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    state_dict = getattr(brain, "state_dict", None)
    has_state = state_dict is not None
    if has_state:
        _atomic_write_npz(run_dir / CHECKPOINT_NAME, state_dict())
    _atomic_write_text(
        run_dir / PROGRESS_NAME,
        json.dumps(
            {
                "schema_version": PROGRESS_SCHEMA_VERSION,
                "mode": brain.mode,
                "chessfly_color": "white" if chessfly_color else "black",
                "plies_completed": records_written,
                "records_written": records_written,
                "max_plies": max_plies,
                "fen": board.fen(),
                "side_to_move": "white" if board.turn else "black",
                "move_stack": [move.uci() for move in board.move_stack],
                "result": result,
                "termination": termination,
                "engine_restarts": engine_restarts,
                "neural_checkpoint": has_state,
                "updated_at": _utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def play_game(
    brain: ChessBrain,
    stockfish: StockfishOpponent,
    run_dir: Path,
    chessfly_color: chess.Color = chess.WHITE,
    max_plies: int = 40,
    resume: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    engine_retries: int = 3,
) -> GameResult:
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if engine_retries < 0:
        raise ValueError("engine_retries cannot be negative")
    run_dir = Path(run_dir)
    log: Callable[[str], None] = progress if progress is not None else (lambda _: None)

    engine_restarts = 0
    if resume:
        state = _read_progress(run_dir)
        if state["mode"] != brain.mode:
            raise ValueError(
                f"checkpoint mode {state['mode']!r} does not match {brain.mode!r}"
            )
        expected_color = "white" if chessfly_color else "black"
        if state["chessfly_color"] != expected_color:
            raise ValueError("checkpoint colour does not match the requested colour")
        records_written = int(state["records_written"])
        if state["termination"] == "MAX_PLIES" and max_plies <= records_written:
            raise ValueError(
                f"run stopped at the {records_written}-ply cap; pass a larger "
                "--max-plies to continue it"
            )
        if state["termination"] not in (IN_PROGRESS, "MAX_PLIES"):
            raise ValueError(
                f"run already finished as {state['termination']}; nothing to resume"
            )
        board, game, node = _replay(brain, chessfly_color, state["move_stack"])
        _truncate_moves(run_dir, records_written)
        loader = getattr(brain, "load_state_dict", None)
        if state.get("neural_checkpoint") and loader is not None:
            with np.load(run_dir / CHECKPOINT_NAME) as handle:
                loader({key: handle[key] for key in handle.files})
            log(f"resumed neural state from {run_dir / CHECKPOINT_NAME}")
        elif loader is not None:
            raise ValueError(
                "this brain carries state but the checkpoint has none; refusing to "
                "resume with a reset network"
            )
        engine_restarts = int(state.get("engine_restarts", 0))
        log(
            f"resuming at ply {records_written + 1} from {board.fen()} "
            f"(max_plies={max_plies})"
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        board, game, node = _replay(brain, chessfly_color, [])
        records_written = 0

    resumed_from_ply = records_written + 1 if resume else 0
    moves_handle = (run_dir / MOVES_NAME).open("a", encoding="utf-8")
    try:
        if records_written == 0:
            _checkpoint(
                run_dir,
                brain,
                stockfish,
                board,
                game,
                chessfly_color,
                0,
                max_plies,
                "*",
                IN_PROGRESS,
                engine_restarts,
            )
        for ply in range(records_written + 1, max_plies + 1):
            if board.is_game_over(claim_draw=True):
                break
            started = time.monotonic()
            fen_before = board.fen()
            evaluation_before = _call_engine(
                stockfish,
                f"evaluation at ply {ply}",
                lambda: stockfish.evaluate_cp(board, chessfly_color),
                engine_retries,
                log,
            )
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
                _clear_ply_artifacts(run_dir, ply)
                exporter = getattr(brain, "export_decision_artifact", None)
                if exporter is not None:
                    exporter(run_dir, ply)
                _write_decision_record(
                    run_dir, ply, brain, fen_before, decision, output_spikes
                )
                actor = "chessfly"
            else:
                move = _call_engine(
                    stockfish,
                    f"move at ply {ply}",
                    lambda: stockfish.play(board),
                    engine_retries,
                    log,
                )
                neural_score = None
                silent = False
                tie_break = False
                output_spikes = ()
                actor = "stockfish"

            san = board.san(move)
            node = node.add_variation(move)
            board.push(move)
            evaluation = _call_engine(
                stockfish,
                f"evaluation after ply {ply}",
                lambda: stockfish.evaluate_cp(board, chessfly_color),
                engine_retries,
                log,
            )
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
            record = MoveRecord(
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
            moves_handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            moves_handle.flush()
            records_written += 1

            outcome = board.outcome(claim_draw=True)
            result = outcome.result() if outcome else "*"
            termination = outcome.termination.name if outcome else IN_PROGRESS
            engine_restarts = stockfish.restarts
            _checkpoint(
                run_dir,
                brain,
                stockfish,
                board,
                game,
                chessfly_color,
                records_written,
                max_plies,
                result,
                termination,
                engine_restarts,
            )
            log(
                _progress_line(
                    ply, max_plies, record, len(output_spikes), time.monotonic() - started
                )
            )
    finally:
        moves_handle.close()

    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else "*"
    termination = outcome.termination.name if outcome else "MAX_PLIES"
    engine_restarts = stockfish.restarts
    _checkpoint(
        run_dir,
        brain,
        stockfish,
        board,
        game,
        chessfly_color,
        records_written,
        max_plies,
        result,
        termination,
        engine_restarts,
    )
    return GameResult(
        result, termination, records_written, run_dir, engine_restarts, resumed_from_ply
    )


def _progress_line(
    ply: int, max_plies: int, record: MoveRecord, readout_spikes: int, seconds: float
) -> str:
    move_number = (ply + 1) // 2
    fields = [
        f"ply {ply:3d}/{max_plies}",
        f"move {move_number:3d}",
        f"{record.actor:9s}",
        f"{record.move_san:7s}",
        f"cp {record.evaluation_before_cp:+6d} -> {record.evaluation_after_cp:+6d}",
    ]
    if record.actor == "chessfly":
        fields.append(f"spikes {readout_spikes:3d}")
        fields.append("tie" if record.tie_break else "   ")
        if record.silent_decision:
            fields.append("silent")
    fields.append(f"{seconds:5.2f}s")
    return " | ".join(fields)
