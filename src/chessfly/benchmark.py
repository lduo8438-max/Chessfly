"""Fixed-position comparisons for frozen Chessfly controllers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np

from .male_cns_brain import MaleCNSSubgraphBrain
from .chess_control import ChessMoveDecoder, assign_readout_populations
from .stockfish import StockfishOpponent


POSITION_LINES: tuple[tuple[str, ...], ...] = (
    (),
    ("e2e4", "e7e5"),
    ("d2d4", "d7d5", "c2c4"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("g1f3", "d7d5", "g2g3", "c7c5", "f1g2"),
    ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3"),
    ("d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"),
    ("e2e4", "c7c6", "d2d4", "d7d5", "e4e5"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"),
    ("e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"),
)


@dataclass(frozen=True)
class BenchmarkRecord:
    position: int
    controller: str
    fen: str
    move_uci: str
    evaluation_before_cp: int
    evaluation_after_cp: int
    evaluation_delta_cp: int
    silent: bool
    tie_break: bool
    total_spikes: int | None
    readout_spikes: int | None


def fixed_positions(limit: int | None = None) -> list[chess.Board]:
    lines = POSITION_LINES if limit is None else POSITION_LINES[:limit]
    boards = []
    for line in lines:
        board = chess.Board()
        for uci in line:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"invalid frozen benchmark line at {uci}")
            board.push(move)
        boards.append(board)
    return boards


def run_fixed_benchmark(
    stockfish: StockfishOpponent,
    frozen: MaleCNSSubgraphBrain,
    shuffled: MaleCNSSubgraphBrain,
    output_dir: Path,
    position_limit: int = 10,
    random_seed: int = 20260912,
    evaluation_depth: int = 10,
) -> dict[str, object]:
    if position_limit < 1 or position_limit > len(POSITION_LINES):
        raise ValueError(f"position_limit must be between 1 and {len(POSITION_LINES)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(random_seed)
    records: list[BenchmarkRecord] = []
    for index, source_board in enumerate(fixed_positions(position_limit), start=1):
        pov = source_board.turn
        before = stockfish.evaluate_cp(source_board, pov, depth=evaluation_depth)
        legal = sorted(source_board.legal_moves, key=lambda move: move.uci())
        choices: list[tuple[str, chess.Move, bool, bool, int | None, int | None]] = []
        choices.append(("fixed-first-legal", legal[0], False, False, None, None))
        choices.append(
            (
                f"random-seed-{random_seed}",
                legal[int(rng.integers(len(legal)))],
                False,
                False,
                None,
                None,
            )
        )
        for controller, brain in (("male-cns-frozen", frozen), ("male-cns-shuffled", shuffled)):
            brain.reset()
            decision, _ = brain.decide(source_board.copy(stack=True))
            if decision.selected_uci is None:
                raise RuntimeError("neural controller returned no move for a live position")
            stats = brain.last_stats
            choices.append(
                (
                    controller,
                    chess.Move.from_uci(decision.selected_uci),
                    decision.silent,
                    decision.tie_break,
                    stats.total_spikes if stats else None,
                    stats.readout_spikes if stats else None,
                )
            )
        after_by_move: dict[str, int] = {}
        for _, move, _, _, _, _ in choices:
            if move.uci() not in after_by_move:
                board = source_board.copy(stack=False)
                board.push(move)
                after_by_move[move.uci()] = stockfish.evaluate_cp(
                    board, pov, depth=evaluation_depth
                )
        for controller, move, silent, tie_break, total_spikes, readout_spikes in choices:
            after = after_by_move[move.uci()]
            records.append(
                BenchmarkRecord(
                    index,
                    controller,
                    source_board.fen(),
                    move.uci(),
                    before,
                    after,
                    after - before,
                    silent,
                    tie_break,
                    total_spikes,
                    readout_spikes,
                )
            )

    with (output_dir / "positions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    controllers = sorted({record.controller for record in records})
    summary_rows = []
    for controller in controllers:
        selected = [record for record in records if record.controller == controller]
        deltas = [record.evaluation_delta_cp for record in selected]
        neural = [record for record in selected if record.total_spikes is not None]
        summary_rows.append(
            {
                "controller": controller,
                "positions": len(selected),
                "mean_evaluation_delta_cp": float(np.mean(deltas)),
                "median_evaluation_delta_cp": float(np.median(deltas)),
                "silent_fraction": (
                    float(np.mean([record.silent for record in neural])) if neural else None
                ),
                "mean_readout_spikes": (
                    float(np.mean([record.readout_spikes for record in neural]))
                    if neural
                    else None
                ),
            }
        )
    summary: dict[str, object] = {
        "schema_version": 1,
        "suite": "frozen-opening-positions-v1",
        "random_seed": random_seed,
        "episode_seed": frozen.episode_seed,
        "shuffle_seed": shuffled.shuffle_seed,
        "positions": position_limit,
        "stockfish_evaluation_depth": evaluation_depth,
        "metric": "Stockfish centipawn change from side-to-move POV; higher is better",
        "controllers": summary_rows,
        "selection_warning": "This small suite is an engineering check, not evidence of chess skill.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_seed_sweep(
    brain: MaleCNSSubgraphBrain,
    board: chess.Board,
    output_dir: Path,
    start_seed: int = 0,
    count: int = 100,
) -> dict[str, object]:
    """Replay one frozen neural response through deterministic readout seeds."""
    if count < 1:
        raise ValueError("seed count must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    brain.reset()
    _, frames = brain.decide(board.copy(stack=True))
    rows = []
    for seed in range(start_seed, start_seed + count):
        populations = assign_readout_populations(
            brain.output_neuron_indices, seed=f"chessfly-male-cns-v1:{seed}"
        )
        decoder = ChessMoveDecoder(populations)
        first = decoder.decode(board, frames, brain.window_ms)
        second = decoder.decode(board, frames, brain.window_ms)
        if first != second:
            raise RuntimeError(f"readout seed {seed} did not replay exactly")
        rows.append(
            {
                "seed": seed,
                "selected_uci": first.selected_uci,
                "silent": first.silent,
                "tie_break": first.tie_break,
                "selected_score": first.move_scores.get(first.selected_uci),
            }
        )
    with (output_dir / "seeds.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    moves: dict[str, int] = {}
    for row in rows:
        move = str(row["selected_uci"])
        moves[move] = moves.get(move, 0) + 1
    summary: dict[str, object] = {
        "schema_version": 1,
        "fen": board.fen(),
        "start_seed": start_seed,
        "seeds": count,
        "exact_replay_verified": True,
        "unique_moves": len(moves),
        "move_counts": dict(sorted(moves.items())),
        "silent_fraction": float(np.mean([row["silent"] for row in rows])),
        "readout_spikes": brain.last_stats.readout_spikes if brain.last_stats else None,
        "scope": "Readout assignment sweep over one frozen neural response; not independent games.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
