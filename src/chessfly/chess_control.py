"""Convert descending-neuron spike counts into a legal chess move."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import chess


FILES = "abcdefgh"
RANKS = "12345678"
CHANNELS = tuple(
    [f"from_file_{value}" for value in FILES]
    + [f"from_rank_{value}" for value in RANKS]
    + [f"to_file_{value}" for value in FILES]
    + [f"to_rank_{value}" for value in RANKS]
)


def assign_readout_populations(
    neuron_ids: Iterable[int], seed: str = "chessfly-v1"
) -> dict[str, tuple[int, ...]]:
    """Assign neurons to 32 balanced channels with a stable hash ordering."""
    ids = tuple(neuron_ids)
    if len(ids) < len(CHANNELS):
        raise ValueError("at least 32 unique readout neurons are required")
    if len(set(ids)) != len(ids):
        raise ValueError("readout neuron IDs must be unique")
    ordered = sorted(
        ids,
        key=lambda neuron: hashlib.sha256(
            f"{seed}:{neuron}".encode("utf-8")
        ).digest(),
    )
    assigned: dict[str, list[int]] = {channel: [] for channel in CHANNELS}
    for index, neuron in enumerate(ordered):
        assigned[CHANNELS[index % len(CHANNELS)]].append(neuron)
    return {channel: tuple(values) for channel, values in assigned.items()}


@dataclass(frozen=True)
class NeuralDecision:
    selected_uci: Optional[str]
    move_scores: Mapping[str, float]
    channel_rates_hz: Mapping[str, float]
    silent: bool
    tie_break: bool
    reason: str


class ChessMoveDecoder:
    """Score legal moves using from/to file and rank populations."""

    def __init__(self, populations: Mapping[str, Sequence[int]]) -> None:
        missing = set(CHANNELS) - set(populations)
        if missing:
            raise ValueError(f"missing readout channels: {sorted(missing)}")
        self.populations = {
            channel: frozenset(populations[channel]) for channel in CHANNELS
        }
        if any(not population for population in self.populations.values()):
            raise ValueError("every readout channel must contain at least one neuron")

    def decode(
        self,
        board: chess.Board,
        spike_frames: Iterable[Iterable[int]],
        window_ms: float,
    ) -> NeuralDecision:
        if window_ms <= 0:
            raise ValueError("window_ms must be positive")
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return NeuralDecision(None, {}, {}, True, False, "game_over")

        counts: Counter[int] = Counter()
        for frame in spike_frames:
            counts.update(frame)
        seconds = window_ms / 1000.0
        rates = {
            channel: sum(counts[neuron] for neuron in population)
            / len(population)
            / seconds
            for channel, population in self.populations.items()
        }
        silent = not any(rates.values())
        scored = {
            move.uci(): self._score_move(move, rates) for move in legal_moves
        }
        best_score = max(scored.values())
        winners = [move for move in legal_moves if scored[move.uci()] == best_score]
        selected = min(winners, key=self._tie_key)
        return NeuralDecision(
            selected.uci(),
            scored,
            rates,
            silent,
            len(winners) > 1,
            "selected",
        )

    @staticmethod
    def _score_move(move: chess.Move, rates: Mapping[str, float]) -> float:
        from_name = chess.square_name(move.from_square)
        to_name = chess.square_name(move.to_square)
        return (
            rates[f"from_file_{from_name[0]}"]
            + rates[f"from_rank_{from_name[1]}"]
            + rates[f"to_file_{to_name[0]}"]
            + rates[f"to_rank_{to_name[1]}"]
        )

    @staticmethod
    def _tie_key(move: chess.Move) -> tuple[str, int]:
        promotion_order = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3,
            None: 0,
        }
        return move.uci()[:4], promotion_order[move.promotion]

