"""A transparent toy bridge used before MaleCNS data is prepared.

This is a smoke-test network, not a fruit-fly connectome. Each piece identity
activates two deterministically selected output channels through LIF synapses.
"""

from __future__ import annotations

import hashlib

import chess

from .chess_control import CHANNELS, ChessMoveDecoder, NeuralDecision
from .snn import LIFNetwork, LIFParameters, Synapse


PIECE_PLANES = 12
SQUARES = 64
INPUT_COUNT = PIECE_PLANES * SQUARES
OUTPUT_START = INPUT_COUNT
OUTPUT_COUNT = len(CHANNELS)


def _input_index(square: int, piece: chess.Piece) -> int:
    color_offset = 0 if piece.color == chess.WHITE else 6
    plane = color_offset + piece.piece_type - 1
    return plane * SQUARES + square


def _output_channels(input_neuron: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"chessfly-toy:{input_neuron}".encode()).digest()
    first = int.from_bytes(digest[:4], "big") % OUTPUT_COUNT
    second = int.from_bytes(digest[4:8], "big") % (OUTPUT_COUNT - 1)
    if second >= first:
        second += 1
    return first, second


class ToyChessBrain:
    """Run a real LIF decision path without claiming MaleCNS provenance."""

    mode = "toy-not-male-cns"
    display_name = "Chessfly (toy)"
    window_ms = 3.0

    def __init__(self) -> None:
        synapses = []
        for input_neuron in range(INPUT_COUNT):
            for channel in _output_channels(input_neuron):
                synapses.append(
                    Synapse(input_neuron, OUTPUT_START + channel, 20.0, 1)
                )
        self.network = LIFNetwork(
            INPUT_COUNT + OUTPUT_COUNT,
            synapses,
            LIFParameters(dt_ms=1.0, refractory_ms=1.0),
        )
        populations = {
            channel: (OUTPUT_START + index,) for index, channel in enumerate(CHANNELS)
        }
        self.decoder = ChessMoveDecoder(populations)
        self.output_neuron_indices = frozenset(
            range(OUTPUT_START, OUTPUT_START + OUTPUT_COUNT)
        )

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "neurons": self.network.neuron_count,
            "edges": len(self.network.synapses),
            "window_ms": self.window_ms,
            "male_cns": False,
        }

    def decide(self, board: chess.Board) -> tuple[NeuralDecision, list[tuple[int, ...]]]:
        self.network.reset()
        drive = {
            _input_index(square, piece): 20.0
            for square, piece in board.piece_map().items()
        }
        frames = [self.network.step(drive), self.network.step(), self.network.step()]
        return self.decoder.decode(board, frames, self.window_ms), frames
