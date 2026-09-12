"""Chessfly's reference neural and chess-control interfaces."""

from .chess_control import (
    CHANNELS,
    ChessMoveDecoder,
    NeuralDecision,
    assign_readout_populations,
)
from .snn import LIFNetwork, LIFParameters, Synapse

__all__ = [
    "CHANNELS",
    "ChessMoveDecoder",
    "LIFNetwork",
    "LIFParameters",
    "NeuralDecision",
    "Synapse",
    "assign_readout_populations",
]

