"""Explicit Stockfish-shaped reward calculation for later plasticity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardSignal:
    value: float
    valence: str
    delta_cp: int
    terminal_component: float


def stockfish_shaped_reward(
    before_cp: int,
    after_cp: int,
    terminal_result: str | None = None,
    deadband_cp: int = 10,
    scale_cp: int = 200,
) -> RewardSignal:
    """Return clipped evaluation delta plus an optional terminal component.

    Evaluations must already be expressed from Chessfly's point of view.
    ``terminal_result`` is one of ``win``, ``loss``, ``draw`` or ``None``.
    """
    if deadband_cp < 0 or scale_cp <= 0:
        raise ValueError("reward deadband and scale are invalid")
    if terminal_result not in (None, "win", "loss", "draw"):
        raise ValueError("terminal_result must be win, loss, draw, or None")
    delta = after_cp - before_cp
    shaped = 0.0 if abs(delta) < deadband_cp else max(-1.0, min(1.0, delta / scale_cp))
    terminal = {None: 0.0, "win": 1.0, "loss": -1.0, "draw": 0.0}[terminal_result]
    value = shaped + terminal
    valence = "positive" if value > 0 else "negative" if value < 0 else "neutral"
    return RewardSignal(value, valence, delta, terminal)

