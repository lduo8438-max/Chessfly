"""Narrow Stockfish UCI adapter with explicit strength and time limits."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine


@dataclass(frozen=True)
class StockfishConfig:
    path: Optional[str] = None
    elo: int = 1320
    movetime_ms: int = 50
    threads: int = 1
    hash_mb: int = 64

    def __post_init__(self) -> None:
        if not 1320 <= self.elo <= 3190:
            raise ValueError("Stockfish Elo must be between 1320 and 3190")
        if self.movetime_ms <= 0 or self.threads <= 0 or self.hash_mb <= 0:
            raise ValueError("Stockfish resource limits must be positive")


class StockfishOpponent:
    def __init__(self, config: StockfishConfig) -> None:
        path = config.path or shutil.which("stockfish")
        if not path:
            raise FileNotFoundError(
                "Stockfish was not found; pass --stockfish-path or install it"
            )
        self.config = config
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure(
            {
                "Threads": config.threads,
                "Hash": config.hash_mb,
                "UCI_LimitStrength": True,
                "UCI_Elo": config.elo,
            }
        )

    def play(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(
            board, chess.engine.Limit(time=self.config.movetime_ms / 1000.0)
        )
        if result.move is None:
            raise RuntimeError("Stockfish returned no move for a live position")
        return result.move

    def evaluate_cp(self, board: chess.Board, pov: chess.Color) -> int:
        info = self.engine.analyse(
            board, chess.engine.Limit(time=self.config.movetime_ms / 1000.0)
        )
        score = info["score"].pov(pov).score(mate_score=100000)
        if score is None:
            raise RuntimeError("Stockfish returned an unavailable evaluation")
        return int(score)

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> "StockfishOpponent":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

