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
        self.path = path
        self.restarts = 0
        self.engine = self._spawn()

    def _spawn(self) -> chess.engine.SimpleEngine:
        engine = chess.engine.SimpleEngine.popen_uci(self.path)
        engine.configure(
            {
                "Threads": self.config.threads,
                "Hash": self.config.hash_mb,
                "UCI_LimitStrength": True,
                "UCI_Elo": self.config.elo,
            }
        )
        return engine

    def restart(self) -> None:
        """Replace a wedged or dead engine process with a freshly configured one."""
        try:
            self.engine.quit()
        except BaseException:
            pass
        self.engine = self._spawn()
        self.restarts += 1

    def play(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(
            board, chess.engine.Limit(time=self.config.movetime_ms / 1000.0)
        )
        if result.move is None:
            raise chess.engine.EngineError("Stockfish returned no move for a live position")
        return result.move

    def evaluate_cp(
        self, board: chess.Board, pov: chess.Color, depth: int | None = None
    ) -> int:
        if depth is not None and depth <= 0:
            raise ValueError("analysis depth must be positive")
        info = self.engine.analyse(
            board,
            chess.engine.Limit(
                depth=depth,
                time=None if depth is not None else self.config.movetime_ms / 1000.0,
            ),
        )
        score = info["score"].pov(pov).score(mate_score=100000)
        if score is None:
            raise chess.engine.EngineError("Stockfish returned an unavailable evaluation")
        return int(score)

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> "StockfishOpponent":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
