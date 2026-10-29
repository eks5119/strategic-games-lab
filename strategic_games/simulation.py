"""Simulation of repeated 2x2 games and round-robin tournaments."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .games import Game, GameError
from .strategies import COOPERATE, DEFECT, STRATEGIES, Strategy, resolve


@dataclass
class RoundResult:
    number: int
    row_move: str
    col_move: str
    row_payoff: float
    col_payoff: float


@dataclass
class MatchResult:
    """Everything that happened in one repeated game between two strategies."""

    game: Game
    row_strategy: str
    col_strategy: str
    rounds: list[RoundResult]
    noise: float = 0.0
    seed: int | None = None

    @property
    def row_total(self) -> float:
        return float(sum(r.row_payoff for r in self.rounds))

    @property
    def col_total(self) -> float:
        return float(sum(r.col_payoff for r in self.rounds))

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def cooperation_rate(self) -> tuple[float, float]:
        if not self.rounds:
            return (0.0, 0.0)
        row_c = sum(1 for r in self.rounds if r.row_move == COOPERATE) / len(self.rounds)
        col_c = sum(1 for r in self.rounds if r.col_move == COOPERATE) / len(self.rounds)
        return (row_c, col_c)

    def frame(self) -> pd.DataFrame:
        """Round-by-round table including running totals."""
        data = {
            "round": [r.number for r in self.rounds],
            "row move": [r.row_move for r in self.rounds],
            "col move": [r.col_move for r in self.rounds],
            "row payoff": [r.row_payoff for r in self.rounds],
            "col payoff": [r.col_payoff for r in self.rounds],
        }
        df = pd.DataFrame(data)
        df["row cumulative"] = df["row payoff"].cumsum()
        df["col cumulative"] = df["col payoff"].cumsum()
        return df

    def summary(self) -> dict:
        row_c, col_c = self.cooperation_rate
        return {
            "row strategy": self.row_strategy,
            "col strategy": self.col_strategy,
            "rounds": self.n_rounds,
            "noise": self.noise,
            "row total": self.row_total,
            "col total": self.col_total,
            "row avg": self.row_total / self.n_rounds if self.n_rounds else 0.0,
            "col avg": self.col_total / self.n_rounds if self.n_rounds else 0.0,
            "row cooperation": row_c,
            "col cooperation": col_c,
        }


def _move_index(game: Game, move: str) -> int:
    index = 0 if move == COOPERATE else 1
    if index >= game.n_cols:
        raise GameError("game has fewer than 2 actions per player")
    return index


def play_match(
    game: Game,
    row_strategy: str | Strategy,
    col_strategy: str | Strategy,
    rounds: int = 200,
    noise: float = 0.0,
    seed: int | None = None,
) -> MatchResult:
    """Play a 2x2 game repeatedly.

    ``noise`` is the probability that a chosen move is flipped on the way to the
    table (an implementation error). Strategies see the *actual* moves, not the
    intended ones, which is why noise makes unforgiving strategies collapse.
    """
    if game.shape != (2, 2):
        raise GameError(
            f"repeated games need a 2x2 payoff matrix, got {game.shape}"
        )
    if rounds < 1:
        raise GameError("rounds must be at least 1")
    if not 0.0 <= noise < 1.0:
        raise GameError("noise must be in [0, 1)")

    row_s = resolve(row_strategy)
    col_s = resolve(col_strategy)
    # Strategies with internal randomness get reset so a run is reproducible.
    for s in (row_s, col_s):
        if s.reset is not None:
            s.reset()
    rng = random.Random(seed)

    row_history: list[str] = []
    col_history: list[str] = []
    results: list[RoundResult] = []

    for t in range(1, rounds + 1):
        row_move = row_s(row_history, col_history)
        col_move = col_s(col_history, row_history)
        if rng.random() < noise:
            row_move = DEFECT if row_move == COOPERATE else COOPERATE
        if rng.random() < noise:
            col_move = DEFECT if col_move == COOPERATE else COOPERATE

        i = _move_index(game, row_move)
        j = _move_index(game, col_move)
        a, b = game.payoff(i, j)

        row_history.append(row_move)
        col_history.append(col_move)
        results.append(
            RoundResult(
                number=t,
                row_move=row_move,
                col_move=col_move,
                row_payoff=a,
                col_payoff=b,
            )
        )

    return MatchResult(
        game=game,
        row_strategy=row_s.name,
        col_strategy=col_s.name,
        rounds=results,
        noise=noise,
        seed=seed,
    )


def round_robin(
    game: Game,
    strategies: Iterable[str | Strategy] | None = None,
    rounds: int = 200,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[MatchResult]]:
    """Every strategy plays every other (and itself) once.

    Returns a score matrix (rows = strategy, columns = opponent, last column
    "Total") plus the list of individual matches so you can plot any of them.
    """
    names = list(strategies) if strategies is not None else list(STRATEGIES)
    resolved = [resolve(s) for s in names]
    if not resolved:
        raise GameError("need at least one strategy")

    table: dict[str, dict[str, float]] = {}
    matches: list[MatchResult] = []
    totals: dict[str, float] = {s.name: 0.0 for s in resolved}

    for row_s in resolved:
        table[row_s.name] = {}
        for col_s in resolved:
            match = play_match(
                game,
                row_s,
                col_s,
                rounds=rounds,
                noise=noise,
                seed=seed,
            )
            table[row_s.name][col_s.name] = match.row_total
            totals[row_s.name] += match.row_total
            matches.append(match)

    df = pd.DataFrame(table).T
    df["Total"] = [totals[name] for name in df.index]
    df["Average per round"] = df["Total"] / (rounds * len(resolved))
    df = df.sort_values("Total", ascending=False)
    return df, matches


def best_response(
    game: Game,
    opponent: str | Strategy,
    candidates: Iterable[str] | None = None,
    rounds: int = 200,
    noise: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Score every candidate strategy against one fixed opponent."""
    candidates = list(candidates) if candidates else list(STRATEGIES)
    rows = []
    for name in candidates:
        match = play_match(game, name, opponent, rounds=rounds, noise=noise, seed=seed)
        rows.append(
            {
                "strategy": match.row_strategy,
                "score": match.row_total,
                "average": match.row_total / match.n_rounds,
                "cooperation": match.cooperation_rate[0],
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(
        drop=True
    )
