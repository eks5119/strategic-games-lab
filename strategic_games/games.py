"""Core representation of two-player normal-form games.

A game is stored as two payoff matrices of the same shape:

    row_payoffs[i, j] = payoff to the ROW player when row plays i, column plays j
    col_payoffs[i, j] = payoff to the COLUMN player for the same profile

Everything else in the package (equilibrium solvers, repeated-game simulator,
plots, the Streamlit app) is built on top of this one class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


class GameError(ValueError):
    """Raised when the inputs do not describe a valid two-player game."""


def _as_matrix(values, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise GameError(f"{label} must be a 2-D matrix, got {arr.ndim}-D")
    if arr.size == 0:
        raise GameError(f"{label} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise GameError(f"{label} contains NaN or infinite payoffs")
    return arr


def _as_labels(labels: Sequence[str], count: int, axis: str) -> list[str]:
    labels = [str(x).strip() for x in labels]
    if len(labels) != count:
        raise GameError(
            f"expected {count} {axis} action label(s), got {len(labels)}"
        )
    if any(not lab for lab in labels):
        raise GameError(f"{axis} action labels must not be empty")
    if len(set(labels)) != count:
        raise GameError(f"{axis} action labels must be unique, got {labels}")
    return labels


@dataclass(eq=False)
class Game:
    """A two-player normal-form game (bimatrix form)."""

    row_payoffs: np.ndarray
    col_payoffs: np.ndarray
    row_actions: Sequence[str]
    col_actions: Sequence[str]
    name: str = "Custom game"

    def __post_init__(self) -> None:
        self.row_payoffs = _as_matrix(self.row_payoffs, "row_payoffs")
        self.col_payoffs = _as_matrix(self.col_payoffs, "col_payoffs")
        if self.row_payoffs.shape != self.col_payoffs.shape:
            raise GameError(
                "row_payoffs and col_payoffs must have the same shape, got "
                f"{self.row_payoffs.shape} and {self.col_payoffs.shape}"
            )
        m, n = self.row_payoffs.shape
        self.row_actions = _as_labels(self.row_actions, m, "row")
        self.col_actions = _as_labels(self.col_actions, n, "column")

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, int]:
        return self.row_payoffs.shape

    @property
    def n_rows(self) -> int:
        return self.row_payoffs.shape[0]

    @property
    def n_cols(self) -> int:
        return self.row_payoffs.shape[1]

    def payoff(self, row: int, col: int) -> tuple[float, float]:
        """Payoffs (row player, column player) at profile (row, col)."""
        return (
            float(self.row_payoffs[row, col]),
            float(self.col_payoffs[row, col]),
        )

    def profile_name(self, row: int, col: int) -> str:
        return f"({self.row_actions[row]}, {self.col_actions[col]})"

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Game(name={self.name!r}, shape={self.shape}, "
            f"row_actions={list(self.row_actions)}, "
            f"col_actions={list(self.col_actions)})"
        )

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_table(
        cls,
        table: Sequence[Sequence[Sequence[float]]],
        row_actions: Sequence[str] | None = None,
        col_actions: Sequence[str] | None = None,
        name: str = "Custom game",
    ) -> "Game":
        """Build a game from a table of ``(row_payoff, col_payoff)`` pairs.

        Example::

            Game.from_table([[(3, 3), (0, 5)],
                             [(5, 0), (1, 1)]],
                            ["Cooperate", "Defect"],
                            ["Cooperate", "Defect"])
        """
        cells = [[tuple(float(v) for v in cell) for cell in row] for row in table]
        for row in cells:
            for cell in row:
                if len(cell) != 2:
                    raise GameError(
                        f"each cell must be a (row, col) pair, got {cell!r}"
                    )
        row_payoffs = [[cell[0] for cell in row] for row in cells]
        col_payoffs = [[cell[1] for cell in row] for row in cells]
        n_rows = len(cells)
        n_cols = len(cells[0]) if n_rows else 0
        row_actions = row_actions or [f"R{i + 1}" for i in range(n_rows)]
        col_actions = col_actions or [f"C{j + 1}" for j in range(n_cols)]
        return cls(
            row_payoffs=row_payoffs,
            col_payoffs=col_payoffs,
            row_actions=list(row_actions),
            col_actions=list(col_actions),
            name=name,
        )

    def replace(
        self,
        row_payoffs: np.ndarray | None = None,
        col_payoffs: np.ndarray | None = None,
        name: str | None = None,
    ) -> "Game":
        """Return a copy with some payoff values swapped out."""
        return Game(
            row_payoffs=self.row_payoffs if row_payoffs is None else row_payoffs,
            col_payoffs=self.col_payoffs if col_payoffs is None else col_payoffs,
            row_actions=self.row_actions,
            col_actions=self.col_actions,
            name=self.name if name is None else name,
        )

    # ------------------------------------------------------------------ display
    def payoff_frame(self):
        """A DataFrame of "(row payoff, col payoff)" strings, for display."""
        import pandas as pd

        data = [
            [
                f"({self.row_payoffs[i, j]:g}, {self.col_payoffs[i, j]:g})"
                for j in range(self.n_cols)
            ]
            for i in range(self.n_rows)
        ]
        return pd.DataFrame(
            data, index=list(self.row_actions), columns=list(self.col_actions)
        )

    def to_text(self) -> str:
        """A plain-text rendering of the payoff matrix (handy in the terminal)."""
        width = max(len(lab) for lab in list(self.row_actions) + list(self.col_actions))
        cell_w = max(
            len(f"({self.row_payoffs[i, j]:g}, {self.col_payoffs[i, j]:g})")
            for i in range(self.n_rows)
            for j in range(self.n_cols)
        )
        header = " " * (width + 2) + "".join(
            lab.rjust(cell_w + 2) for lab in self.col_actions
        )
        lines = [f"{self.name}", header]
        for i, lab in enumerate(self.row_actions):
            cells = "".join(
                f"({self.row_payoffs[i, j]:g}, {self.col_payoffs[i, j]:g})".rjust(
                    cell_w + 2
                )
                for j in range(self.n_cols)
            )
            lines.append(f"{lab.ljust(width)}  {cells}")
        return "\n".join(lines)


# --------------------------------------------------------------------- presets
def prisoners_dilemma(
    temptation: float = 5.0,
    reward: float = 3.0,
    punishment: float = 1.0,
    sucker: float = 0.0,
) -> Game:
    """Classic Prisoner's Dilemma: Defect strictly dominates Cooperate."""
    if not (temptation > reward > punishment > sucker):
        raise GameError("need temptation > reward > punishment > sucker")
    return Game(
        row_payoffs=[[reward, sucker], [temptation, punishment]],
        col_payoffs=[[reward, temptation], [sucker, punishment]],
        row_actions=["Cooperate", "Defect"],
        col_actions=["Cooperate", "Defect"],
        name="Prisoner's Dilemma",
    )


def matching_pennies() -> Game:
    """Zero-sum, no pure equilibrium, one fully mixed equilibrium."""
    return Game(
        row_payoffs=[[1, -1], [-1, 1]],
        col_payoffs=[[-1, 1], [1, -1]],
        row_actions=["Heads", "Tails"],
        col_actions=["Heads", "Tails"],
        name="Matching Pennies",
    )


def coordination_game() -> Game:
    """Two pure equilibria plus one mixed equilibrium."""
    return Game(
        row_payoffs=[[2, 0], [0, 1]],
        col_payoffs=[[2, 0], [0, 1]],
        row_actions=["Top", "Bottom"],
        col_actions=["Left", "Right"],
        name="Coordination Game",
    )


def stag_hunt() -> Game:
    """Cooperation and safety both equilibria: the classic trust problem."""
    return Game(
        row_payoffs=[[4, 0], [3, 3]],
        col_payoffs=[[4, 3], [0, 3]],
        row_actions=["Stag", "Hare"],
        col_actions=["Stag", "Hare"],
        name="Stag Hunt",
    )


def battle_of_the_sexes() -> Game:
    """Two pure equilibria with opposed preferences (a coordination conflict)."""
    return Game(
        row_payoffs=[[2, 0], [0, 1]],
        col_payoffs=[[1, 0], [0, 2]],
        row_actions=["Opera", "Football"],
        col_actions=["Opera", "Football"],
        name="Battle of the Sexes",
    )


def rock_paper_scissors() -> Game:
    """Symmetric zero-sum game with a single mixed equilibrium."""
    return Game(
        row_payoffs=[[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
        col_payoffs=[[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        row_actions=["Rock", "Paper", "Scissors"],
        col_actions=["Rock", "Paper", "Scissors"],
        name="Rock-Paper-Scissors",
    )


def hawk_dove(value: float = 2.0, cost: float = 4.0) -> Game:
    """Hawk-Dove (chicken): one mixed equilibrium at Hawk probability v/c."""
    if not (cost > value > 0):
        raise GameError("need cost > value > 0")
    half = (value - cost) / 2
    return Game(
        row_payoffs=[[half, value], [0.0, value / 2]],
        col_payoffs=[[half, 0.0], [value, value / 2]],
        row_actions=["Hawk", "Dove"],
        col_actions=["Hawk", "Dove"],
        name="Hawk-Dove",
    )


def zero_sum(matrix: Sequence[Sequence[float]], name: str = "Zero-sum game") -> Game:
    """Build a zero-sum game from the row player's payoff matrix only."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise GameError("matrix must be 2-D")
    m, n = arr.shape
    return Game(
        row_payoffs=arr,
        col_payoffs=-arr,
        row_actions=[f"R{i + 1}" for i in range(m)],
        col_actions=[f"C{j + 1}" for j in range(n)],
        name=name,
    )


PRESETS: dict[str, Callable[[], Game]] = {
    "Prisoner's Dilemma": prisoners_dilemma,
    "Matching Pennies": matching_pennies,
    "Coordination Game": coordination_game,
    "Stag Hunt": stag_hunt,
    "Battle of the Sexes": battle_of_the_sexes,
    "Rock-Paper-Scissors": rock_paper_scissors,
    "Hawk-Dove": hawk_dove,
}
