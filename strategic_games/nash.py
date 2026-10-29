"""Equilibrium solvers: best responses, dominance, pure and mixed Nash equilibria.

The maths, briefly:

* A profile (i, j) is a **pure Nash equilibrium** when i is a best response to j
  *and* j is a best response to i. Nothing more.
* A strategy is **strictly dominated** when another strategy pays strictly more
  against every opponent action. Removing dominated strategies can never remove
  a Nash equilibrium.
* A **mixed Nash equilibrium** is a pair of probability distributions where every
  action that gets positive probability is a best response. For a 2-player game
  we find them by *support enumeration*: guess which actions are used, solve the
  linear "make the opponent indifferent" system, then verify no unused action
  beats the mix.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .games import Game

TOL = 1e-9


@dataclass(frozen=True)
class PureEquilibrium:
    """A pure-strategy Nash equilibrium."""

    row: int
    col: int
    row_action: str
    col_action: str
    row_payoff: float
    col_payoff: float

    @property
    def profile(self) -> tuple[str, str]:
        return (self.row_action, self.col_action)

    def describe(self) -> str:
        return (
            f"({self.row_action}, {self.col_action}) "
            f"-> payoffs ({self.row_payoff:g}, {self.col_payoff:g})"
        )


@dataclass(frozen=True)
class MixedEquilibrium:
    """A mixed-strategy Nash equilibrium (includes pure ones as degenerate cases)."""

    row_probs: tuple[float, ...]
    col_probs: tuple[float, ...]
    row_payoff: float
    col_payoff: float

    @property
    def is_pure(self) -> bool:
        return self.support_size_row == 1 and self.support_size_col == 1

    @property
    def support_size_row(self) -> int:
        return sum(1 for p in self.row_probs if p > 1e-9)

    @property
    def support_size_col(self) -> int:
        return sum(1 for p in self.col_probs if p > 1e-9)

    def describe(self, game: Game) -> str:
        row_mix = ", ".join(
            f"{game.row_actions[i]}={p:.3f}"
            for i, p in enumerate(self.row_probs)
            if p > 1e-9
        )
        col_mix = ", ".join(
            f"{game.col_actions[j]}={p:.3f}"
            for j, p in enumerate(self.col_probs)
            if p > 1e-9
        )
        return (
            f"Row: [{row_mix}] vs Column: [{col_mix}] "
            f"-> expected payoffs ({self.row_payoff:.3f}, {self.col_payoff:.3f})"
        )


# ------------------------------------------------------------- best responses
def row_best_responses(game: Game) -> dict[int, tuple[int, ...]]:
    """For each column action index, the row actions that maximise the row payoff."""
    A = game.row_payoffs
    out: dict[int, tuple[int, ...]] = {}
    for j in range(game.n_cols):
        column = A[:, j]
        best = column.max()
        out[j] = tuple(int(i) for i in np.flatnonzero(column >= best - TOL))
    return out


def col_best_responses(game: Game) -> dict[int, tuple[int, ...]]:
    """For each row action index, the column actions that maximise the col payoff."""
    B = game.col_payoffs
    out: dict[int, tuple[int, ...]] = {}
    for i in range(game.n_rows):
        row = B[i, :]
        best = row.max()
        out[i] = tuple(int(j) for j in np.flatnonzero(row >= best - TOL))
    return out


def is_best_response(game: Game, row: int, col: int) -> tuple[bool, bool]:
    """Is (row, col) a mutual best response? Returns (row_ok, col_ok)."""
    row_ok = game.row_payoffs[row, col] >= game.row_payoffs[:, col].max() - TOL
    col_ok = game.col_payoffs[row, col] >= game.col_payoffs[row, :].max() - TOL
    return bool(row_ok), bool(col_ok)


# ----------------------------------------------------------------- pure Nash
def pure_nash_equilibria(game: Game) -> list[PureEquilibrium]:
    """Every pure-strategy Nash equilibrium, in row-major order."""
    found: list[PureEquilibrium] = []
    for i in range(game.n_rows):
        for j in range(game.n_cols):
            row_ok, col_ok = is_best_response(game, i, j)
            if row_ok and col_ok:
                a, b = game.payoff(i, j)
                found.append(
                    PureEquilibrium(
                        row=i,
                        col=j,
                        row_action=game.row_actions[i],
                        col_action=game.col_actions[j],
                        row_payoff=a,
                        col_payoff=b,
                    )
                )
    return found


# ------------------------------------------------------------------ dominance
def strictly_dominated_rows(game: Game) -> dict[int, list[int]]:
    """Map each strictly dominated row action to the actions that dominate it."""
    A = game.row_payoffs
    out: dict[int, list[int]] = {}
    for i in range(game.n_rows):
        dominators = [
            k
            for k in range(game.n_rows)
            if k != i and np.all(A[k, :] > A[i, :] + TOL)
        ]
        if dominators:
            out[i] = dominators
    return out


def strictly_dominated_cols(game: Game) -> dict[int, list[int]]:
    """Map each strictly dominated column action to the actions that dominate it."""
    B = game.col_payoffs
    out: dict[int, list[int]] = {}
    for j in range(game.n_cols):
        dominators = [
            l
            for l in range(game.n_cols)
            if l != j and np.all(B[:, l] > B[:, j] + TOL)
        ]
        if dominators:
            out[j] = dominators
    return out


def weakly_dominated_rows(game: Game) -> dict[int, list[int]]:
    """Rows weakly dominated by another row (never worse, sometimes better)."""
    A = game.row_payoffs
    out: dict[int, list[int]] = {}
    for i in range(game.n_rows):
        dominators = [
            k
            for k in range(game.n_rows)
            if k != i
            and np.all(A[k, :] >= A[i, :] - TOL)
            and np.any(A[k, :] > A[i, :] + TOL)
        ]
        if dominators:
            out[i] = dominators
    return out


def iterated_elimination(game: Game, strict: bool = True) -> dict:
    """Iteratively delete dominated actions.

    Returns a dict with the surviving action indices, the surviving sub-game and
    a human-readable log of each deletion round.
    """
    A = game.row_payoffs.copy()
    B = game.col_payoffs.copy()
    rows = list(range(game.n_rows))
    cols = list(range(game.n_cols))
    log: list[str] = []

    while True:
        sub = Game(
            row_payoffs=A[np.ix_(rows, cols)],
            col_payoffs=B[np.ix_(rows, cols)],
            row_actions=[game.row_actions[i] for i in rows],
            col_actions=[game.col_actions[j] for j in cols],
            name=game.name,
        )
        if strict:
            bad_rows = strictly_dominated_rows(sub)
            bad_cols = strictly_dominated_cols(sub)
        else:
            bad_rows = weakly_dominated_rows(sub)
            bad_cols = {
                j: [
                    l
                    for l in range(sub.n_cols)
                    if l != j
                    and np.all(sub.col_payoffs[:, l] >= sub.col_payoffs[:, j] - TOL)
                    and np.any(sub.col_payoffs[:, l] > sub.col_payoffs[:, j] + TOL)
                ]
                for j in range(sub.n_cols)
            }
            bad_cols = {k: v for k, v in bad_cols.items() if v}
        if not bad_rows and not bad_cols:
            break
        removed_rows = sorted(bad_rows, reverse=True)
        removed_cols = sorted(bad_cols, reverse=True)
        for i in removed_rows:
            log.append(
                f"Deleted row '{sub.row_actions[i]}' "
                f"(dominated by '{sub.row_actions[bad_rows[i][0]]}')"
            )
        for j in removed_cols:
            log.append(
                f"Deleted column '{sub.col_actions[j]}' "
                f"(dominated by '{sub.col_actions[bad_cols[j][0]]}')"
            )
        rows = [r for k, r in enumerate(rows) if k not in removed_rows]
        cols = [c for k, c in enumerate(cols) if k not in removed_cols]

    return {
        "row_indices": rows,
        "col_indices": cols,
        "row_actions": [game.row_actions[i] for i in rows],
        "col_actions": [game.col_actions[j] for j in cols],
        "log": log,
        "subgame": Game(
            row_payoffs=A[np.ix_(rows, cols)],
            col_payoffs=B[np.ix_(rows, cols)],
            row_actions=[game.row_actions[i] for i in rows],
            col_actions=[game.col_actions[j] for j in cols],
            name=f"{game.name} (after elimination)",
        ),
    }


# ---------------------------------------------------------------- mixed Nash
def _solve_support(
    A: np.ndarray,
    B: np.ndarray,
    rows: tuple[int, ...],
    cols: tuple[int, ...],
    tol: float = 1e-9,
) -> MixedEquilibrium | None:
    """Try one candidate support pair; return an equilibrium or None."""
    k = len(rows)
    A_sub = A[np.ix_(rows, cols)]
    B_sub = B[np.ix_(rows, cols)]

    # Column player's mix y over `cols` must make the row player indifferent.
    M = np.zeros((k + 1, k + 1))
    M[:k, :k] = A_sub
    M[:k, k] = -1.0
    M[k, :k] = 1.0
    rhs = np.zeros(k + 1)
    rhs[k] = 1.0

    try:
        y_sub = np.linalg.solve(M, rhs)[:k]
        M2 = np.zeros((k + 1, k + 1))
        M2[:k, :k] = B_sub.T
        M2[:k, k] = -1.0
        M2[k, :k] = 1.0
        x_sub = np.linalg.solve(M2, rhs)[:k]
    except np.linalg.LinAlgError:
        return None  # degenerate support: no isolated interior solution

    # Probabilities must be strictly inside the simplex, else this is a
    # smaller support that we will test separately.
    if np.any(x_sub <= tol) or np.any(y_sub <= tol):
        return None

    x = np.zeros(A.shape[0])
    y = np.zeros(A.shape[1])
    x[list(rows)] = x_sub
    y[list(cols)] = y_sub
    x = x / x.sum()
    y = y / y.sum()

    row_payoff = float(x @ A @ y)
    col_payoff = float(x @ B @ y)

    # No action outside the support may do better than the mix does.
    if np.any(A @ y > row_payoff + 1e-7):
        return None
    if np.any(x @ B > col_payoff + 1e-7):
        return None

    return MixedEquilibrium(
        row_probs=tuple(round(float(p), 12) for p in x),
        col_probs=tuple(round(float(p), 12) for p in y),
        row_payoff=row_payoff,
        col_payoff=col_payoff,
    )


def mixed_nash_equilibria(game: Game, tol: float = 1e-9) -> list[MixedEquilibrium]:
    """All Nash equilibria found by support enumeration.

    Works for any 2-player game whose equilibria are *isolated* (the usual case
    in textbook games). Highly degenerate games - e.g. one player is fully
    indifferent everywhere - admit continua of equilibria and are reported here
    only partially. See README "Known limitations".
    """
    A, B = game.row_payoffs, game.col_payoffs
    m, n = game.shape
    found: list[MixedEquilibrium] = []

    for k in range(1, min(m, n) + 1):
        for rows in itertools.combinations(range(m), k):
            for cols in itertools.combinations(range(n), k):
                eq = _solve_support(A, B, rows, cols, tol=tol)
                if eq is None:
                    continue
                if any(_close(eq, other) for other in found):
                    continue
                found.append(eq)

    found.sort(key=lambda e: (e.support_size_row + e.support_size_col, -e.row_payoff))
    return found


def _close(a: MixedEquilibrium, b: MixedEquilibrium, atol: float = 1e-6) -> bool:
    if len(a.row_probs) != len(b.row_probs):
        return False
    return bool(
        np.allclose(a.row_probs, b.row_probs, atol=atol)
        and np.allclose(a.col_probs, b.col_probs, atol=atol)
    )


def is_nash(game: Game, row_mix, col_mix, tol: float = 1e-7) -> bool:
    """Verify a candidate mixed profile: no pure action beats the mix."""
    x = np.asarray(row_mix, dtype=float)
    y = np.asarray(col_mix, dtype=float)
    if not np.isclose(x.sum(), 1.0, atol=1e-6) or not np.isclose(
        y.sum(), 1.0, atol=1e-6
    ):
        return False
    if np.any(x < -tol) or np.any(y < -tol):
        return False
    u = float(x @ game.row_payoffs @ y)
    v = float(x @ game.col_payoffs @ y)
    return bool(
        np.all(game.row_payoffs @ y <= u + tol)
        and np.all(x @ game.col_payoffs <= v + tol)
    )


# -------------------------------------------------------------- explanations
def explain(game: Game) -> list[str]:
    """Plain-English reading of the game's strategic structure."""
    lines: list[str] = []
    pure = pure_nash_equilibria(game)
    rb = row_best_responses(game)
    cb = col_best_responses(game)

    if pure:
        for eq in pure:
            row_alts = ", ".join(
                f"{game.row_actions[k]} {game.row_payoffs[k, eq.col]:g}"
                for k in range(game.n_rows)
                if k != eq.row
            ) or "no alternative"
            col_alts = ", ".join(
                f"{game.col_actions[l]} {game.col_payoffs[eq.row, l]:g}"
                for l in range(game.n_cols)
                if l != eq.col
            ) or "no alternative"
            lines.append(
                f"**({eq.row_action}, {eq.col_action})** is a Nash equilibrium. "
                f"Against {eq.col_action}, the row player's best reply is "
                f"{eq.row_action} ({eq.row_payoff:g} vs {row_alts}). "
                f"Against {eq.row_action}, the column player's best reply is "
                f"{eq.col_action} ({eq.col_payoff:g} vs {col_alts}). "
                "Neither player gains by switching alone, so the profile is stable."
            )
    else:
        lines.append(
            "There is **no pure-strategy Nash equilibrium**: at every profile at "
            "least one player wants to switch. Rational play therefore requires "
            "randomising, and the prediction is a mixed equilibrium."
        )

    if len(pure) > 1:
        lines.append(
            f"There are {len(pure)} pure equilibria, so the theory does not say "
            "which one players reach. Coordination, history and focal points "
            "(Schelling) decide it - this is a coordination problem, not a "
            "prediction."
        )

    for dom, dominators in strictly_dominated_rows(game).items():
        lines.append(
            f"Row action '{game.row_actions[dom]}' is strictly dominated by "
            f"'{game.row_actions[dominators[0]]}' - a rational row player never "
            "plays it."
        )
    for dom, dominators in strictly_dominated_cols(game).items():
        lines.append(
            f"Column action '{game.col_actions[dom]}' is strictly dominated by "
            f"'{game.col_actions[dominators[0]]}'."
        )

    mixed = mixed_nash_equilibria(game)
    interior = [e for e in mixed if not e.is_pure]
    for eq in interior:
        lines.append(
            "Mixed equilibrium: "
            + eq.describe(game)
            + ". Each player mixes exactly so the other is indifferent between "
            "the actions in their support - that indifference is what makes the "
            "other player willing to randomise."
        )

    if not pure and not interior:
        lines.append(
            "No isolated equilibrium was found; the game is probably degenerate "
            "(a player indifferent across several actions)."
        )
    return lines


def summary_table(game: Game):
    """A DataFrame listing every equilibrium found for the game."""
    import pandas as pd

    rows = []
    for eq in pure_nash_equilibria(game):
        rows.append(
            {
                "type": "pure",
                "row strategy": eq.row_action,
                "col strategy": eq.col_action,
                "row payoff": eq.row_payoff,
                "col payoff": eq.col_payoff,
            }
        )
    for eq in mixed_nash_equilibria(game):
        if eq.is_pure:
            continue
        rows.append(
            {
                "type": "mixed",
                "row strategy": ", ".join(
                    f"{game.row_actions[i]} {p:.2f}"
                    for i, p in enumerate(eq.row_probs)
                    if p > 1e-9
                ),
                "col strategy": ", ".join(
                    f"{game.col_actions[j]} {p:.2f}"
                    for j, p in enumerate(eq.col_probs)
                    if p > 1e-9
                ),
                "row payoff": round(eq.row_payoff, 4),
                "col payoff": round(eq.col_payoff, 4),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["type", "row strategy", "col strategy", "row payoff", "col payoff"],
    )
