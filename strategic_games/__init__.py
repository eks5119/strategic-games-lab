"""Strategic Games Lab - a small engine for analysing strategic situations.

Quick start::

    from strategic_games import prisoners_dilemma, pure_nash_equilibria, explain

    game = prisoners_dilemma()
    print(game.to_text())
    for line in explain(game):
        print("-", line)
"""

from .games import (
    Game,
    GameError,
    PRESETS,
    battle_of_the_sexes,
    coordination_game,
    hawk_dove,
    matching_pennies,
    prisoners_dilemma,
    rock_paper_scissors,
    stag_hunt,
    zero_sum,
)
from .nash import (
    MixedEquilibrium,
    PureEquilibrium,
    col_best_responses,
    explain,
    is_nash,
    iterated_elimination,
    mixed_nash_equilibria,
    pure_nash_equilibria,
    row_best_responses,
    strictly_dominated_cols,
    strictly_dominated_rows,
    summary_table,
    weakly_dominated_rows,
)
from .simulation import MatchResult, best_response, play_match, round_robin
from .strategies import DEFAULT_STRATEGIES, STRATEGIES, Strategy

__version__ = "0.1.0"

__all__ = [
    "Game",
    "GameError",
    "PRESETS",
    "battle_of_the_sexes",
    "coordination_game",
    "hawk_dove",
    "matching_pennies",
    "prisoners_dilemma",
    "rock_paper_scissors",
    "stag_hunt",
    "zero_sum",
    "PureEquilibrium",
    "MixedEquilibrium",
    "pure_nash_equilibria",
    "mixed_nash_equilibria",
    "row_best_responses",
    "col_best_responses",
    "strictly_dominated_rows",
    "strictly_dominated_cols",
    "weakly_dominated_rows",
    "iterated_elimination",
    "is_nash",
    "explain",
    "summary_table",
    "play_match",
    "round_robin",
    "best_response",
    "MatchResult",
    "Strategy",
    "STRATEGIES",
    "DEFAULT_STRATEGIES",
    "__version__",
]
