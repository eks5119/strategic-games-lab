"""Strategies for the repeated Prisoner's Dilemma (and any 2x2 C/D game).

A strategy is just a function of history:

    move = strategy(my_history, opponent_history)  ->  "C" or "D"

where each history is the list of moves actually played so far (actual moves,
so noise or mistakes are visible to the strategy - which is the interesting
part).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

COOPERATE = "C"
DEFECT = "D"

History = Sequence[str]
MoveFn = Callable[[History, History], str]


@dataclass
class Strategy:
    """A named decision rule for the iterated game."""

    name: str
    fn: MoveFn
    description: str = ""
    tag: str = "classic"
    reset: Callable[[], None] | None = None

    def __call__(self, my_history: History, opp_history: History) -> str:
        return self.fn(my_history, opp_history)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Strategy({self.name!r})"


# --------------------------------------------------------------------- rules
def always_cooperate(my: History, opp: History) -> str:
    return COOPERATE


def always_defect(my: History, opp: History) -> str:
    return DEFECT


def tit_for_tat(my: History, opp: History) -> str:
    """Cooperate first, then copy the opponent's previous move."""
    if not opp:
        return COOPERATE
    return opp[-1]


def tit_for_two_tats(my: History, opp: History) -> str:
    """Defect only after two consecutive defections - more forgiving than TFT."""
    if len(opp) < 2:
        return COOPERATE
    return DEFECT if opp[-1] == DEFECT and opp[-2] == DEFECT else COOPERATE


def grim_trigger(my: History, opp: History) -> str:
    """Cooperate until the opponent defects once, then defect forever."""
    return DEFECT if DEFECT in opp else COOPERATE


def pavlov(my: History, opp: History) -> str:
    """Win-stay, lose-shift: repeat a move that produced a good outcome."""
    if not my:
        return COOPERATE
    won = my[-1] == opp[-1]  # (C,C) or (D,D) both pay at least as well as switching
    return my[-1] if won else (DEFECT if my[-1] == COOPERATE else COOPERATE)


def generous_tit_for_tat(forgiveness: float = 0.1, seed: int = 0) -> Strategy:
    """TFT that occasionally forgives a defection (breaks defection spirals)."""
    rng = random.Random(seed)

    def fn(my: History, opp: History) -> str:
        if not opp:
            return COOPERATE
        if opp[-1] == DEFECT and rng.random() < forgiveness:
            return COOPERATE
        return opp[-1]

    return Strategy(
        name=f"Generous Tit-for-Tat ({forgiveness:.0%})",
        fn=fn,
        description="Tit-for-tat that forgives a defection with fixed probability.",
        tag="forgiving",
        reset=lambda: rng.seed(seed),
    )


def random_strategy(seed: int = 0) -> Strategy:
    """Fair coin each round - the equilibrium behaviour of Matching Pennies."""
    rng = random.Random(seed)

    def fn(my: History, opp: History) -> str:
        return COOPERATE if rng.random() < 0.5 else DEFECT

    return Strategy(
        name=f"Random (seed {seed})",
        fn=fn,
        description="Coin flip every round: 50% cooperate, 50% defect.",
        tag="random",
        reset=lambda: rng.seed(seed),
    )


def alternator(my: History, opp: History) -> str:
    """Strict alternation C, D, C, D ... (punishes nothing, exploits nothing)."""
    return COOPERATE if len(my) % 2 == 0 else DEFECT


# ------------------------------------------------------------------ registry
def _build_registry() -> dict[str, Strategy]:
    registry = {
        "Always Cooperate": Strategy(
            "Always Cooperate", always_cooperate, "Cooperate no matter what."
        ),
        "Always Defect": Strategy(
            "Always Defect", always_defect, "Defect no matter what."
        ),
        "Tit for Tat": Strategy(
            "Tit for Tat",
            tit_for_tat,
            "Cooperate first, then copy the opponent's last move.",
        ),
        "Tit for Two Tats": Strategy(
            "Tit for Two Tats",
            tit_for_two_tats,
            "Defect only after two defections in a row.",
        ),
        "Grim Trigger": Strategy(
            "Grim Trigger",
            grim_trigger,
            "One defection and cooperation is over forever.",
        ),
        "Pavlov": Strategy(
            "Pavlov", pavlov, "Win-stay, lose-shift: repeat what worked."
        ),
        "Alternator": Strategy(
            "Alternator", alternator, "Alternate cooperate and defect."
        ),
        "Random": random_strategy(0),
        "Generous Tit for Tat": generous_tit_for_tat(0.1, seed=0),
    }
    return registry


STRATEGIES: dict[str, Strategy] = _build_registry()
DEFAULT_STRATEGIES = ["Always Cooperate", "Always Defect", "Tit for Tat", "Grim Trigger"]


def resolve(strategy: str | Strategy | MoveFn) -> Strategy:
    """Accept a registry name, a Strategy, or a bare function."""
    if isinstance(strategy, Strategy):
        return strategy
    if isinstance(strategy, str):
        try:
            return STRATEGIES[strategy]
        except KeyError:
            raise KeyError(
                f"unknown strategy {strategy!r}; available: {sorted(STRATEGIES)}"
            ) from None
    if callable(strategy):
        return Strategy(getattr(strategy, "__name__", "custom"), strategy)
    raise TypeError(f"cannot use {strategy!r} as a strategy")
