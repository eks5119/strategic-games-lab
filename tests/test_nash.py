"""Tests for the equilibrium solvers - the mathematical heart of the project.

Each test pins down a result you can check by hand, which is exactly why these
games are the standard textbook examples.
"""

import numpy as np
import pytest

from strategic_games.games import (
    Game,
    battle_of_the_sexes,
    coordination_game,
    hawk_dove,
    matching_pennies,
    prisoners_dilemma,
    rock_paper_scissors,
    stag_hunt,
    chicken,
)
from strategic_games.nash import (
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
)


# --------------------------------------------------------------- best responses
def test_best_responses_prisoners_dilemma():
    game = prisoners_dilemma()
    # Defect is the best reply to either opponent move
    assert row_best_responses(game) == {0: (1,), 1: (1,)}
    assert col_best_responses(game) == {0: (1,), 1: (1,)}


def test_best_response_set_can_have_two_elements():
    game = Game(
        row_payoffs=[[1, 1], [1, 0]],
        col_payoffs=[[1, 1], [1, 0]],
        row_actions=["a", "b"],
        col_actions=["x", "y"],
    )
    # against column x, both rows pay 1 -> two best responses
    assert row_best_responses(game)[0] == (0, 1)
    # against column y only the top row pays anything
    assert row_best_responses(game)[1] == (0,)


# -------------------------------------------------------------------- pure Nash
def test_prisoners_dilemma_has_one_pure_equilibrium():
    game = prisoners_dilemma()
    eq = pure_nash_equilibria(game)
    assert len(eq) == 1
    assert eq[0].profile == ("Defect", "Defect")
    assert eq[0].row_payoff == 1.0


def test_matching_pennies_has_no_pure_equilibrium():
    assert pure_nash_equilibria(matching_pennies()) == []


def test_coordination_game_has_two_pure_equilibria():
    eq = pure_nash_equilibria(coordination_game())
    assert {e.profile for e in eq} == {("Top", "Left"), ("Bottom", "Right")}


def test_stag_hunt_has_two_pure_equilibria():
    eq = pure_nash_equilibria(stag_hunt())
    assert {e.profile for e in eq} == {("Stag", "Stag"), ("Hare", "Hare")}


def test_battle_of_the_sexes_has_two_pure_equilibria():
    eq = pure_nash_equilibria(battle_of_the_sexes())
    assert {e.profile for e in eq} == {("Opera", "Opera"), ("Football", "Football")}


def test_rock_paper_scissors_has_no_pure_equilibrium():
    assert pure_nash_equilibria(rock_paper_scissors()) == []


def test_hawk_dove_asymmetric_equilibria():
    # Hawk vs Dove and Dove vs Hawk are both stable; both-hawk and both-dove are not
    eq = {e.profile for e in pure_nash_equilibria(hawk_dove())}
    assert eq == {("Hawk", "Dove"), ("Dove", "Hawk")}


# ------------------------------------------------------------------ dominance
def test_defect_strictly_dominates_cooperate_in_pd():
    game = prisoners_dilemma()
    assert strictly_dominated_rows(game) == {0: [1]}
    assert strictly_dominated_cols(game) == {0: [1]}


def test_iterated_elimination_solves_pd():
    result = iterated_elimination(prisoners_dilemma())
    assert result["row_actions"] == ["Defect"]
    assert result["col_actions"] == ["Defect"]
    assert len(result["log"]) == 2


def test_iterated_elimination_on_coordination_game_keeps_everything():
    result = iterated_elimination(coordination_game())
    assert result["row_actions"] == ["Top", "Bottom"]
    assert result["log"] == []


def test_no_dominance_in_matching_pennies():
    assert strictly_dominated_rows(matching_pennies()) == {}
    assert strictly_dominated_cols(matching_pennies()) == {}


# ------------------------------------------------------------------ mixed Nash
def test_matching_pennies_mixed_equilibrium_is_fifty_fifty():
    eq = [e for e in mixed_nash_equilibria(matching_pennies()) if not e.is_pure]
    assert len(eq) == 1
    assert eq[0].row_probs == pytest.approx((0.5, 0.5))
    assert eq[0].col_probs == pytest.approx((0.5, 0.5))
    assert eq[0].row_payoff == pytest.approx(0.0)


def test_rock_paper_scissors_mixed_equilibrium_is_uniform():
    eq = [e for e in mixed_nash_equilibria(rock_paper_scissors()) if not e.is_pure]
    assert len(eq) == 1
    assert eq[0].row_probs == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert eq[0].col_probs == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert eq[0].row_payoff == pytest.approx(0.0)


def test_coordination_game_mixed_equilibrium_values():
    # Row indifferent when 2q = 1 - q  ->  q = 1/3
    eq = [e for e in mixed_nash_equilibria(coordination_game()) if not e.is_pure]
    assert len(eq) == 1
    assert eq[0].row_probs == pytest.approx((1 / 3, 2 / 3))
    assert eq[0].col_probs == pytest.approx((1 / 3, 2 / 3))
    assert eq[0].row_payoff == pytest.approx(2 / 3)


def test_hawk_dove_mixed_equilibrium_is_value_over_cost():
    eq = [e for e in mixed_nash_equilibria(hawk_dove(value=2, cost=4)) if not e.is_pure]
    assert len(eq) == 1
    assert eq[0].row_probs == pytest.approx((0.5, 0.5))
    assert eq[0].col_probs == pytest.approx((0.5, 0.5))


def test_pd_has_no_interior_mixed_equilibrium():
    # Defect strictly dominates, so the only equilibrium is the pure one
    interior = [e for e in mixed_nash_equilibria(prisoners_dilemma()) if not e.is_pure]
    assert interior == []


def test_every_reported_mixed_equilibrium_verifies():
    for factory in (
        matching_pennies,
        coordination_game,
        rock_paper_scissors,
        hawk_dove,
        stag_hunt,
        battle_of_the_sexes,
    ):
        game = factory()
        for eq in mixed_nash_equilibria(game):
            assert is_nash(game, eq.row_probs, eq.col_probs), (
                f"unverified equilibrium for {game.name}"
            )


def test_is_nash_rejects_a_bad_profile():
    game = matching_pennies()
    assert not is_nash(game, [1.0, 0.0], [1.0, 0.0])
    assert not is_nash(game, [0.5, 0.6], [0.5, 0.5])  # does not sum to 1


def test_mixed_equilibrium_probabilities_sum_to_one():
    for eq in mixed_nash_equilibria(rock_paper_scissors()):
        assert np.isclose(sum(eq.row_probs), 1.0)
        assert np.isclose(sum(eq.col_probs), 1.0)


# --------------------------------------------------------------- presentation
def test_explain_mentions_equilibrium_and_dominance():
    text = " ".join(explain(prisoners_dilemma()))
    assert "Nash equilibrium" in text
    assert "strictly dominated" in text


def test_explain_flags_missing_pure_equilibrium():
    text = " ".join(explain(matching_pennies())).lower()
    assert "no pure-strategy nash equilibrium" in text


def test_summary_table_lists_equilibria():
    table = summary_table(coordination_game())
    assert list(table.columns) == [
        "type",
        "row strategy",
        "col strategy",
        "row payoff",
        "col payoff",
    ]
    assert (table["type"] == "pure").sum() == 2
    assert (table["type"] == "mixed").sum() == 1


 # --------------------------------------------------------------------- chicken
def test_chicken_has_two_asymmetric_equilibria():
    eq = {e.profile for e in pure_nash_equilibria(chicken())}
    assert eq == {("Straight", "Swerve"), ("Swerve", "Straight")}


def test_chicken_mutual_swerve_is_not_an_equilibrium():
    profiles = {e.profile for e in pure_nash_equilibria(chicken())}
    assert ("Swerve", "Swerve") not in profiles


def test_chicken_mixed_equilibrium_values():
    # row indifferent when 3q + 2(1 - q) = 4q  ->  q = 2/3
    eq = [e for e in mixed_nash_equilibria(chicken()) if not e.is_pure]
    assert len(eq) == 1
    assert eq[0].row_probs == pytest.approx((2 / 3, 1 / 3))
    assert eq[0].col_probs == pytest.approx((2 / 3, 1 / 3))
    assert eq[0].row_payoff == pytest.approx(8 / 3)


def test_chicken_has_no_strictly_dominated_strategy():
    assert strictly_dominated_rows(chicken()) == {}
    assert strictly_dominated_cols(chicken()) == {}

