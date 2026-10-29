"""Tests for game construction and validation."""

import numpy as np
import pytest

from strategic_games.games import (
    Game,
    GameError,
    hawk_dove,
    matching_pennies,
    prisoners_dilemma,
    rock_paper_scissors,
    zero_sum,
)


def test_prisoners_dilemma_structure():
    game = prisoners_dilemma()
    assert game.shape == (2, 2)
    assert list(game.row_actions) == ["Cooperate", "Defect"]
    assert game.payoff(1, 0) == (5.0, 0.0)  # I defect, you cooperate: I get the temptation
    assert game.payoff(0, 0) == (3.0, 3.0)


def test_preset_payoff_ordering_is_enforced():
    with pytest.raises(GameError):
        prisoners_dilemma(temptation=2, reward=3, punishment=1, sucker=0)


def test_mismatched_shapes_rejected():
    with pytest.raises(GameError):
        Game(
            row_payoffs=[[1, 2], [3, 4]],
            col_payoffs=[[1, 2, 3]],
            row_actions=["a", "b"],
            col_actions=["x", "y"],
        )


def test_wrong_label_count_rejected():
    with pytest.raises(GameError):
        Game(
            row_payoffs=[[1, 2], [3, 4]],
            col_payoffs=[[1, 2], [3, 4]],
            row_actions=["a"],
            col_actions=["x", "y"],
        )


def test_duplicate_labels_rejected():
    with pytest.raises(GameError):
        Game(
            row_payoffs=[[1, 2], [3, 4]],
            col_payoffs=[[1, 2], [3, 4]],
            row_actions=["a", "a"],
            col_actions=["x", "y"],
        )


def test_nan_payoff_rejected():
    with pytest.raises(GameError):
        Game(
            row_payoffs=[[1, float("nan")], [3, 4]],
            col_payoffs=[[1, 2], [3, 4]],
            row_actions=["a", "b"],
            col_actions=["x", "y"],
        )


def test_one_dimensional_input_rejected():
    with pytest.raises(GameError):
        Game(
            row_payoffs=[1, 2],
            col_payoffs=[[1, 2]],
            row_actions=["a"],
            col_actions=["x", "y"],
        )


def test_from_table_builds_both_matrices():
    game = Game.from_table(
        [[(3, 3), (0, 5)], [(5, 0), (1, 1)]],
        ["C", "D"],
        ["C", "D"],
        name="PD",
    )
    assert game.row_payoffs.tolist() == [[3.0, 0.0], [5.0, 1.0]]
    assert game.col_payoffs.tolist() == [[3.0, 5.0], [0.0, 1.0]]
    assert game.name == "PD"


def test_zero_sum_negates_column_payoffs():
    game = zero_sum([[1, -1], [-1, 1]])
    assert np.allclose(game.col_payoffs, -game.row_payoffs)


def test_replace_keeps_labels():
    game = matching_pennies()
    edited = game.replace(row_payoffs=[[2, -2], [-2, 2]])
    assert edited.row_payoffs.tolist() == [[2.0, -2.0], [-2.0, 2.0]]
    assert list(edited.row_actions) == ["Heads", "Tails"]
    # the original is untouched
    assert game.row_payoffs.tolist() == [[1.0, -1.0], [-1.0, 1.0]]


def test_text_rendering_contains_payoffs():
    text = rock_paper_scissors().to_text()
    assert "Rock" in text and "Scissors" in text
    assert "(-1, 1)" in text


def test_hawk_dove_payoff_formula():
    game = hawk_dove(value=2, cost=4)
    assert game.payoff(0, 0) == (-1.0, -1.0)  # both fight: (v - c) / 2
    assert game.payoff(1, 1) == (1.0, 1.0)  # both dove: v / 2
