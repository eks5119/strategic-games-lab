"""Tests for strategies, repeated matches and tournaments."""

import pytest

from strategic_games.games import GameError, prisoners_dilemma, rock_paper_scissors
from strategic_games.simulation import best_response, play_match, round_robin
from strategic_games.strategies import (
    STRATEGIES,
    generous_tit_for_tat,
    resolve,
    tit_for_tat,
)

PD = prisoners_dilemma()


# -------------------------------------------------------------------- strategies
def test_tit_for_tat_copies_last_move():
    assert tit_for_tat([], []) == "C"
    assert tit_for_tat(["C"], ["D"]) == "D"
    assert tit_for_tat(["C", "D"], ["C", "C"]) == "C"


def test_resolve_accepts_name_strategy_or_function():
    assert resolve("Tit for Tat").name == "Tit for Tat"
    assert resolve(tit_for_tat).name == "tit_for_tat"
    with pytest.raises(KeyError):
        resolve("No Such Strategy")


# ----------------------------------------------------------------------- matches
def test_tit_for_tat_versus_itself_cooperates_forever():
    match = play_match(PD, "Tit for Tat", "Tit for Tat", rounds=200, seed=0)
    assert match.row_total == pytest.approx(600.0)
    assert match.col_total == pytest.approx(600.0)
    assert all(r.row_move == "C" for r in match.rounds)
    assert match.cooperation_rate == (1.0, 1.0)


def test_always_defect_exploits_cooperation_then_earns_punishment():
    # Round 1: D vs C = 5 for row. Tit for Tat then defects forever: 1 each.
    match = play_match(PD, "Always Defect", "Tit for Tat", rounds=200, seed=0)
    assert match.row_total == pytest.approx(5.0 + 199 * 1.0)
    assert match.col_total == pytest.approx(0.0 + 199 * 1.0)
    assert match.rounds[0].row_payoff == 5.0
    assert match.rounds[1].row_payoff == 1.0


def test_grim_trigger_never_forgives():
    match = play_match(PD, "Grim Trigger", "Always Defect", rounds=50, seed=0)
    assert match.rounds[0].row_payoff == 0.0  # C vs D: the sucker payoff
    assert all(r.row_move == "D" for r in match.rounds[1:])


def test_noise_breaks_cooperation():
    clean = play_match(PD, "Tit for Tat", "Tit for Tat", rounds=200, seed=1)
    noisy = play_match(PD, "Tit for Tat", "Tit for Tat", rounds=200, noise=0.05, seed=1)
    assert noisy.row_total < clean.row_total
    assert noisy.cooperation_rate[0] < 1.0


def test_noise_is_reproducible_with_a_seed():
    a = play_match(PD, "Tit for Tat", "Grim Trigger", rounds=100, noise=0.1, seed=7)
    b = play_match(PD, "Tit for Tat", "Grim Trigger", rounds=100, noise=0.1, seed=7)
    assert a.row_total == b.row_total


def test_generous_tit_for_tat_stops_defection_spirals():
    forgiving = generous_tit_for_tat(forgiveness=1.0, seed=0)
    match = play_match(PD, forgiving, "Tit for Tat", rounds=100, noise=0.05, seed=3)
    harsh = play_match(PD, "Tit for Tat", "Tit for Tat", rounds=100, noise=0.05, seed=3)
    assert match.row_total >= harsh.row_total


def test_match_frame_has_cumulative_columns():
    # TFT is exploited in round 1 (0), then both defect: 1 per round
    frame = play_match(PD, "Tit for Tat", "Always Defect", rounds=10).frame()
    assert len(frame) == 10
    assert frame["row cumulative"].iloc[-1] == pytest.approx(9.0)
    assert frame["col cumulative"].iloc[-1] == pytest.approx(14.0)


def test_invalid_inputs_raise():
    with pytest.raises(GameError):
        play_match(rock_paper_scissors(), "Tit for Tat", "Tit for Tat", rounds=10)
    with pytest.raises(GameError):
        play_match(PD, "Tit for Tat", "Tit for Tat", rounds=0)
    with pytest.raises(GameError):
        play_match(PD, "Tit for Tat", "Tit for Tat", rounds=10, noise=1.0)


# ------------------------------------------------------------------- tournaments
def test_round_robin_shape_and_ordering():
    names = ["Always Cooperate", "Always Defect", "Tit for Tat", "Grim Trigger"]
    scores, matches = round_robin(PD, names, rounds=100, seed=0)
    assert scores.shape == (4, 6)  # 4 opponents + Total + Average per round
    assert len(matches) == 16
    assert "Total" in scores.columns
    assert scores["Total"].is_monotonic_decreasing


def test_tit_for_tat_beats_always_defect_overall():
    # The Axelrod result: in a pool that contains reciprocity, nice strategies win.
    pool = ["Always Cooperate", "Always Defect", "Tit for Tat", "Grim Trigger"]
    scores, _ = round_robin(PD, pool, rounds=200)
    assert scores.loc["Tit for Tat", "Total"] > scores.loc["Always Defect", "Total"]


def test_defect_can_win_a_tiny_pool():
    # Honest limitation: with no reciprocator in the pool, defection dominates.
    scores, _ = round_robin(PD, ["Always Cooperate", "Always Defect"], rounds=200)
    assert scores.index[0] == "Always Defect"


def test_always_cooperate_is_the_doormat():
    scores, _ = round_robin(PD, ["Always Cooperate", "Always Defect"], rounds=100)
    # AllC: 300 vs itself, 0 vs AllD. AllD: 500 vs AllC, 100 vs itself.
    assert scores.loc["Always Cooperate", "Total"] == pytest.approx(300.0)
    assert scores.loc["Always Defect", "Total"] == pytest.approx(600.0)


def test_round_robin_rejects_empty_strategy_list():
    with pytest.raises(GameError):
        round_robin(PD, [], rounds=10)


def test_best_response_scores_every_candidate():
    table = best_response(PD, "Always Defect", candidates=list(STRATEGIES), rounds=50)
    assert len(table) == len(STRATEGIES)
    assert table["score"].is_monotonic_decreasing
    # against Always Defect nothing scores more than mutual defection plus the
    # one-round sucker hit that only a first-move cooperator can avoid
    assert table.iloc[0]["score"] <= 5.0 * 50
