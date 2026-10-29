"""Strategic Games Lab - Streamlit interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from strategic_games import Game, GameError, PRESETS
from strategic_games.games import prisoners_dilemma
from strategic_games.nash import (
    explain,
    iterated_elimination,
    mixed_nash_equilibria,
    pure_nash_equilibria,
    strictly_dominated_cols,
    strictly_dominated_rows,
)
from strategic_games.simulation import play_match, round_robin
from strategic_games.strategies import DEFAULT_STRATEGIES, STRATEGIES
from strategic_games.viz import (
    cumulative_payoff_figure,
    mixed_strategy_figure,
    moves_figure,
    payoff_matrix_figure,
    tournament_figure,
)

st.set_page_config(page_title="Strategic Games Lab", page_icon="🎲", layout="wide")


# --------------------------------------------------------------------- helpers
def _default_labels(n: int, axis: str) -> list[str]:
    rows = ["Top", "Bottom", "Third"]
    cols = ["Left", "Right", "Third"]
    return (rows if axis == "row" else cols)[:n]


def _matrix_editor(
    label: str,
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    key: str,
) -> np.ndarray:
    df = pd.DataFrame(values, index=row_labels, columns=col_labels)
    edited = st.data_editor(df, key=key, num_rows="fixed")
    return edited.to_numpy(dtype=float)


def _build_game(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    name: str,
) -> Game | None:
    try:
        return Game(
            row_payoffs=row_matrix,
            col_payoffs=col_matrix,
            row_actions=row_labels,
            col_actions=col_labels,
            name=name,
        )
    except GameError as exc:
        st.error(f"Invalid game: {exc}")
        return None


@st.cache_data(show_spinner=False)
def _cached_match(row, col, row_labels, col_labels, name, s_row, s_col, rounds, noise, seed):
    game = Game(row, col, list(row_labels), list(col_labels), name)
    return play_match(game, s_row, s_col, rounds=rounds, noise=noise, seed=seed)


@st.cache_data(show_spinner=False)
def _cached_tournament(row, col, row_labels, col_labels, name, names, rounds, noise, seed):
    game = Game(row, col, list(row_labels), list(col_labels), name)
    scores, _ = round_robin(game, names, rounds=rounds, noise=noise, seed=seed)
    return scores


# ---------------------------------------------------------------------- sidebar
st.sidebar.title("🎲 Strategic Games Lab")
st.sidebar.caption("Build a game, solve it, then watch strategies play it out.")

choice = st.sidebar.selectbox("Game", list(PRESETS) + ["Custom matrix"])

if choice == "Custom matrix":
    size = st.sidebar.radio("Size", [2, 3], horizontal=True)
    row_labels = _default_labels(size, "row")
    col_labels = _default_labels(size, "col")
    st.sidebar.markdown("**Row player's payoffs**")
    row_matrix = _matrix_editor(
        "Row payoffs",
        np.zeros((size, size)),
        row_labels,
        col_labels,
        key="side_row",
    )
    st.sidebar.markdown("**Column player's payoffs**")
    col_matrix = _matrix_editor(
        "Column payoffs",
        np.zeros((size, size)),
        row_labels,
        col_labels,
        key="side_col",
    )
    game_name = "Custom game"
else:
    game = PRESETS[choice]()
    row_matrix = game.row_payoffs.copy()
    col_matrix = game.col_payoffs.copy()
    row_labels = list(game.row_actions)
    col_labels = list(game.col_actions)
    game_name = game.name
    with st.sidebar.expander("Edit payoffs"):
        row_matrix = _matrix_editor(
            "Row payoffs", row_matrix, row_labels, col_labels, key="preset_row"
        )
        col_matrix = _matrix_editor(
            "Column payoffs", col_matrix, row_labels, col_labels, key="preset_col"
        )

st.sidebar.divider()
st.sidebar.caption(
    "Pure equilibria are found by best-response checking. Mixed equilibria come "
    "from support enumeration over all action subsets."
)

game = _build_game(row_matrix, col_matrix, row_labels, col_labels, game_name)
if game is None:
    st.stop()

# ------------------------------------------------------------------------- tabs
tab_eq, tab_rep, tab_tour, tab_sens = st.tabs(
    ["Equilibrium analysis", "Repeated games", "Tournament", "Payoff sensitivity"]
)

# ------------------------------------------------------- 1. equilibrium analysis
with tab_eq:
    st.subheader(game.name)
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(payoff_matrix_figure(game), width="stretch")
    with right:
        st.markdown("**Payoff matrix** (row payoff, column payoff)")
        st.dataframe(game.payoff_frame(), width="stretch")

    pure = pure_nash_equilibria(game)
    mixed = [e for e in mixed_nash_equilibria(game) if not e.is_pure]

    st.markdown("### Pure-strategy equilibria")
    if pure:
        cols = st.columns(len(pure))
        for col, eq in zip(cols, pure):
            col.metric(
                label=f"({eq.row_action}, {eq.col_action})",
                value=f"{eq.row_payoff:g} / {eq.col_payoff:g}",
                help="Payoff to row player / payoff to column player",
            )
    else:
        st.warning(
            "No pure-strategy equilibrium. Every profile has a player who wants "
            "to deviate, so the prediction must be mixed."
        )

    st.markdown("### Mixed-strategy equilibria")
    if mixed:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "row mix": ", ".join(
                            f"{game.row_actions[i]} {p:.2%}"
                            for i, p in enumerate(eq.row_probs)
                            if p > 1e-9
                        ),
                        "column mix": ", ".join(
                            f"{game.col_actions[j]} {p:.2%}"
                            for j, p in enumerate(eq.col_probs)
                            if p > 1e-9
                        ),
                        "row expected": round(eq.row_payoff, 3),
                        "column expected": round(eq.col_payoff, 3),
                    }
                    for eq in mixed
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        if len(mixed) > 1:
            pick = st.selectbox(
                "Visualise a mixed equilibrium",
                range(len(mixed)),
                format_func=lambda k: f"Equilibrium {k + 1}",
            )
        else:
            pick = 0
        st.plotly_chart(
            mixed_strategy_figure(game, mixed[pick]), width="stretch"
        )
    else:
        st.info("No non-degenerate mixed equilibrium was found for this game.")

    st.markdown("### What this means")
    for line in explain(game):
        st.markdown(f"- {line}")

    with st.expander("Dominance and iterated elimination"):
        dom_rows = strictly_dominated_rows(game)
        dom_cols = strictly_dominated_cols(game)
        if dom_rows or dom_cols:
            for i, ds in dom_rows.items():
                st.write(
                    f"Row '{game.row_actions[i]}' is strictly dominated by "
                    f"'{game.row_actions[ds[0]]}'."
                )
            for j, ds in dom_cols.items():
                st.write(
                    f"Column '{game.col_actions[j]}' is strictly dominated by "
                    f"'{game.col_actions[ds[0]]}'."
                )
        else:
            st.write("No strictly dominated strategies in this game.")
        result = iterated_elimination(game)
        if result["log"]:
            st.markdown("**Elimination log**")
            for line in result["log"]:
                st.write("• " + line)
        st.markdown("**Surviving strategies**")
        st.write(
            f"Row: {result['row_actions']} | Column: {result['col_actions']}"
        )
        st.dataframe(result["subgame"].payoff_frame(), width="stretch")

# ------------------------------------------------------------ 2. repeated games
with tab_rep:
    st.subheader("Iterated play")
    if game.shape != (2, 2):
        st.warning(
            "Repeated-game simulation needs a 2x2 matrix. Using the Prisoner's "
            "Dilemma instead."
        )
        rep_game = prisoners_dilemma()
    else:
        rep_game = game

    st.caption(
        "Move 1 = the first row/column of the matrix, move 2 = the second. "
        "Strategies see the moves actually played, including noise."
    )

    c1, c2, c3, c4 = st.columns(4)
    s_row = c1.selectbox("Row strategy", list(STRATEGIES), index=2)
    s_col = c2.selectbox("Column strategy", list(STRATEGIES), index=3)
    rounds = c3.slider("Rounds", 10, 1000, 200, step=10)
    noise = c4.slider("Noise (mistake rate)", 0.0, 0.30, 0.0, step=0.01)
    seed = st.number_input("Random seed", 0, 10_000, 0, step=1)

    match = _cached_match(
        tuple(map(tuple, rep_game.row_payoffs)),
        tuple(map(tuple, rep_game.col_payoffs)),
        tuple(rep_game.row_actions),
        tuple(rep_game.col_actions),
        rep_game.name,
        s_row,
        s_col,
        rounds,
        float(noise),
        int(seed),
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Row: {match.row_strategy}", f"{match.row_total:g}")
    m2.metric(f"Column: {match.col_strategy}", f"{match.col_total:g}")
    row_c, col_c = match.cooperation_rate
    m3.metric("Row cooperation", f"{row_c:.0%}")
    m4.metric("Column cooperation", f"{col_c:.0%}")

    st.plotly_chart(cumulative_payoff_figure(match), width="stretch")
    st.plotly_chart(moves_figure(match), width="stretch")

    with st.expander("Round-by-round table"):
        st.dataframe(match.frame(), width="stretch", hide_index=True)

    st.download_button(
        "Download this match (CSV)",
        match.frame().to_csv(index=False).encode("utf-8"),
        file_name=f"match_{s_row}_{s_col}.csv".replace(" ", "_"),
        mime="text/csv",
    )

# ---------------------------------------------------------------- 3. tournament
with tab_tour:
    st.subheader("Round-robin tournament")
    if game.shape != (2, 2):
        st.warning("Tournament needs a 2x2 matrix. Using the Prisoner's Dilemma.")
        tour_game = prisoners_dilemma()
    else:
        tour_game = game

    c1, c2, c3 = st.columns([3, 1, 1])
    names = c1.multiselect(
        "Strategies", list(STRATEGIES), default=DEFAULT_STRATEGIES
    )
    rounds_t = c2.slider("Rounds per match", 20, 1000, 200, step=20)
    noise_t = c3.slider("Noise", 0.0, 0.30, 0.0, step=0.01)

    if len(names) < 1:
        st.info("Pick at least one strategy.")
    else:
        scores = _cached_tournament(
            tuple(map(tuple, tour_game.row_payoffs)),
            tuple(map(tuple, tour_game.col_payoffs)),
            tuple(tour_game.row_actions),
            tuple(tour_game.col_actions),
            tour_game.name,
            tuple(names),
            int(rounds_t),
            float(noise_t),
            0,
        )
        st.dataframe(
            scores.style.format("{:.0f}"),
            width="stretch",
        )
        st.plotly_chart(tournament_figure(scores), width="stretch")
        st.caption(
            "Each cell is the row strategy's total payoff against that opponent. "
            "Scores are not zero-sum here: mutual cooperation creates value."
        )
        st.download_button(
            "Download scores (CSV)",
            scores.to_csv().encode("utf-8"),
            file_name="tournament_scores.csv",
            mime="text/csv",
        )

# --------------------------------------------------------- 4. payoff sensitivity
with tab_sens:
    st.subheader("Change a payoff, watch the equilibrium move")
    st.caption(
        "Edit any cell below. The analysis on the left of this page updates with "
        "the same values, so this tab is for finding the payoff values where the "
        "equilibrium set changes."
    )
    sens_row = _matrix_editor(
        "Row payoffs",
        game.row_payoffs,
        list(game.row_actions),
        list(game.col_actions),
        key="sens_row",
    )
    sens_col = _matrix_editor(
        "Column payoffs",
        game.col_payoffs,
        list(game.row_actions),
        list(game.col_actions),
        key="sens_col",
    )
    sens_game = _build_game(
        sens_row,
        sens_col,
        list(game.row_actions),
        list(game.col_actions),
        game.name + " (edited)",
    )
    if sens_game is not None:
        pure_s = pure_nash_equilibria(sens_game)
        mixed_s = [e for e in mixed_nash_equilibria(sens_game) if not e.is_pure]
        st.markdown(
            f"**Pure equilibria:** "
            + (
                ", ".join(f"({e.row_action}, {e.col_action})" for e in pure_s)
                or "none"
            )
            + f" &nbsp;|&nbsp; **Mixed equilibria:** {len(mixed_s)}"
        )
        for eq in mixed_s:
            st.write("• " + eq.describe(sens_game))

st.divider()
st.caption(
    "Strategic Games Lab · engine + tests + UI · "
    "analysis is exact for 2-player normal-form games with isolated equilibria."
)
