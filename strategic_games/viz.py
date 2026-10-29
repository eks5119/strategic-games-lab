"""Plotly figures. Kept separate from the app so they can be reused or tested."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .games import Game
from .nash import MixedEquilibrium, pure_nash_equilibria
from .simulation import MatchResult

ROW_COLOR = "#1f6feb"
COL_COLOR = "#d1495b"


def payoff_matrix_figure(game: Game, show_equilibria: bool = True) -> go.Figure:
    """Heatmap of the row player's payoff, annotated with both payoffs.

    Cells that are pure Nash equilibria get a star.
    """
    eq_cells = {(e.row, e.col) for e in pure_nash_equilibria(game)} if show_equilibria else set()
    text = []
    for i in range(game.n_rows):
        row_text = []
        for j in range(game.n_cols):
            label = f"({game.row_payoffs[i, j]:g}, {game.col_payoffs[i, j]:g})"
            if (i, j) in eq_cells:
                label += " ★"
            row_text.append(label)
        text.append(row_text)

    fig = go.Figure(
        go.Heatmap(
            z=game.row_payoffs,
            x=list(game.col_actions),
            y=list(game.row_actions),
            text=text,
            texttemplate="%{text}",
            textfont={"size": 15, "color": "#101418"},
            colorscale="Blues",
            showscale=False,
            hovertemplate="Row %{y}, Column %{x}<br>(row payoff, col payoff) = %{text}<extra></extra>",
            xgap=3,
            ygap=3,
        )
    )
    fig.update_layout(
        title=f"{game.name} — payoffs as (row, column); ★ = pure Nash equilibrium",
        xaxis_title="Column player",
        yaxis_title="Row player",
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def cumulative_payoff_figure(match: MatchResult) -> go.Figure:
    """Running totals for both players over the rounds."""
    df = match.frame()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["round"],
            y=df["row cumulative"],
            mode="lines",
            name=f"Row: {match.row_strategy}",
            line=dict(color=ROW_COLOR, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["round"],
            y=df["col cumulative"],
            mode="lines",
            name=f"Column: {match.col_strategy}",
            line=dict(color=COL_COLOR, width=2.5, dash="dot"),
        )
    )
    fig.update_layout(
        title="Cumulative payoff",
        xaxis_title="Round",
        yaxis_title="Total payoff so far",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )
    return fig


def moves_figure(match: MatchResult) -> go.Figure:
    """A strip chart of who played what, round by round."""
    df = match.frame()
    y_row = [1 if m == "C" else 0 for m in df["row move"]]
    y_col = [1 if m == "C" else 0 for m in df["col move"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["round"],
            y=[v + 0.12 for v in y_row],
            mode="markers",
            name="Row move",
            marker=dict(color=ROW_COLOR, size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["round"],
            y=[v - 0.12 for v in y_col],
            mode="markers",
            name="Column move",
            marker=dict(color=COL_COLOR, size=6, symbol="square"),
        )
    )
    fig.update_layout(
        title="Moves over time (1 = cooperate, 0 = defect)",
        xaxis_title="Round",
        yaxis=dict(tickvals=[0, 1], ticktext=["Defect", "Cooperate"], range=[-0.6, 1.6]),
        height=300,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def tournament_figure(scores) -> go.Figure:
    """Bar chart of total tournament scores."""
    df = scores.sort_values("Total", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df["Total"],
            y=df.index,
            orientation="h",
            marker_color=ROW_COLOR,
            text=df["Total"].round(0),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Round-robin totals",
        xaxis_title="Total payoff across all matches",
        height=max(320, 60 + 40 * len(df)),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def mixed_strategy_figure(game: Game, eq: MixedEquilibrium) -> go.Figure:
    """Side-by-side bars showing the mixing probabilities of one equilibrium."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Row player",
            x=[f"Row: {a}" for a in game.row_actions],
            y=list(eq.row_probs),
            marker_color=ROW_COLOR,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Column player",
            x=[f"Column: {a}" for a in game.col_actions],
            y=list(eq.col_probs),
            marker_color=COL_COLOR,
        )
    )
    fig.update_layout(
        title="Mixed equilibrium probabilities",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=340,
        barmode="group",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
