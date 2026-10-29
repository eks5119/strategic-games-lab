"""Terminal demo - run this first to see the engine work without Streamlit.

    python demo.py
"""

from __future__ import annotations

from strategic_games import PRESETS, explain, pure_nash_equilibria, mixed_nash_equilibria
from strategic_games.simulation import play_match, round_robin
from strategic_games.strategies import STRATEGIES


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    rule("1. Every preset game, solved")
    for name, factory in PRESETS.items():
        game = factory()
        pure = pure_nash_equilibria(game)
        mixed = [e for e in mixed_nash_equilibria(game) if not e.is_pure]
        print(f"\n{name}  ({game.n_rows}x{game.n_cols})")
        print(game.to_text())
        print(
            "  pure equilibria: "
            + (", ".join(e.profile[0] + "/" + e.profile[1] for e in pure) or "none")
        )
        print(f"  mixed equilibria: {len(mixed)}")
        for eq in mixed:
            print("    " + eq.describe(game))

    rule("2. Why the Prisoner's Dilemma traps both players")
    pd_game = PRESETS["Prisoner's Dilemma"]()
    for line in explain(pd_game):
        print("- " + line)

    rule("3. Tit for Tat vs Always Defect (200 rounds)")
    match = play_match(pd_game, "Tit for Tat", "Always Defect", rounds=200, seed=0)
    print(f"{match.row_strategy:>20} : {match.row_total:>7.0f}")
    print(f"{match.col_strategy:>20} : {match.col_total:>7.0f}")
    print(f"cooperation rates: row {match.cooperation_rate[0]:.0%}, "
          f"column {match.cooperation_rate[1]:.0%}")

    rule("4. Round-robin tournament, all strategies")
    scores, _ = round_robin(pd_game, list(STRATEGIES), rounds=200, seed=0)
    print(scores.round(0).to_string())

    rule("5. Same tournament, but with 5% noise (mistakes)")
    noisy, _ = round_robin(pd_game, list(STRATEGIES), rounds=200, noise=0.05, seed=0)
    print(noisy.round(0).to_string())
    print(
        "\nCompare the tables: Grim Trigger is near the top without noise and "
        "near the bottom with 5% noise. One mistake ends cooperation forever.\n"
        "Two caveats worth writing down: (1) these totals are not comparable "
        "across different pools - the winner depends on who enters, and "
        "(2) Alternator doing well under noise is an artefact of a small pool, "
        "not a general result."
    )


if __name__ == "__main__":
    main()
