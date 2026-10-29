# Strategic Games Lab


**Live demo:** https://strategic-games-lab-kvf5ye2xz9urrtejwszope.streamlit.app/

A small, tested game-theory engine with an interactive web front end. Enter a
strategic situation as a payoff matrix, and the engine finds the Nash equilibria,
explains why each one is stable, and then simulates how different strategies
actually behave when the game is played over and over.

Built with Python, NumPy, Streamlit, Plotly and pytest. No database, no server,
no dependencies beyond those five.


---

## What it does

| Module | What you get |
| --- | --- |
| **1. Equilibrium engine** | Best responses, pure Nash equilibria, strictly/weakly dominated strategies, iterated elimination, and *all* mixed equilibria via support enumeration. |
| **2. Repeated game simulator** | Nine strategies (Always Cooperate, Always Defect, Tit for Tat, Tit for Two Tats, Grim Trigger, Pavlov, Generous Tit for Tat, Alternator, Random) played head to head, with optional noise. Round-robin tournaments, cumulative-payoff charts. |
| **3. Explorer UI** | Edit any payoff cell and watch the equilibrium set change; drill into any match round by round. |

The engine also writes a plain-English explanation of each equilibrium, e.g.

> **(Defect, Defect)** is a Nash equilibrium. Against Defect, the row player's
> best reply is Defect (1 vs Cooperate 0). Against Defect, the column player's
> best reply is Defect (1 vs Cooperate 0). Neither player gains by switching
> alone, so the profile is stable.
>
> Row action 'Cooperate' is strictly dominated by 'Defect' - a rational row
> player never plays it.

---

## Quickstart

```bash
# 1. get the code and create an isolated environment
cd strategic-games-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. prove the maths works (52 tests, ~2 seconds)
pytest

# 4. see the engine work in the terminal, no browser needed
python demo.py

# 5. open the app
streamlit run app.py
```

`streamlit run app.py` prints a local URL (usually <http://localhost:8501>).

---

## Project layout

```
strategic-games-lab/
├── app.py                      # Streamlit UI (4 tabs, ~350 lines)
├── demo.py                     # terminal demo: solves every preset game
├── strategic_games/
│   ├── __init__.py             # public API - import from here
│   ├── games.py                # Game class, validation, 7 preset games
│   ├── nash.py                 # best responses, dominance, pure + mixed Nash
│   ├── strategies.py           # the nine repeated-game strategies
│   ├── simulation.py           # matches, round-robin tournaments
│   └── viz.py                  # Plotly figures (reusable outside the app)
├── tests/
│   ├── test_games.py           # construction and validation
│   ├── test_nash.py            # every classic game, checked by hand
│   └── test_simulation.py      # strategies, noise, tournaments
├── requirements.txt
├── pytest.ini
└── .github/workflows/tests.yml # CI: run pytest on every push
```

---

## Using it as a library

```python
from strategic_games import (
    Game, prisoners_dilemma, pure_nash_equilibria,
    mixed_nash_equilibria, explain, round_robin,
)
from strategic_games.simulation import play_match

# a preset
game = prisoners_dilemma(temptation=5, reward=3, punishment=1, sucker=0)
print(game.to_text())
print(pure_nash_equilibria(game))       # [(Defect, Defect)]

# your own game, from a table of (row payoff, col payoff)
custom = Game.from_table(
    [[(3, 3), (0, 5)],
     [(5, 0), (1, 1)]],
    row_actions=["Cooperate", "Defect"],
    col_actions=["Cooperate", "Defect"],
    name="My game",
)
for line in explain(custom):
    print("-", line)

# play it out
match = play_match(game, "Tit for Tat", "Always Defect", rounds=200, seed=0)
print(match.row_total, match.col_total, match.cooperation_rate)

scores, _ = round_robin(game, ["Always Defect", "Tit for Tat", "Grim Trigger"], rounds=200)
print(scores)
```

---

## How the maths works

**Best responses.** For each column `j`, the row player's best replies are the
`i` maximising `A[i, j]`. Symmetrically for the column player. A profile
`(i, j)` is a **pure Nash equilibrium** exactly when both conditions hold at
once — a mutual best response.

**Dominance.** Row `i` is strictly dominated when some row `k` pays strictly
more against *every* column action. Deleting strictly dominated actions can
never delete an equilibrium, so **iterated elimination** shrinks the game without
changing the answer. In the Prisoner's Dilemma it solves the game outright.

**Mixed equilibria** are found by *support enumeration*. For each pair of
candidate support sets `I ⊆ rows`, `J ⊆ cols` with `|I| = |J| = k`:

1. Solve the linear system that makes the row player indifferent across `I`
   given the column player's mix `y` over `J`:
   `A[I, J] · y = u · 1` together with `Σ y = 1`.
2. Solve the mirrored system for the row player's mix `x` making the column
   player indifferent.
3. Reject the candidate unless `x > 0`, `y > 0` (otherwise a smaller support
   would describe it), and no action outside the support beats the mix.

That is `O(4^min(m,n))` support pairs — instant for the 2×2 and 3×3 games this
tool targets, and the reason the UI caps matrices at 3×3.

**Repeated games.** A strategy is a pure function of history:
`move = strategy(my_history, opponent_history)`. Noise flips a chosen move with
probability `p` *before* it is recorded, so strategies react to what actually
happened — which is why Grim Trigger collapses after a single mistake and
Generous Tit for Tat recovers.

---

## Known limitations

Stated plainly, because knowing where a model breaks is part of the model:

- **Isolated equilibria only.** Support enumeration assumes each equilibrium is
  isolated. In degenerate games (a player indifferent across several actions)
  there can be a continuum of equilibria; this tool reports the isolated ones
  and can miss the continuum.
- **Two players, simultaneous moves, complete information.** No extensive-form
  games, no sequential moves, no incomplete information (no Bayesian games).
- **Repeated games are 2×2.** The simulator maps move "C" to the first action
  and "D" to the second; the strategy vocabulary is cooperate/defect.
- **Payoffs are treated as utilities.** The engine never questions whether the
  numbers are comparable — that is an assumption, not a result.
- **Tournament winners depend on the pool.** With only Always Cooperate and
  Always Defect in the pool, defection wins. Reciprocity only pays when
  reciprocators are present. The app lets you see this happen.

---

## Tests

```bash
pytest              # 52 tests
pytest -v           # one line per test
pytest tests/test_nash.py::test_coordination_game_mixed_equilibrium_values
```

The suite pins down results you can verify by hand: the Prisoner's Dilemma has
exactly one equilibrium and it is Pareto-dominated; Matching Pennies has none in
pure strategies and one at 50/50; Coordination has two pure plus one at 1/3–2/3;
Rock-Paper-Scissors is uniform; Hawk-Dove mixes at value/cost. If a refactor
breaks the maths, a test with a hand-computed number fails.

CI runs the same command on every push (`.github/workflows/tests.yml`).

---

## Roadmap

- [ ] Extensive-form games: game trees and backward induction
- [ ] Correlated equilibria via linear programming (`scipy.optimize.linprog`)
- [ ] Replicator dynamics for evolutionary stability
- [ ] Save and load games as JSON
- [ ] A small REST API (FastAPI) so the engine is usable from other apps
- [ ] Fictitious play: watch learning converge to a mixed equilibrium

---

## Background

The equilibrium concepts come from John Nash (1950) and the dominance/elimination
machinery from standard texts (Osborne, *An Introduction to Game Theory*;
Fudenberg & Tirole, *Game Theory*). The strategy tournament design follows
Robert Axelrod's 1980 Prisoner's Dilemma tournaments, where Tit for Tat won:
nice, retaliatory, forgiving, and clear.
