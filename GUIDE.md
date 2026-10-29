# Strategic Games Lab — your build plan

This is the mentoring half of the project. The code is done and tested; this file
tells you what to do with it, in what order, and what to say about it in an
interview.

Read it once end to end (20 minutes), then work through the phases. Total effort
is roughly 15–20 hours spread over three weeks of evenings, and most of that is
learning rather than typing.

---

## 0. What you already have

```
strategic-games-lab/
├── strategic_games/        the engine (5 modules, ~1200 lines with comments)
├── tests/                  52 tests, all passing
├── app.py                  Streamlit UI with 4 tabs
├── demo.py                 terminal demo — run this first
├── README.md               the portfolio-facing document
└── .github/workflows/      CI that runs pytest on every push
```

Verified working: `pytest` → 52 passed; `python demo.py` → solves all seven
preset games and runs two tournaments.

### Day 1: get it running on your machine (45 minutes)

```bash
cd strategic-games-lab
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                              # expect: 52 passed
python demo.py                      # read the output carefully
streamlit run app.py                # open the URL it prints
```

Do not skip `demo.py`. Reading the solved games in a terminal is how the maths
becomes concrete: Rock-Paper-Scissors prints 1/3, 1/3, 1/3 and you can see why.

Then push it to GitHub (Section 4) so you have a link to share from day one.

---

## Phase 1 — Understand the engine (3–4 hours)

You cannot defend code you did not read. Open the files in this order and, for
each one, do the exercise. Write the answers in a notebook — you will reuse them
in the interview.

### 1. `strategic_games/games.py`

A game is two NumPy matrices of the same shape. `row_payoffs[i, j]` is what the
row player gets when row plays `i` and column plays `j`.

**Exercise.** By hand, fill in the payoff matrices for the Prisoner's Dilemma
with T=5, R=3, P=1, S=0. Then print `prisoners_dilemma().payoff_frame()` and
compare. Fix any mistake before moving on.

### 2. `strategic_games/nash.py` — the heart of the project

Read `is_best_response` first (four lines), then `pure_nash_equilibria` (a double
loop over profiles). That is the whole idea of a Nash equilibrium: a profile where
neither player wants to move alone.

**Exercise.** Take the Coordination Game. Write down, on paper, the best response
of each player to each of the other player's actions. Confirm that exactly two
profiles are mutual best responses.

Then read `_solve_support` and `mixed_nash_equilibria`. The linear system solves
"make the other player indifferent". Verify the Coordination Game by hand:

- Row player is indifferent between Top and Bottom when
  `2q = 1 - q`, so `q = 1/3`.
- Hence the column player's mix is (1/3, 2/3), and symmetrically the row
  player's is too.
- Expected payoff: `2 · (1/3) · ... = 2/3` for each player.

Run `pytest tests/test_nash.py -v` and find the test that asserts this. Now you
know why the test exists.

### 3. `strategic_games/strategies.py` and `simulation.py`

A strategy is a function of history. `tit_for_tat` is three lines. `play_match`
loops, applies noise, looks up the payoff, appends to history.

**Exercise.** Work out by hand what Always Defect scores against Tit for Tat over
200 rounds. Round 1: 5 (T vs C). Then Tit for Tat copies the defection forever,
so 199 rounds of 1 each. Total = 204. Confirm with:

```python
from strategic_games import prisoners_dilemma
from strategic_games.simulation import play_match
m = play_match(prisoners_dilemma(), "Always Defect", "Tit for Tat", rounds=200)
print(m.row_total, m.col_total)      # 204.0 199.0
```

This is the single most useful exercise in the project: it turns "the shadow of
the future" from a slogan into arithmetic.

### 4. `app.py`

Four tabs: equilibrium analysis, repeated games, tournament, payoff sensitivity.
Read `_cached_match` to see how `st.cache_data` avoids re-running a simulation on
every UI interaction — that is a real engineering concern, not decoration.

---

## Phase 2 — Prove it yourself (2–3 hours)

Do not just read the tests. Write some.

**Task 1 — add a game.** Pick a game from a textbook that is not in the presets
(e.g. "Deadlock", "Chicken" with your own payoffs, or a 3×3 game). Add a factory
function to `games.py`, add it to `PRESETS`, then write three tests: number of
pure equilibria, the mixed equilibrium values, and one dominance claim. Run
`pytest`. If a test fails, decide whether the code or your hand calculation is
wrong — that decision is the skill.

**Task 2 — break it on purpose.** Change `>=` to `>` in `is_best_response` and run
the suite. Read which tests fail and why. Then revert. This is what the tests are
for, and it is a great thing to say in an interview.

**Task 3 — a test for a bug that does not exist yet.** What happens if someone
passes a 1×1 game? A matrix with a NaN? A repeated game with `noise=1.0`? Some of
these already raise `GameError`; check which, and add a test for any input you
think should be rejected but is not.

**Task 4 — coverage.** `pip install pytest-cov` then
`pytest --cov=strategic_games --cov-report=term-missing`. Anything uncovered is
probably either dead code or an untested path. Both are worth knowing about.

---

## Phase 3 — Make the interface yours (3–4 hours)

The app works, but a portfolio project needs one thing that is unmistakably your
own. Pick **two** of these, no more:

1. **A "what changed" panel.** Remember the previous equilibrium set in
   `st.session_state`; when the user edits a payoff, show "pure equilibria went
   from 2 to 1 — (Top, Left) is no longer stable because …". This directly serves
   the project's stated purpose: explain the consequences.
2. **Fictitious play.** Both players best-respond to the opponent's *empirical*
   frequency of play, updating each round. Plot the empirical frequencies
   converging to the mixed equilibrium. About 40 lines and it looks impressive.
3. **Evolutionary view.** Replace "two players" with a population: strategies
   reproduce in proportion to payoff, round after round. Plot strategy shares
   over time — you will watch defectors take over and then cooperation recover.
4. **Save/load games as JSON** plus a `?game=` URL parameter so a specific game
   can be shared as a link.

Keep the style consistent with the existing tabs and add a test for the logic you
add (UI code can stay untested; the maths behind it must not be).

---

## Phase 4 — Turn it into a portfolio piece (3–4 hours)

### 4.1 Git and GitHub

```bash
git init
git add .
git commit -m "Game theory engine: pure/mixed Nash solvers, 52 tests, Streamlit UI"
git branch -M main
# create an empty repo on github.com first, then:
git remote add origin https://github.com/<your-username>/strategic-games-lab.git
git push -u origin main
```

Then work in small commits, one idea each:

```bash
git add strategic_games/nash.py tests/test_nash.py
git commit -m "Add iterated elimination of strictly dominated strategies"
git push
```

Commit messages that say *what changed and why* are themselves a signal of
engineering maturity. "fixed stuff" is not.

### 4.2 Deploy it (free)

1. Push the repo to GitHub (public).
2. Go to <https://share.streamlit.io>, sign in with GitHub, "New app".
3. Repository → `strategic-games-lab`, branch `main`, main file `app.py`, deploy.
4. Copy the live URL into the top of your README and onto your résumé.

The `requirements.txt` is already correct for this; nothing else to configure.

### 4.3 Polish the README (30 minutes)

Checklist, in order of what a recruiter actually looks at:

- [ ] One sentence at the top saying what it does, in plain words.
- [ ] A screenshot or GIF of the app (take it after deploying).
- [ ] The live URL, near the top.
- [ ] How to run it locally, copy-pasteable.
- [ ] One short section on how the maths works — this is what separates you
      from someone who followed a tutorial.
- [ ] Known limitations, stated honestly.
- [ ] A roadmap with unchecked boxes (signals you know what comes next).

### 4.4 Write one short post (optional but effective)

A 500-word post — "Why Tit for Tat wins, and what my simulator showed when I added
noise" — with one chart, linked from the README. It gives you something to talk
about that is not just code, and it is the kind of artefact that gets shared.

---

## Interview script

**The 30-second version.** "I built a game-theory engine in Python. You give it a
payoff matrix; it computes best responses, finds pure equilibria by checking
mutual best responses, and finds mixed equilibria by support enumeration — solving
a small linear system that makes each player indifferent across their support.
Then a second module simulates repeated play between strategies like Tit for Tat
and Grim Trigger, including noise. It's tested with 52 tests pinned to
hand-computed values, and there's a Streamlit front end deployed on Streamlit
Community Cloud."

**Questions you should expect, and honest answers:**

| Question | What to say |
| --- | --- |
| *How do you find mixed equilibria?* | Support enumeration: for each candidate pair of support sets, solve the indifference linear system, then verify no action outside the support is profitable. Exponential in the number of actions, which is why the UI caps at 3×3. |
| *What's a limitation of support enumeration?* | It assumes isolated equilibria. In degenerate games a player can be indifferent across many actions, giving a continuum of equilibria, and the method can miss those. A general solver would need linear programming or Lemke-Howson. |
| *Why is (Defect, Defect) an equilibrium if it's worse for both?* | Nash equilibrium is about unilateral deviation, not efficiency. The distinction between equilibrium and Pareto efficiency is the whole point of the Prisoner's Dilemma. |
| *What did noise change?* | Grim Trigger collapses after one accidental defection — one mistake costs the rest of the game. Tit for Tat recovers; Generous Tit for Tat recovers faster. I have a test asserting the noisy match scores lower than the clean one. |
| *Is this code or maths?* | Both, and the tests are the bridge: every assertion is a hand-computed number, so a refactor that breaks the maths fails CI. |
| *What would you do next?* | Extensive-form games with backward induction, or correlated equilibria via linear programming — the roadmap in the README lists both. |
| *How would you scale it?* | Move the solvers into a FastAPI service, cache results, and use a proper solver (Lemke-Howson / LP) for larger games. The current design keeps the engine UI-independent precisely so this is easy. |

**One trap to avoid.** Do not claim the tool "predicts what people do". It
computes what *rational* players would do under stated assumptions. Saying that
clearly, unprompted, reads as maturity — overclaiming reads as the opposite.

---

## Traps that cost people weeks

- **Rewriting from scratch** because you want it to "feel like yours". Read it,
  extend it, keep it. Your contribution is Phase 2 and Phase 3.
- **Learning Docker/Kubernetes/FastAPI first.** None of them are needed for this
  to be an excellent project. Add one only after it is deployed.
- **Skipping the tests** because they "already pass". Run them after every change;
  the moment you extend the engine, they become your safety net.
- **Untested claims in the README.** If you write "handles degenerate games",
  either test it or delete the claim.
- **A giant README.** Three screens is the limit. Depth belongs in the code
  comments, which are already written for you.

---

## If you want to go further, in order of value

1. **Correlated equilibria via LP** — one `scipy.optimize.linprog` call, a genuinely
   different solution concept, and a strong interview story.
2. **Replicator dynamics** — connects game theory to biology and to ML.
3. **Extensive-form games** — game trees, backward induction, subgame perfection.
   Bigger job; do it as a second project if you enjoy this one.
4. **A FastAPI wrapper** — only worth it if you want a backend line on your résumé.

Good luck. Build Phase 2 first — the tests are where you actually learn the
theory.
