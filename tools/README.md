# Agent improvement loop

Tools to run the **evaluate → specialize → re-evaluate** cycle over the 36
deck agents. All agent tuning lives in one file, `agents/tuning.json`, and is
baked into self-contained `agents/<deck>.py` files by the generator (so Kaggle
submissions stay `main.py` + `deck.csv` + `cg/`).

Run everything from the repo root with the venv active (the tools add `cg-lib`
to the path themselves).

## The cycle

1. **Evaluate** (deck-vs-deck round robin):
   ```
   python tools/evaluate.py --games 20
   ```
   Prints a "vs-field" win-rate ranking and saves the full matrix to
   `evaluations/eval_<timestamp>.json`. Low-ranked decks are the tuning targets.

2. **Diagnose a weak deck**:
   ```
   python tools/tune.py diagnose <deck>
   ```
   Shows its win rate, which Pokemon it actually attacks with, and auto-suggests
   `main_attackers` hints — the common failure mode is a 0-listed-damage scaling
   attacker (e.g. Mega Diancie ex, Dipplin) that the engine ignores.

3. **A/B a candidate tuning** (no files changed):
   ```
   python tools/tune.py test <deck> --main 766          # concentrate on id 766
   python tools/tune.py test <deck> --style spread
   ```
   Reports candidate vs current win rate.

4. **Commit the tuning** that helped: edit `agents/tuning.json`, e.g.
   ```json
   "mega_diancie": {"style": "spread", "main_attackers": [766]}
   ```
   then regenerate the agent files:
   ```
   python tools/generate_agents.py           # or: ... mega_diancie
   ```

5. **Re-evaluate** (step 1) and compare rankings.

## tuning.json schema

```json
"<deck>": {
  "style": "aggro | evolve | spread",   // proactive styles; 'control' is unused (measured worse)
  "main_attackers": [<cardId>, ...],     // optional: concentrate energy / promote / value scaling attacks
  "accel": ["<item name substring>"],    // optional: extra energy-accel items to play
  "play":  ["<item name substring>"]     // optional: deck engine items to always play
}
```

## Files

- `arena.py` — low-level self-play (`play`, `match`, `winrate_vs_random`, `random_policy`).
- `evaluate.py` — round-robin cross-play (multiprocess), writes `evaluations/`.
- `tune.py` — `diagnose` / `test` for a single deck.
- `generate_agents.py` — bake `tuning.json` into `agents/<deck>.py` (the "apply tuning" step).

## Notes

- The cg engine keeps one global battle pointer, so `evaluate.py` parallelises
  across **processes** (`--workers`), not threads.
- Win rate vs random is a fast proxy; the vs-field ranking (evaluate.py) is the
  real signal — a deck's "correct" slow plan can beat random less but win the
  mirror-field more, so trust the round robin for final decisions.
