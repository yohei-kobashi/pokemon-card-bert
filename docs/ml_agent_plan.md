# LM Agent Plan — Qwen3.5-0.8B for cabt Pokémon TCG

Status: **planning / scaffolding** (no training runs yet).
Owner: local `ptcgabc` project. Companion code: `tools/gen_selfplay.py`.

## 1. Goal & motivation

Our deck agents are piloted by a single shared heuristic engine (`agents/_engine.py`).
It is already strong in the **local bot arena**, but the local arena massively
**overrates passive stall/mill decks** and diverges from the **live Kaggle
leaderboard** (see memory `leaderboard-deck-scouting`, `anti-meta-matchups-jul-2026`).
The heuristic also mis-develops the board for some archetypes (empty bench, wrong
active) and cannot learn matchup-specific lines.

The aim of this plan is a **learned policy** (a small language model) that:

1. **distills** the heuristic engine's competent play from large-scale self-play, then
2. **surpasses** it on specific decks via reinforcement learning on the decisions
   that actually decide games.

The north-star metric is unchanged: **win rate of the LM agent vs the full
heuristic field**, measured with the existing `tools/evaluate.py` (the LM agent
conforms to `agent(obs) -> list[int]`, so it drops straight into the arena).

## 2. Model: Qwen3.5-0.8B

Confirmed specs (HF `Qwen/Qwen3.5-0.8B`, released after Jan-2026):

| Property | Value | Implication for us |
|---|---|---|
| Params | 0.8B, 24 layers, hidden 1024 | Full FT feasible on Colab (T4/A100) |
| Architecture | **Hybrid**: Gated DeltaNet + Gated Attention + **MoE** | New arch → verify `transformers`/`trl`/`peft` support & LoRA target modules |
| Context | **262,144 native** (linear-attention hybrid → cheap long ctx) | Feed whole board + long future-log reasoning windows freely |
| Vocab | **248,320**, tied embeddings | Adding a few hundred game tokens is negligible; resize touches input+LM head |
| Base variant | **Qwen3.5-0.8B-Base** | SFT starting point |
| License | Apache 2.0 | OK to use & bundle |

MoE means **active params ≪ 0.8B** → inference is light, which improves the odds
that a Kaggle submission can run per-move inference inside the time limit (still a
PoC, see §8).

## 3. Three stages (overview)

1. **Data generation** — large-scale heuristic self-play, captured losslessly as
   structured records (`tools/gen_selfplay.py`, format in §5).
2. **SFT** — fine-tune Qwen3.5-0.8B-Base to imitate the **winning** side: given the
   state at step *t*, output the next action, with the **winner's future logs**
   (window *t+2 .. t+6/t+11*) supplied as reasoning. Learn only from winners' steps.
3. **RL specialization** — take a chosen deck (proactive archetype recommended:
   `mega_lucario`, `alakazam`) and improve it on the **decisions that decide games**
   (online self-play RL, or offline DPO on mined pivotal decisions).

All training runs on **Colab**.

## 4. Shared foundation (build first — used by all three stages)

These four assets gate everything downstream:

1. **obs → text serializer** (deterministic, compact). Renders the acting player's
   board, opponent's public board, hand, prizes/deck counts, status, and the
   `select` context + option list. Card/attack IDs become tokens (§ below).
2. **text → action decoder** with **legality check + heuristic fallback**. Maps the
   LM's *semantic* output (e.g. `<attack><atk_982>`) back to `select.option`
   indices. On any illegal/unparseable output, **fall back to `_engine.act()`** —
   the arena forfeits on an illegal selection (instant loss), so we must never emit
   one.
3. **tokenizer extension**. Candidate special-token categories: `<card_{id}>`
   (311 in-pool, ~1267 total), `<atk_{id}>`, `<ctx_{SelectContext}>`,
   `<opt_{OptionType}>`, `<area_{AreaType}>`, `<energy_{type}>`, and action tags
   `<attack> <play> <retreat> <attach> ...`. **Whether special tokens actually
   shorten sequences vs. the existing 248k BPE must be measured before committing.**
4. **LM-agent adapter** implementing `agent(obs) -> list[int]` so the model is
   evaluated by `tools/evaluate.py` with zero extra harness.

## 5. Stage 1 — data generation (format)

`tools/gen_selfplay.py` drives heuristic self-play through the cg C-library and
writes one gzipped JSONL file per matchup plus a run-level `manifest.jsonl`.

### Design principles

- **Store structured obs, not pre-rendered text.** The serializer (§4) is still
  being designed and the cg RNG is **not seedable** (games cannot be reproduced),
  so faithful lossless capture is the only safe option. Text is produced later by a
  separate `build_sft.py`. Gzip absorbs the frame-to-frame redundancy.
- **Store both raw indices and semantic action.** `action` (raw `select.option`
  indices) is needed for replay/validation; `chosen` (the picked Option dicts) is
  index-independent and is what the LM learns to emit.
- **Denormalize filter keys** (`player`, `is_winner`, `turn`, `context`) onto each
  step so `build_sft.py` can filter (winner-only / MAIN-only) and window the future
  logs without joining the header.

### File layout

```
data/selfplay/<tag>/<deckA>__vs__<deckB>.jsonl.gz   # header line + step lines
data/selfplay/<tag>/manifest.jsonl                  # one lightweight line per game
```

### `game` header (first line of each file's game block)

```json
{
  "kind": "game", "schema": 1,
  "game_id": "mega_lucario__vs__crustle#00007",
  "decks":  {"0": [677, ...60], "1": [345, ...60]},
  "agents": {"0": "mega_lucario", "1": "crustle"},
  "first_player": 0,               // from State.firstPlayer (engine coin flip)
  "winner": 0,                     // 0/1; null on draw/timeout
  "end_reason": "result",          // result | timeout | draw | forfeit
  "n_steps": 36,
  "prize_remaining": {"0": 0, "1": 4},   // remaining prizes; taken = 6 - remaining
  "deck_remaining":  {"0": 21, "1": 17}
}
```

### `step` record (one decision)

```json
{
  "kind": "step", "game_id": "...#00007", "i": 12,
  "player": 0, "is_winner": true,       // is_winner = (player == winner); null on draw
  "turn": 7, "turn_action": 2,
  "context": 0, "is_main": true,
  "min": 1, "max": 1, "n_options": 5,
  "obs":    { "select": {...}, "logs": [...], "current": {...} },  // as passed to the agent
  "action": [3],                        // raw chosen option indices
  "chosen": [ { "type": 12, "attackId": 982, "cardId": 678, ... } ]  // picked Option dicts
}
```

Notes:
- The initial deck-selection call (`select is None`) does **not** occur in local
  arena play (decks are passed to `battle_start`), so there are no deck-selection
  steps; decks live in the header.
- A step is recorded **only if its selection was legal** (a forfeiting illegal move
  is not stored as training data; the game is closed with `end_reason: "forfeit"`).
- `obs["logs"]` are the new events since the previous selection, so the
  **future-log reasoning window** for step *i* is the concatenation of
  `obs["logs"]` from steps *i+1, i+2, …* (no overlap). `build_sft.py` slices this to
  *t+2 .. t+6/t+11*.

### How `build_sft.py` (later) consumes it

- **Winner filter**: keep steps with `is_winner == true` ("learn only the winner").
- **Target**: `obs@i` → `chosen@i` in semantic form (index-independent).
- **Reasoning (COMPACT, event-anchored — see §6.1)**: NOT the raw verbatim future
  log. Build it from subsequent steps' `obs["logs"]` but (a) cut the window at the
  next **significant event** (KO / prize taken / evolution — detected via `LogType`)
  rather than a fixed t+6/t+11, (b) **delta-encode** (emit only what changes), (c)
  **run-length** repeats (`attach ×3`, not three lines), (d) end with a one-line
  **abstracted outcome** (`→ KO active, +2 prizes`). This is a pure build-time
  transform on already-captured logs — **no change to `gen_selfplay` is needed**.
- **MAIN-only option**: filter on `is_main` (compatible with delegating
  sub-selections — targets, discards, coin flips — to the heuristic).
- **Split by game**, not by step (avoid leakage). Dedup near-duplicate states.

### Volume / storage estimate

Full round-robin (43 decks, 903 pairs) × 20 games × ~120 selections ≈ **~2M steps**.
Raw JSON ≈ 7 GB → **~1 GB gzipped per round**. `--lean` prunes None-valued option
fields if smaller footprint is needed; default keeps full obs (capture is
irreversible).

## 6. Stage 2 — SFT

- Start from **Qwen3.5-0.8B-Base**; extend tokenizer (§4-3) and
  `resize_token_embeddings` (tied → input + LM head).
- Sample: `prompt = serialize(obs@t)`; `target = [reasoning] + action@t+1`.
  **Mask the prompt** in the loss (train only on the target span).
- **Action as semantics**, decoded back to indices at inference (raw indices are
  brittle to option shuffling).
- Colab: `trl` `SFTTrainer` with LoRA (or full FT — 0.8B fits). Verify LoRA target
  modules for the DeltaNet/MoE hybrid. Data via Drive/HF Hub.
- Watch class balance introduced by winner-only filtering (first/second player,
  archetype mix).

### 6.1 Reasoning design — repetition countermeasures (ADOPTED)

Game logs are highly repetitive, and in this domain the *correct* play is itself
repetitive (attach energy → attack, every turn). Training the model to
autoregressively reproduce verbose future logs would teach high-probability loops
and risk degenerate repetition at inference. Two countermeasures are adopted (other
options — decoding penalties / `no_repeat_ngram` / unlikelihood loss / runtime
loop-detector — were considered and **deferred**, not needed if these two hold):

1. **Compact, non-repetitive reasoning target** (root-cause fix). Replace the
   verbatim future-log reasoning with: **event-anchored window** (cut at the next
   KO / prize / evolution, not a fixed length) + **delta encoding** (changes only) +
   **run-length** (`attach ×3`) + a one-line **abstracted outcome**
   (`→ KO active, +2 prizes, win`). Removes most repetition *before* training. Pure
   build-time transform in `build_sft.py` (§5) — `gen_selfplay` is unchanged.

2. **Learn-with / infer-without** (structural fix, also solves Kaggle latency). The
   reasoning is a **hindsight training signal**; at inference the future is unknown,
   so a generated "future log" is a hallucinated rollout that is both unreliable and
   loop-prone. Therefore:
   - `build_sft.py` emits **dual-mode samples**: *with-reasoning* (prompt → reasoning
     + action) **and** *action-only* (prompt → action), toggled by a mode flag/prefix
     token in the prompt. Same games, two views.
   - **At inference, decode the action directly** (action-only mode) — no reasoning
     generation → zero repetition risk and minimal latency. The with-reasoning
     samples still shape the representation during training.
   - Optional middle ground kept open: a **fixed-slot** short plan
     (`plan:[target,attack,prizes]`) instead of free-form reasoning if a small
     inference-time rationale proves useful — slot-filling cannot loop unboundedly.

## 7. Stage 3 — RL specialization

- **Environment on Colab**: `cg-lib/libcg.so` is a Linux `.so`; Colab is Linux, so
  it should load — **verify glibc/python compatibility first**. If it loads, the
  arena loop becomes the RL env directly.
- **Online RL (PPO/GRPO)**: LM pilots the chosen deck in self-play; reward = win, or
  shaped by prize differential (`prize_remaining` in the header). Inference count is
  huge → run **LM only on MAIN decisions, heuristic on sub-selections**, and batch
  inference.
- **Cheaper first step — DPO / preference**: no online env needed. Mine "decision
  that decided the game" pairs from logs (winner's choice vs. loser's/heuristic's
  alternative near the same state) and train preferences.
- **Pivotal-decision mining**: steps where prize differential swings, KO
  before/after, branch points.
- **Deck choice**: prefer **proactive** archetypes (`mega_lucario`, `alakazam`) —
  their local↔live divergence is small, so learned improvements transfer to live.

## 8. Risks & open questions

- **The heuristic is already strong.** SFT's ceiling is ≈ heuristic distillation;
  real value is **live generalization** and RL beyond the heuristic. But leaderboard
  data contains **decklists only, not action traces**, so the only imitation source
  is self-play → surpassing the heuristic depends on the RL design.
- **Submission format (biggest unknown — PoC early).** Kaggle submissions inline
  `main.py`. Bundling ~0.8B weights (~1.6 GB fp16 / ~0.5 GB int8) and running
  per-move inference within the time limit must be proven. If infeasible, fall back
  to: use the LM offline to discover lines/rules and **distill them back into the
  heuristic**.
- **New architecture compatibility.** MoE + Gated DeltaNet is recent; confirm the
  `transformers`/`trl`/`peft` versions, LoRA target modules, gradient checkpointing,
  and QLoRA quantization all work before large runs.
- **Determinism.** cg RNG is not seedable from Python → games are not reproducible;
  this is why Stage 1 captures losslessly.

## 9. Order of work

1. **Data format + `gen_selfplay.py` + manifest** (this doc + code). ✅ scaffolded
2. Environment PoCs (in parallel, cheap, high-value): (a) Qwen3.5-0.8B loads +
   LoRA/QLoRA trains on Colab; (b) per-move inference latency & weight bundling for
   a Kaggle submission.
3. Shared foundation §4: serializer, action decoder, tokenizer extension, LM-agent
   adapter.
4. `build_sft.py` (structured → SFT text): compact event-anchored reasoning +
   dual-mode (with-reasoning / action-only) samples per §6.1, then a first SFT run.
5. Evaluate the SFT agent vs the heuristic field (`tools/evaluate.py`).
6. Stage 3 RL on one proactive deck.
