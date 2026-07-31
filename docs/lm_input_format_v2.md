# LM input format v2 — complete board, single task

## Why v2

v1 fed the model a *summary* and trained it on three tasks at once. Two consequences,
both measured on the trained Qwen3.5-2B (2026-07-15/16):

1. **Ambiguous labels.** v1 emitted the hand as a COUNT (`h10`). The hand is not
   incidental — Alakazam's Powerful Hand is literally `20 x hand size`, and whether you
   hold Rare Candy decides the move. Identical serialized states therefore carried
   different "correct" actions. The model reproduced its *own training data* only
   **70%** of the time; some of that is irreducible label noise we created ourselves.
2. **Unlearnable reasoning.** The `reason` target was a copy of the winner's ACTUAL
   future logs (`tools/build_sft.py::_reasoning` slices `winner_steps[k+1:]`). That
   future depends on the opponent's hidden hand, deck order and coin flips — it cannot
   be predicted from the observation. Training on it teaches confident fabrication, and
   that is exactly what we saw in play (`') L3 L2 L5 L10(c410) L10(c1227) L7 L7'`).

v2: **complete observable board in, one action out.** Fix the input first; only add
reasoning later if a correct input still leaves a gap — and then only reasoning that is
*derivable from the observation* (e.g. `PH dmg = 20*h10 = 200 >= opp hp 150 -> KO`).

## Cost: adding the board is nearly free

Domain tokens make a card 1 token. Measured with the 11,924-token domain tokenizer:

| | tokens | chars |
|---|---|---|
| v1 state (no hand, no discard) | 90 | 161 |
| **v2 state (+hand +both discards)** | **157** | 282 |

+67 tokens against a 640 MAXLEN and a measured mean context of 383. Input tokens are
prefilled in parallel and KV-cached; output tokens are decoded serially at ~286 ms.
**Board detail and reasoning are not competing for the same budget** — the board is
cheap, the reasoning is not.

## ⚠️ NEVER build training data from replays / `visualize`

The competition's replay viewer shows the prize cards face-up — that view is the
**spectator's, not the agent's**. Verified side by side on the same game:

| | prize contents | opponent hand |
|---|---|---|
| replay / `visualize` array (what `tools/export_live_logs.py` writes into `logs/`) | **fully visible** (`{"id": 3, "name": "Basic {W} Energy"}`) | **fully visible** (`Cyrano`, `Crustle`, ...) |
| **obs handed to the agent in-game** | `[null, null, null, null, null, null]` | `[]`, only `handCount` |

Training on the replay view would teach the model to read prizes and the opponent's hand,
and it would then collapse in a real game where neither exists — a leak that is very hard
to notice after the fact, because the model would look excellent in offline eval.

`tools/gen_selfplay.py` is safe: it records the obs returned by `battle_start` /
`battle_select` verbatim (`"obs": o`), and `_clean_obs` only drops `search_begin_input`.
Confirmed on the stored self-play data (turn>=4): `OPP prize=[null x6] hand=null hc=5`,
`ME prize=[null x6] hand=[{345,...}]`. **Keep it that way — `logs/` is for analysis and
the visualiser, never for SFT.**

## What the simulator actually exposes (verified, not assumed)

The observation is written from `yourIndex`'s seat. `players[yi]` is us, `players[1-yi]`
the opponent. Hidden information is already hidden — there is nothing to leak:

| field | us | opponent |
|---|---|---|
| `hand` | **full contents** | `[]` (only `handCount`) |
| `prize` | `[null, ...]` — count only | `[null, ...]` — count only |
| `discard` | full contents | **full contents** (public, correctly) |
| deck | absent; `deckCount` only | absent; `deckCount` only |

So "everything the sim gives us" == "everything a legal player may know". v2 emits all
of it.

### Every available field

`current`: `turn`, `turnActionCount`, `yourIndex`, `firstPlayer`, `supporterPlayed`,
`stadiumPlayed`, `energyAttached`, `retreated`, `result`, `stadium`, `looking`

`player`: `active[]`, `bench[]`, `benchMax`, `deckCount`, `discard[]`, `prize[]`,
`handCount`, `hand[]`, `poisoned`, `burned`, `asleep`, `paralyzed`, `confused`

`pokemon`: `id`, `serial`, `playerIndex`, `hp`, `maxHp`, `appearThisTurn`, `energies[]`,
`energyCards[]`, `tools[]`, `preEvolution[]`

v1 used only: `id`, `hp/maxHp`, `energies` (as letters), `tools` (as a **count**).
v1 dropped: **hand contents, both discards**, `appearThisTurn`, `preEvolution`,
`energyCards` (special-energy identity), tool identity, `benchMax`, `firstPlayer`,
`stadiumPlayed`.

## Format

```
[ACT]
T<turn>.<actionCount>[/<flags>] first<0|1>
ME A[<pk>] B[<pk>,...] bm<benchMax> pz<n> dk<n> h<n> HAND[<c...>] DISC[<c...>] [<conds>]
OP A[<pk>] B[<pk>,...] bm<benchMax> pz<n> dk<n> h<n> DISC[<c...>] [<conds>]
STAD[<c>|-] [LOOK[<c...>]]
SEL <CTX> n<min>-<max> :: 0=<opt> 1=<opt> ...
```

### `LOOK[...]` — information a card effect just revealed

Hidden zones are not permanently hidden: card effects open them, and the model must see
what was opened. Present in the pool: **27 cards look at the top of your own deck**
(Pokegear 3.0 = top 7, Drakloak = top 2, Tatsugiri = top 6, Morpeko = top 1, ...),
**22 touch the opponent's deck** (Durant ex, Deino, Great Tusk, ...), and cards like
**Snorunt reveal a random card from the opponent's HAND**.

That information exists only in `current.looking`, only while it is visible, and the
option that picks from it is `area=LOOKING(12), index=N, cardId=None` — a reference into
that list, **with `sel.deck` EMPTY**. So the deck-search fix (`sel.deck[index]`) does NOT
cover this path. Measured: **94 such decisions in 30 games** — routine, not exotic.
Without `LOOK[...]` the model chooses "1 of these 7" while blind to all 7.

`?` marks an entry the schema allows to be face-down (`None`); never observed so far.

Real example (Pokegear, top 7 revealed):

```
STAD[c1264] LOOK[c1123 c1197 c345 c14 c1225 c1087 c18]
SEL TO_HAND n0-1 :: 0=card:c1197@LOOKING1 1=card:c1225@LOOKING4
```

* `<flags>` — per-turn resources already spent: `E` energyAttached, `S` supporterPlayed,
  `D` stadiumPlayed, `R` retreated. Absent = still available. These gate legality, so
  they must be explicit.
* `first<0|1>` — are we the player who went first (affects turn-1 rules / parity).
* `<pk>` = `c<id>:<hp>/<maxHp>[|E<letters>][|S<c...>][|T<c...>][|P<c...>][|new]`
  * `E<letters>` attached energy TYPES (as today)
  * `S<c...>` **special-energy card ids** — only when a special energy is attached;
    basic energy is already covered by `E`. Special energies change legality/effects
    (e.g. Legacy Energy changes prize value), so identity matters.
  * `T<c...>` **tool ids** (v1 emitted only `t1` — a count, which cannot tell Hero's
    Cape from anything else)
  * `P<c...>` **preEvolution ids** — the line under this Pokemon. Needed to know what
    it evolved from (devolve effects, Rare Candy legality, what a KO returns).
  * `new` = `appearThisTurn` — a body played this turn (cannot evolve/attack in some
    cases). Pure legality information; v1 dropped it.
* `HAND[...]` — **ours only**; the opponent's is `h<n>` alone, because the sim (rightly)
  does not show it.
* `DISC[...]` — **both**; discards are public.
* `bm<benchMax>` — bench capacity is not always 5 (effects change it), and "can I bench
  this?" depends on it.
* `pz` is a COUNT: prize *contents* are `null` for both seats, as they should be.

Order is fixed (`ME` then `OP`, active then bench, then piles) so the same board always
serialises to the same string — the model should never have to learn that two orderings
mean the same thing.

## Target

```
<action>
```

One action, exactly as `lm/actions.py` encodes an option today (`play:c1086`,
`attack:1072`, `attach:c4@ACTIVE0`, `retreat`, `end`, ...). No reasoning, no separator,
no rollouts. **One task**: every sample teaches the thing we deploy. v1's mix spent
~50% of its budget on `act`, 28% on `compare`, 22% on `reason`.

## Deliberately NOT included

* **Deck contents / order** — not exposed, and must not be.
* **Prize contents** — not exposed.
* **Opponent hand contents** — not exposed.
* **Card rules text** — the card id IS the rules text once the model has learned the id;
  spelling out "prevent all damage from ex" every turn would cost tokens on every card
  in play to restate something constant. (Revisit only if the model shows it cannot
  learn a card's behaviour from its id.)

## Open question for generation

v1 trained only on the WINNER's moves. Keep that (clean signal, halves the data) or use
both seats and condition on the outcome? Not decided here.
