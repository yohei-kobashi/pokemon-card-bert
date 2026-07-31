# Authoring `card_roles` (tuning.json)

`card_roles` maps every unique card id in a deck to ONE importance tier. The engine turns
that into a score (`_TIER_VALUE`, defined identically in `agents/_engine.py` and
`agents/engine_v2.py`):

| tier | value | meaning |
|---|---|---|
| `win` | 900 | the win condition itself — the card that actually takes prizes |
| `engine` | 700 | finds / enables / accelerates the win condition |
| `line` | 600 | the evolution line's OTHER pieces (pre-evolutions of a `win` card) |
| `fuel` | 400 | energy (further need-adjusted at runtime by `_card_need`) |
| `tech` | 200 | situational, matchup-dependent |
| `filler` | 50 | never actually wanted |

**Searches fetch from the top; discards go from the bottom.** So a tier is the answer to
two questions at once: *"if I could pull one card from my deck, how badly do I want this?"*
and *"if I must throw one card away, how much does it hurt?"*

## Why this is hand-authored

Both mechanical shortcuts were tried and measured to fail:

* **Card type** (Pokemon > Energy > Supporter > Item) is the global hierarchy this
  replaces. It is wrong per-deck in both directions — see the failures below.
* **Copy count** carries no signal: a finished deck is almost all 4-ofs (alakazam 44/60 =
  73%, crustle 52/60 = 87%), so it separates nothing. Measured: 4-of discards stayed at 52%.

## Measured failures to calibrate against

These are real regressions this system exists to prevent. Read them before judging a card.

* **Printed damage is not the win condition.** Alakazam's *Powerful Hand* does
  `20 x hand size`, so the DB prints **damage 0** — it scored 50, LAST among its own
  line, below Fezandipiti ex (70). Resolved searches then never fetched the deck's own win
  condition (-6.9 / -3.7pt, reproduced on two seed sets). Alakazam is `win`.
* **Items can be the engine.** Alakazam's Rare Candy / Buddy-Buddy Poffin / Dawn score
  "Item = 25" and were discarded FIRST. They are `engine`.
* **Energy can be the damage.** Hydrapple ex / Teal Mask Ogerpon ex read "30 more damage
  for each Energy attached" — "Pokemon > Energy" made every search fetch a body instead
  (-13.0pt). Energy in such a deck is `fuel`, and `_card_need` adds a runtime bonus for
  energy-scaling attackers; do NOT try to encode that by inflating the tier.
* **ACE SPEC is a 1-of BY RULE, not by preference.** Hero's Cape / Prime Catcher / Maximum
  Belt etc. are capped at 1 because they are too strong to run in multiples. Never `tech`
  on the grounds of being a single copy — judge the effect.

## Rules

1. **Every unique id in the deck gets exactly one tier.** 100% coverage — an unclassified
   card silently falls back to the global guess this system exists to override.
2. **Judge the card IN THIS DECK.** Boss's Orders is `engine` in a deck that wins by
   gusting up a benched sitter, and `tech` in one that just attacks the Active. The same
   id legitimately gets different tiers in different decks.
3. **`win` is scarce.** Usually 1–2 ids: the attacker(s) the deck actually wins with. If
   a deck has several equal attackers (a "box" deck), several `win` are fine.
4. **`line` = pre-evolutions of a `win`/attacker**, plus Rare Candy targets' middles.
   A Basic that is only ever a stepping stone is `line`, not `win`.
5. **Draw/search supporters and items are `engine`** when the deck needs them every game
   (Iono, Professor's Research, Ultra Ball, Nest Ball, Poffin, Rare Candy, Night Stretcher
   for a key body, energy accel like Electric Generator / Dark Patch).
6. **`tech`** = answers you want in some matchups: single-copy gusts you can win without,
   stadiums that only matter vs one archetype, tools, switch cards, disruption in a deck
   that is not built around it.
7. **`filler`** is genuinely rare — use it only for a card the deck would cut first.
   Do not use it as "I'm unsure".
8. **Special Energy that does more than pay costs** (e.g. draws, accelerates, adds damage)
   is `engine`, not `fuel`, when the deck is built on that rider.

## Workflow

```
python tools/dump_deck.py <deck>          # full rules text of every card in the deck
# author the mapping into agents/tuning.json under <deck>.card_roles
python tools/generate_agents.py <deck>    # bake into agents/<deck>.py (legacy engine)
```

`engine_v2` reads `card_roles` straight out of the tuning entry it is handed as `profile`
(no regeneration step needed); the legacy engine reads it from the baked `HINTS`.

**`_TIER_VALUE` is defined in BOTH `agents/_engine.py` and `agents/engine_v2.py` — keep the
two in sync.** They are separate because each engine is bundled standalone into a
submission (`main.py` + `deck.csv` + `cg/`), so neither may import the other.

## Coverage

All 60 decks are authored. Check with:

```
python -c "
import json; t = json.load(open('agents/tuning.json'))
miss = [d for d, v in t.items() if not v.get('card_roles')]
print('missing card_roles:', miss or 'none')"
```
