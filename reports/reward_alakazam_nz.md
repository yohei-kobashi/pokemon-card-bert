# Reward design: alakazam_nz

11.1% of the top-153 ladder. The deck our own agent loses to live (`live-alakazam-beats-us`:
2-11, and self-play was 48pt wrong about it), so getting this reward right matters twice —
once as an opponent model, once because the same terms say what beating it requires.

## The cards that matter

```
Alakazam (743) 140 HP  Stage 2, NO Rule Box
  ABL Psychic Draw   on evolving from hand: draw 3
  ATK Powerful Hand  {P}  0 printed -- "place 2 damage counters on their Active FOR EACH CARD
                     IN YOUR HAND"     -> 20 damage per card held, for ONE {P}
Kadabra (742)  ABL Psychic Draw (draw 2 on evolving)
Abra (741)     ATK Teleportation Attack {P}  10, switch this Pokemon with a benched one
Dudunsparce (66) ABL Run Away Draw  draw 3, then shuffle itself back into the deck
Neutralization Zone (ACE SPEC stadium)
                   prevents ALL damage to Pokemon WITHOUT a Rule Box, both sides, from
                   attacks by the opponent's Pokemon ex / V
Xerosic's Machinations x3   opponent discards down to 3 cards
Enhanced Hammer x4          discard a Special Energy from one of their Pokemon
Rare Candy x4, Dawn x4, Hilda x4, Poke Pad x4, Telepath Psychic Energy x4
```

## The loop

```
Alakazam is a 140 HP NON-ex body, and Neutralization Zone makes it immune to attacks from
Pokemon ex -- which is what essentially every meta deck attacks with. So it stands in the
Active Spot and cannot be answered.
  -> the deck then draws as hard as it can (Psychic Draw on every evolution, Run Away Draw,
     four searchers) because HAND SIZE IS THE DAMAGE FORMULA
  -> Powerful Hand for 20 x hand, one {P}, from behind a wall
```

**Our hand size is our attack.** No other deck in this set has that property, and it inverts
the usual reading of a full hand as "resources not yet spent". Here a card in hand is 20
damage already banked; playing it to no purpose is throwing damage away.

The three decks written so far now disagree about hand size in all three directions:

| deck | our hand | their hand |
|---|---|---|
| alakazam_nz | **large = damage** | small (Xerosic) |
| marnie_grimmsnarl | neutral | **small** (denial) |
| dudunsparce_box | neutral | **large** (Resentful Refrain does 50 per card) |

A single shared "hand" term cannot be right for more than one of them.

## Phi, in units of one prize

```
Phi = Phi_prize + Phi_hand + Phi_wall + Phi_ready + Phi_engine + Phi_deny
```

**Phi_prize = 1.00 x (their prizes left - ours).**

**Phi_hand — the damage formula itself.** Anchor it to what the hand can actually kill:
```
dmg   = 20 * our_handCount
Phi_hand = 0.40 * clip(dmg / max(1, their_active_hp), 0, 1)
         + 0.10 * clip(our_handCount / 10, 0, 1)
```
The first term is "can I kill what is in front of me right now", which is what the extra card
is worth; the second is a small unconditional pull so the policy still draws when nothing is
in range. Do NOT reward hand size linearly and without a cap -- that trains hoarding past the
point where more cards kill nothing.

**Phi_wall — Neutralization Zone, which is most of the deck's win rate.**
```
Phi_wall = 0.35 * 1{Neutralization Zone is the stadium in play}
                * 1{our Active has NO Rule Box}
                * clip(#{their in-play Pokemon ex} / 2, 0, 1)
```
Gated three ways on purpose: the stadium alone does nothing, it protects only the non-ex body,
and it is worth nothing against a deck attacking with non-ex Pokemon. It is an ACE SPEC, so
there is exactly one and it cannot be recovered from the discard -- losing the stadium to the
opponent's own stadium is a real event the potential should feel.

**Phi_ready** = `0.20 * 1{an Alakazam in play holds >= 1 {P}}`. One energy is the whole cost;
a second buys nothing, so this saturates immediately.

**Phi_engine** = `0.10 * min(#Kadabra+Alakazam in play, 3)/3 * 3` (the draw comes from
evolving, so bodies mid-line are the engine) `+ 0.08 * 1{Dudunsparce in play}`.

**Phi_deny** — the shared block, plus `0.06 * #{their Special Energy discarded this game}` is
NOT usable (not observable); instead read the board: `0.08 * clip((2 - #their special energy
attached)/2, 0, 1)`. Enhanced Hammer x4 is a quarter of the trainer count and only pays against
decks that run special energy, so this term must be gated on there being any.

## What to verify before training this one

* **Is `Powerful Hand` counted before or after the attack cost is paid?** The hand size read at
  the moment of attacking is what matters; if the observation is taken a step earlier the term
  is off by however many cards get played that turn.
* **Which stadium is in play** is `current.stadium` — confirm it names the card id and whose it
  is, since Neutralization Zone protects both players' non-ex bodies.
* The guides note the deck **plays around Xerosic deliberately** by leaving draw on the board
  (multiple Dudunsparce) and drawing only the minimum needed for the KO. That is a real policy
  our Phi does not express: it rewards a big hand, not a big hand *at the right moment*. Worth
  revisiting if the prize-matched test comes back weak.

## Sources

- [Syndicate Secrets - This Alakazam Tech Can Beat Dragapult | PokéBeach](https://www.pokebeach.com/2026/07/syndicate-secrets-this-alakazam-tech-can-beat-dragapult)
- [Alakazam Deck Guide (Pokémon TCG) | TCGplayer](https://www.tcgplayer.com/content/article/Alakazam-Deck-Guide-Pok%C3%A9mon-TCG/7eb46b82-9dc5-40d8-adf9-28cca05f070f/)
- [Alakazam - Deck Overview – Limitless](https://limitlesstcg.com/decks/350)
