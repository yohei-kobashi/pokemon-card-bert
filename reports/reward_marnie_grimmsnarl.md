# Reward design: marnie_grimmsnarl

29.4% of the top-153 ladder — the single biggest deck we face, and the one whose reward is
most likely to be got backwards by a generic evaluator.

## The cards that matter (from the local DB, not from the archetype name)

```
Marnie's Grimmsnarl ex (648) 320 HP  Stage 2
  ABL Punk Up        on evolving from hand: search the deck for up to 5 basic {D} and
                     attach them to your Marnie's Pokemon in play
  ATK Shadow Bullet  {D}{D}  180 to the Active + 30 to ONE of their Benched

Munkidori (112) 110 HP  Basic
  ABL Adrena-Brain   with {D} attached: move up to 3 damage counters from one of YOUR
                     Pokemon to one of THEIRS
Froslass (104) 90 HP  Stage 1
  ABL Freezing Shroud  at Checkup, 1 damage counter on EVERY Pokemon with an Ability,
                       BOTH SIDES, except Froslass itself

Marnie's Impidimp (646) 70 -> Marnie's Morgrem (647) 100 -> Grimmsnarl ex
Yveltal (689), Budew (235), Tatsugiri (122), Snorunt (103)
9x basic {D} | Spikemuth Gym x4 (search a Marnie's Pokemon) | Rare Candy x3
Boss's Orders x4 | Team Rocket's Petrel x4 | Lillie's Determination x4 | Poke Pad x4
```

## The loop

```
Froslass          puts a counter on EVERYTHING with an Ability, ours included
Munkidori (x{D})  moves up to 3 of OUR counters onto THEIR board -- 4 copies = 12/turn
Grimmsnarl ex     Shadow Bullet 180 + 30, into a board already softened by the above
Punk Up           pays for all of it: 5 basic {D} out of the deck on the evolution turn,
                  onto the Active AND onto a benched Impidimp/Morgrem for NEXT turn
```

**Damage on our own board is a RESOURCE here.** Froslass damages our own Ability Pokemon on
purpose so Munkidori has counters to move. A potential that subtracts our damage — which is
what any generic board evaluator does — would train the policy to avoid the deck's own engine.
This is the single most important sign in the whole design.

The bound is real, though: counters are only fuel up to the transfer capacity actually in play
(3 per Munkidori holding {D}), and never on the body we need alive.

## Phi, in units of one prize

```
Phi = Phi_prize + Phi_threat + Phi_fuel + Phi_engine + Phi_line + Phi_deny
```

**Phi_prize = 1.00 x (their prizes left - ours).** The unit, as everywhere.

**Phi_threat — their board's distance to a Shadow Bullet KO.** The threshold is 180, not
Phantom Dive's 200:
```
for each of their Pokemon i (cap 3 bodies):
    ready_i = 1 if hp_i <= 180
    prog_i  = clip(dealt_i / max(1, maxHp_i - 180), 0, 1)
Phi_threat = 0.45*sum(ready_i) + 0.15*sum(prog_i)
```

**Phi_fuel — our own counters, valued only as far as they can be moved.**
```
cap    = 3 * #{Munkidori in play with {D} attached}      # what can actually be transferred
ours   = total damage counters on OUR Pokemon, EXCLUDING the Active attacker
Phi_fuel = 0.08 * min(ours, cap) / 3        # ~one Munkidori's worth = 0.08
         - 0.25 * 1{our Active attacker is within one hit of dying}
```
The subtraction is what stops "damage is good" from becoming suicide: counters anywhere except
the attacker are ammunition, counters ON the attacker are a lost 320 HP body and two prizes.

**Phi_engine — the two abilities that make the loop exist.**
```
Phi_engine = 0.12 * min(#Munkidori with {D}, 3) / 3 * 3    # up to 0.36; each is 3 counters/turn
           + 0.10 * 1{Froslass in play}                     # the counter source
```

**Phi_line — the attacker, and the NEXT attacker.** Punk Up's whole point is that it charges a
benched Impidimp/Morgrem at the same time, so "a second Marnie's body that can already pay
{D}{D}" is a first-class term, not a nicety:
```
Phi_line = 0.20 * 1{a Grimmsnarl ex can pay {D}{D} now}
         + 0.12 * 1{a SECOND Marnie's body already holds {D}{D}}
         + 0.05 * min(#Marnie's line bodies in play, 3)
```

**Phi_deny** — reuse the shared disruption block: their Active cannot pay any of its attacks
(0.12), sustained energy denial (0.08), hand size (0.06), asleep/paralysed (0.06), from turn 3.
Boss's Orders x4 is a lot of gusting, and its value shows up here and in Phi_threat collapsing.

## What to verify before training this one

* **Adrena-Brain's direction.** The local text is truncated at "move up to 3 damage counters
  from 1 of your Pokemon to" — the guides say the destination is the OPPONENT's Pokemon. Read
  the full text out of the DB before wiring Phi_fuel; if the destination were our own board the
  sign of the whole term flips.
* **Punk Up's target set.** "your Marnie's Pokemon" excludes Munkidori and Froslass, so the
  manual attachment for the turn has to go to Munkidori. If Phi rewards {D} anywhere, the
  policy will let Punk Up cover a body that Punk Up already covers.
* The prize-matched separation test, as for dusknoir. Nothing here trains until it passes.

## Sources

- [Pokémon TCG Strategy: Energize Your Deck with Marnie's Grimmsnarl ex | Pokemon.com](https://www.pokemon.com/us/strategy/pokemon-tcg-strategy-energize-your-deck-with-marnies-grimmsnarl-ex)
- [The Marnie's Grimmsnarl ex Deck Takes Spread Damage to a New Level | SNKRDUNK](https://snkrdunk.com/en/magazine/2025/06/13/pokemon-tcg-the-marnies-grimmsnarl-ex-deck-takes-spread-damage-to-a-new-level/)
- [Marnie's Grimmsnarl ex Deck Guide - Spell Mana](https://spellmana.com/marnies-grimmsnarl-ex-deck-guide-pokemon-tcg/)
- [Marnie's Grimmsnarl ex Deck List and Guide — Joseph Writer Anderson](https://www.josephwriteranderson.com/blog/marnies-grimmsnarl-ex-deck-list-and-guide)
