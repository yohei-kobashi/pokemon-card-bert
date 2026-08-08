# Reward design: crustle and crustle_geco

3.9% and 4.6% of the top-153. Two builds of one idea, and the two decks where our own reranker
already beats engine_v2 (53.3% and **67.3%** on the 11-deck evaluation) — so the reward here is
mostly an opponent model, and the terms say what breaks them.

## The cards that matter

```
Crustle (150 HP, NO Rule Box)
  ABL Mysterious Rock Inn  prevent ALL damage to THIS Pokemon from attacks by the
                           opponent's Pokemon ex
  ATK Superb Scissors {G}{C}{C} 120, damage unaffected by effects on their Active
Dwebble  ATK Ascension {C}  0 -- search the deck for its own evolution and evolve on the spot
Mega Kangaskhan ex (300 HP)  Run Errand: draw 2 while Active | Rapid-Fire Combo {C}{C}{C} 200+

crustle_geco only:
Cornerstone Mask Ogerpon ex (210 HP)
  ABL Cornerstone Stance   prevent all attack damage from the opponent's Pokemon WITH AN ABILITY
  ATK Demolish {F}{C}{C} 140, ignores Weakness/Resistance and effects
crustle only:
Shaymin  ABL Flower Curtain  no damage to our BENCHED non-Rule-Box Pokemon from their ex

both: Mist Energy x4, Jumbo Ice Cream x4 (heal), Xerosic's Machinations, Battle Cage,
      Spiky Energy, Grow Grass Energy, Hilda x4, Lillie's Determination x4
```

## The loop

There is no combo. **The deck wins by being un-attackable by the specific thing in front of
it**, and grinding 120 a turn from behind that immunity while Jumbo Ice Cream undoes whatever
does land.

```
Dwebble -> Ascension evolves itself into Crustle immediately (no search card needed)
Crustle is then immune to every attack from an ex Pokemon -- which is what almost the entire
meta attacks with -- and hits for 120 that ignores effects
Mega Kangaskhan ex is the body for anything Crustle cannot wall, and the draw engine
```

**The wall is matchup-conditional, and `geco` carries two different ones.** Mysterious Rock Inn
answers Pokemon ex; Cornerstone Stance answers Pokemon with an Ability. Those sets overlap but
are not the same, and which body should be Active is decided by what the opponent is attacking
with — the single most important decision in the deck, and one a generic evaluator cannot see
at all because both bodies look "healthy".

## Phi, in units of one prize

```
Phi_prize = 1.00 x (their prizes left - ours)

Phi_wall  = 0.40 * 1{our Active is immune to THEIR CURRENT ATTACKER}
            where immunity =  (Active is Crustle    AND their Active is a Pokemon ex)
                           or (Active is Cornerstone AND their Active has any Ability)
          + 0.10 * 1{a SECOND, differently-immune body is on our bench}   # geco only

Phi_grind = 0.20 * 1{our Active can pay its attack now}
          + 0.15 * clip(our_active_hp / max(1, our_active_maxhp), 0, 1)
          + 0.10 * min(#Jumbo Ice Cream in hand, 2) / 2

Phi_line  = 0.05 * min(#Crustle in play, 2) + 0.05 * 1{Mega Kangaskhan ex in play}
Phi_deny  = shared block (their Active cannot attack / hand size / asleep-paralysed)
```

`Phi_wall` is the whole design, and note what it is NOT: it does not reward having Crustle in
play, it rewards **having the right wall in the Active Spot against the attacker actually
there**. A build with both walls that stands the wrong one up gets nothing, which is correct
and is the behaviour to train.

## The counterplay these terms imply

Read the same potential from the other side and it says how to beat them, which is the point of
writing opponent models at all:

* **Attack with a non-ex body without an Ability.** Both immunities are conditional on what our
  attacker IS. `crustle-stall-vs-alakazam-ceiling` records that every aggression, snipe and
  disruption lever we tried against the stall family failed at ~49% — but those were all
  *stronger attacks*, not *differently-typed attackers*. The immunity clause is the thing to
  attack, not the HP.
* **Damage that is not an attack.** Mysterious Rock Inn prevents damage from ATTACKS. Ability
  damage — marnie's Froslass at Checkup, dusknoir's Cursed Blast placement — goes straight
  through. Battle Cage x2 in their list exists to shut exactly that off for the BENCH, so the
  window is their Active.

## What to verify before training

* Does the observation expose "this Pokemon has an Ability" for the opponent's Active, or must
  it be resolved through the card DB? (It must — `skills` on the card, not on the board slot.)
* Whether Ascension's self-evolution appears as an ATTACK option; a pilot that never chooses a
  0-damage attack will never evolve Dwebble, which is the same display trap as Seek Inspiration
  and Delightful Kiss.
