# Reward design: dudunsparce_box

15.0% of the top-153 ladder, and second at the very top (20.0% of the top-50). A TANK deck,
which makes its reward the opposite shape from the two spread decks written so far.

## The cards that matter

```
Mega Lopunny ex (849) 330 HP
  ATK Gale Thrust  {C}  60, +170 MORE if this Pokemon MOVED FROM THE BENCH TO THE ACTIVE
                        SPOT THIS TURN                       -> 230 for one colourless
  ATK Spiky Hopper {C}{C} 160, damage unaffected by effects on their Active
Mega Froslass ex (861) 310 HP
  ATK Resentful Refrain {W}  50 damage FOR EACH CARD IN THE OPPONENT'S HAND
  ATK Absolute Snow {W}{C}{C} 150 + Asleep
Dudunsparce (66) 140 HP
  ABL Run Away Draw  draw 3, then shuffle this Pokemon AND its attached cards into the deck
Dunsparce (305) / Buneary (848)
  ATK Trading Places / Run Around  {C}  0 damage: switch this Pokemon with a Benched one

Wally's Compassion x4  heal ALL damage from one Mega ex; its Energy returns to your hand
Air Balloon x3         retreat -{C}{C}          Battle Cage x3   no damage counters on
Hand Trimmer x3        both hands down to 5                      BENCHED Pokemon, either side
Mist Energy x4 ({C}, and the holder is immune to attack EFFECTS)  Enriching Energy x1 ({C}, draw 4)
```

## The loop

```
cheap body Active (Dunsparce / Buneary / Snorunt)
  -> RETREAT it (Air Balloon makes that nearly free), bringing Lopunny up FROM THE BENCH
  -> Gale Thrust 230 for a single {C}
  -> Wally's Compassion heals the 330 HP body back to full and returns its Energy to hand
```

**Being on the BENCH is a strictly better position for Lopunny than being Active.** The 170
bonus is conditional on having moved up this turn, so a Lopunny that starts the turn Active is
worth 60 damage and one that starts it benched is worth 230. Every generic evaluator in this
codebase scores "our best attacker is Active" as good; here it is the state you have to leave.

## Phi, in units of one prize

```
Phi = Phi_prize + Phi_thrust + Phi_tank + Phi_refrain + Phi_engine + Phi_cage
```

**Phi_prize = 1.00 x (their prizes left - ours).**

**Phi_thrust — is the 230 available NEXT turn?** This is a positional term, and it is the deck:
```
Phi_thrust = 0.35 * 1{a Lopunny is on the BENCH and holds >= 1 energy}
           + 0.15 * 1{the Active can leave cheaply: retreat cost 0-1 after Air Balloon,
                      or it is a Dunsparce/Buneary that can Trading Places out}
           - 0.10 * 1{our only energised Lopunny is already Active}   # the 230 is spent
```

**Phi_tank — the HP the deck actually wins with, plus the heal that restores it.**
```
Phi_tank = 0.20 * (remaining HP of our healthiest Mega ex) / 330
         + 0.15 * min(#Wally's Compassion in hand, 2) / 2
         + 0.10 * 1{a Mega ex holds Mist Energy}      # immune to attack EFFECTS
```
Wally's Compassion is worth counting IN HAND, which is unusual for a potential: it converts a
damaged 330 HP body back to a full one, so holding it is holding most of a prize.

**Phi_refrain — THE SIGN IS INVERTED HERE.** Resentful Refrain scales with the OPPONENT's hand
size, so for this deck a full enemy hand is a target, not a threat:
```
Phi_refrain = 0.30 * 1{a Froslass with {W} is in play} * clip(their_handCount / 5, 0, 1)
```
`marnie_grimmsnarl`'s reward pays for cutting the opponent's hand; this one pays for their hand
being large. A single shared "disruption" term across decks would be wrong for one of them, and
this is the clearest example of why the designs are per-deck. Note Hand Trimmer floors both
hands at 5 rather than emptying them — it is resource denial, not Refrain setup, and the two
must not be conflated.

**Phi_engine**
```
Phi_engine = 0.10 * 1{Dudunsparce in play}        # Run Away Draw, and it recycles itself
           + 0.05 * min(#our benched bodies, 4) / 4
```

**Phi_cage — a matchup term, not a general one.**
```
Phi_cage = 0.12 * 1{Battle Cage is the stadium in play} * 1{the opponent's deck spreads}
```
Battle Cage stops damage counters on BENCHED Pokemon from attacks and abilities on both sides.
Against dragapult_dusknoir's Phantom Dive (6 counters to the bench) and marnie's Froslass and
Shadow Bullet it is close to a hard counter; against a deck that never spreads it is a dead
stadium slot. The opponent identity is already in the prompt, so gating on it is legitimate.

## What to verify before training this one

* **Does the engine expose "moved to the Active Spot this turn"?** The board slots carry
  `appearThisTurn` — check whether it is set by a RETREAT as well as by being played, because
  the whole Phi_thrust term depends on that flag meaning what Gale Thrust means. If it does
  not, the rollout has to track the switch itself, as it already does for Budew's item lock.
* **Retreat cost after Air Balloon** has to be read from the live board, not from the card:
  `prompt-lies-about-retreat-cost` recorded that the printed cost is what gets rendered and
  that 40-43% of legal retreats look unaffordable.
* Prize-matched separation, as for the others. Nothing trains until it passes.

## Sources

- [Standard Deck Tech - Mega Lopunny ex (Tokyo City League Champion) | Cardsrealm](https://pokemon.cardsrealm.com/en-us/articles/pokemon-tcg-standard-deck-tech-mega-lopunny-ex-tokyo-city-league-champion)
- [Tank Decks Are Back - Why Lopunny is Hopping to the Top | PokéBeach](https://www.pokebeach.com/2026/05/tank-decks-are-back-why-lopunny-is-hopping-to-the-top)
- [Mega Lopunny ex Gale Thrust abusing deck | PokéBeach forums](https://www.pokebeach.com/forums/threads/mega-lopunny-ex-gale-thrust-abusing-deck.156660/)
- [Mega Froslass ex - Ascended Heroes (ASC) #47 – Limitless](https://limitlesstcg.com/cards/ASC/47/decklists/jp)
