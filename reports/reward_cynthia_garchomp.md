# Reward design: cynthia_garchomp

2.6% of the top-153, and the deck with the most terms that a board evaluator reads with the
wrong sign — two of them at once.

## The cards that matter

```
Cynthia's Garchomp ex (330 HP) x3
  ATK Corkscrew Dive  {F}     100  and you may DRAW UNTIL YOU HAVE 6 CARDS
  ATK Draconic Buster {F}{F}  260  DISCARD ALL ENERGY from this Pokemon
Cynthia's Roserade x3
  ABL Cheer On to Glory  attacks by your Cynthia's Pokemon do **30 MORE** damage to their Active
Cynthia's Gabite x4
  ABL Champion's Call    search your deck for a Cynthia's Pokemon into your hand
Cynthia's Spiritomb
  ATK Raging Curse {C}   10 damage FOR EACH DAMAGE COUNTER on ALL your Benched Cynthia's Pokemon
Cynthia's Gible x4, Cynthia's Roselia x4
Cynthia's Power Weight x3 | Rock Fighting Energy x4 | Fighting Gong x3 | Boss's Orders x4
```

## Two inversions in one deck

**1. Roserade stacks on the BENCH.** `Cheer On to Glory` is a passive that adds 30 to every
attack our Cynthia's Pokemon make. Three Roserade on the bench is **+90 to every attack**, which
turns Corkscrew Dive from 100 into 190 and Draconic Buster from 260 into 350. A generic
evaluator scores a bench of 130 HP support Pokemon as weak board presence; here each one is a
permanent damage upgrade and they are the reason the deck functions.

**2. Damage on our own bench is Spiritomb's damage.** Raging Curse does 10 per damage counter
on all our benched Cynthia's Pokemon — the same "our damage is fuel" shape as
`marnie_grimmsnarl`, arrived at by a completely different route. Counters our opponent puts on
our bench are ammunition.

## The energy problem

Draconic Buster discards **all** energy from Garchomp, so the deck alternates like
`mega_lucario_tr` but worse: after the 260 the attacker is empty, not merely locked. Corkscrew
Dive at {F} is the recovery turn and it refills the hand to 6 at the same time.

## Phi, in units of one prize

```
Phi_prize  = 1.00 x (their prizes left - ours)

Phi_buff   = 0.12 * min(#Cynthia's Roserade in play, 3)      # up to 0.36 -- the deck's engine
             # NOT capped at 1: the third one is worth as much as the first, +30 each

Phi_punch  = 0.35 * clip(best_attack_damage / max(1, their_active_hp), 0, 1)
             where best_attack_damage INCLUDES the Roserade buff
           + 0.10 * 1{Garchomp can pay {F}{F} now}           # Draconic Buster is available

Phi_refill = 0.15 * 1{Garchomp holds 0 energy AND we hold >= 1 {F} to re-attach}
             # the turn after Draconic Buster is a known, planned hole -- reward being able
             # to climb out of it rather than penalising the hole itself

Phi_curse  = 0.06 * clip(counters_on_our_benched_cynthias / 6, 0, 1)
             * 1{Spiritomb is in play}
             # fuel, exactly as in marnie -- and worthless without the body that spends it

Phi_line   = 0.05 * min(#Gabite in play, 2)    # Champion's Call, one search each per turn
Phi_deny   = shared block
```

`Phi_buff` deliberately does **not** saturate at one Roserade. Most "engine present" terms in
these designs cap at 1-2 because the second copy adds little; here the ability is additive by
the card's own text, so the potential has to be additive too or it will train the policy to
stop at one.

## What to verify before training

* **Does Cheer On to Glory stack?** The text does not say "you can't use more than one", which
  in this game usually means it does — but confirm against the engine's damage calculation
  before weighting three copies at 0.36, because if it does not stack the term is 3x too big.
  `hidden-state-from-blob` gives a way to check: compare a computed damage against the engine's
  own CalcDamage on a board with one Roserade versus three.
* **Whose damage counters count for Raging Curse** — ours only, benched only, Cynthia's only.
  All three restrictions are in the text and all three are easy to drop when implementing.
* Prize-matched separation before training, as everywhere.
