# Reward design: mega_lucario_tr

3.3% of the top-153. A prize-trade beatdown whose defining constraint is a **self-imposed
attack lock**, which makes its reward a two-turn object rather than a one-turn one.

## The cards that matter

```
Mega Lucario ex (340 HP) x4
  ATK Aura Jab   {F}      130  AND attach up to 3 basic {F} from your DISCARD PILE to your
                               Benched Pokemon
  ATK Mega Brave {F}{F}   270  "During your next turn, this Pokemon can't use Mega Brave."
Lunatone  ABL Lunar Cycle  with Solrock in play, discard a basic {F} from hand to ...
Solrock   ATK Cosmic Beam {F} 70 -- does NOTHING unless Lunatone is on your Bench
Hariyama  ABL Heave-Ho Catcher  on evolving from hand (a gust effect)
          ATK Wild Press {F}{F}{F} 210, and 70 to ITSELF
Riolu, Makuhita | 15x basic {F} | Fighting Gong x4, Dusk Ball x4, Premium Power Pro x4
```

## The loop

```
turn N    Mega Brave     270 -- and Mega Lucario is now LOCKED OUT of Mega Brave next turn
turn N+1  Aura Jab       130, and pulls up to 3 basic {F} out of the DISCARD onto the bench
turn N+2  Mega Brave     270 again
```

Aura Jab is not a filler attack: it is the **refuel**, and the discard pile is the fuel tank.
Every {F} that gets discarded (to Ultra Ball, to Lunar Cycle, to a knocked-out body) is
recoverable through it, three at a time, onto the bench where the next Lucario is waiting.

**The lock is the whole sequencing problem.** A potential that scores "we can attack for the
most damage now" will fire Mega Brave whenever it is legal, which is exactly what makes the
following turn a 130. The value of a board is a function of *which turn of the cycle it is on*.

## Phi, in units of one prize

```
Phi_prize  = 1.00 x (their prizes left - ours)

Phi_cycle  = 0.30 * 1{Mega Brave is available THIS turn (not locked out)}
           + 0.15 * 1{their Active would die to 270 but not to 130}
             # i.e. the lock is being spent on a target that needs it
           - 0.20 * 1{Mega Brave is locked out AND their Active would have died to it}
             # the cost of having fired it on the wrong turn, made explicit

Phi_fuel   = 0.15 * clip(#basic {F} in our DISCARD / 3, 0, 1) * 1{Aura Jab is usable}
             # the discard is a resource for this deck, not a graveyard
           + 0.15 * 1{a benched Lucario already holds >= 2 {F}}   # the next attacker

Phi_body   = 0.20 * clip(our_active_hp / 340, 0, 1)
Phi_engine = 0.08 * 1{Solrock AND Lunatone are both in play}    # Cosmic Beam is dead alone
Phi_deny   = shared block
```

`Phi_cycle`'s negative term is unusual and deliberate: most potentials only add. Here the
mistake to be trained out is *using the big attack at the wrong moment*, and that only becomes
visible as a penalty on the state that follows it.

## What to verify before training

* **How the lock is represented.** "This Pokemon can't use Mega Brave next turn" may show up as
  the attack simply being absent from the menu, in which case `Phi_cycle`'s first term is read
  off the option list rather than from any board field. Check before wiring — if it is not
  observable, the rollout must track it, as it already does for Budew's item lock.
* **Lunar Cycle's full text is truncated** in the local dump at "discard a basic {F} Energy card
  from your hand in". Read it in full before pricing `Phi_engine`; if it accelerates energy the
  weight is too low, and if it only draws, 0.08 is right.
* `mega-lucario-live-matchup-profile` records that this deck goes 47% live and loses to walls
  and spread **structurally**, not through pilot error. The reward above will not fix a bad
  matchup; it should be judged on whether it improves the mirror and the races.
