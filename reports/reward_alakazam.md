# Reward design: alakazam (the unprotected build)

5.2% of the top-153. Same damage formula as `alakazam_nz`, without the card that makes it a
tier deck — which is why the two sit at 5.2% and 11.1%.

## What differs from alakazam_nz

```
alakazam_nz   Neutralization Zone x1 (ACE SPEC stadium)
              -- non-Rule-Box Pokemon take NO damage from attacks by the opponent's ex/V
alakazam      Battle Cage x1 instead
              -- no damage counters on BENCHED Pokemon, from attacks and abilities, either side
              Dudunsparce x3 (nz runs 2)
```

Everything else is the same: Abra 4 / Kadabra 4 / Alakazam 4, Powerful Hand ({P}, place 2
damage counters per card in our hand = **20 damage per card held**), Psychic Draw on every
evolution, Dawn / Hilda / Poke Pad / Rare Candy x4 each, Enhanced Hammer x4.

## What that changes about the reward

`Phi_wall` — the 0.35 term that dominates `alakazam_nz` — **does not exist here**. Alakazam is
a 140 HP body standing in the Active Spot with no protection, so the deck is a race rather than
a siege, and the potential has to price survival instead of invulnerability:

```
Phi = Phi_prize + Phi_hand + Phi_survive + Phi_ready + Phi_engine + Phi_deny

Phi_hand    = 0.40 * clip(20*our_handCount / max(1, their_active_hp), 0, 1)
            + 0.10 * clip(our_handCount / 10, 0, 1)          # as in alakazam_nz

Phi_survive = 0.20 * clip(our_active_hp / 140, 0, 1)
            + 0.12 * 1{a SECOND Alakazam is in play}          # the replacement attacker
            + 0.08 * 1{Battle Cage is the stadium AND the opponent's deck spreads}

Phi_ready   = 0.20 * 1{an Alakazam in play holds >= 1 {P}}
Phi_engine  = 0.10 * min(#Kadabra+Alakazam in play, 3)/3*3 + 0.08 * 1{Dudunsparce in play}
Phi_deny    = shared block + 0.08 * clip((2 - #their special energy attached)/2, 0, 1)
```

`Phi_survive`'s second-Alakazam term is the substitute for the wall: `nz` does not need a
replacement because the first one cannot be knocked out; this build does, and losing the only
Alakazam ends the deck's damage output entirely.

The Battle Cage term is gated on the opponent spreading, exactly as in `dudunsparce_box` —
against dragapult's Phantom Dive or marnie's Froslass it protects the bench, and against
anything else it is a dead stadium.

## Note for the opponent model

Against us, the two Alakazam builds require **different counterplay**, which is the practical
reason to give them separate adapters rather than one shared "alakazam" policy: `nz` must be
attacked with a non-ex body or answered by removing its stadium, while this one can simply be
raced. A policy that learned only `nz` would waste turns hunting a stadium that is not there.

## Sources

Card texts from the local DB; deck comparison from `decks/alakazam.csv` vs
`decks/alakazam_nz.csv`. Shares from the 2026-08-08 top-153 scout. Live context:
[[live-alakazam-beats-us]], [[live-number1-deck-alakazam]].
