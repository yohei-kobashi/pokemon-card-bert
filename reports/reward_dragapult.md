# Reward design: dragapult (the straight build)

5.2% of the top-153, and 8.0% of the top-50 — it climbs. This is the build the ladder actually
plays: **all eight dragapult teams in the top-153 reconstruct as the straight list, and zero
run the Dusknoir line** (checked card by card, 60/60 on one of them). Human competitive play
rates Dusknoir higher; the Kaggle field does not play it at all.

## What differs from dragapult_dusknoir

```
absent   Duskull / Dusclops / Dusknoir  -- so NO Cursed Blast: no 13-counter placement, and
                                          no deliberate self-KO to buy a prize's worth of reach
present  Munkidori x2 (Adrena-Brain: with {D}, move up to 3 counters from OUR Pokemon to
                       THEIRS)          -- the damage-moving line instead
         Dunsparce / Dudunsparce        -- Run Away Draw
         Crispin x2, Risky Ruins x2, Rare Candy x2
same     Dreepy 4 / Drakloak 4 / Dragapult ex 3, Budew, Fezandipiti ex, Meowth ex
         Phantom Dive {R}{P} 200 + 6 counters on their bench
```

So Phi is `dusk_potential.phi` with one term removed and one added.

## Phi

Start from the dusknoir potential (`tools/dusk_potential.py`), which is already validated
(+67.8pt raw, +22.0pt prize-matched), and change two things:

**REMOVE nothing structurally** — the Cursed Blast trade was never an explicit term; it is
priced by `Phi_prize` falling by one when the body knocks itself out. With no Dusknoir in the
list that path simply never occurs, and the potential is unchanged.

**ADD the Munkidori transfer, with the same shape marnie uses:**
```
cap  = 3 * #{Munkidori in play holding {D}}
ours = damage counters on OUR Pokemon, EXCLUDING the Active attacker
Phi_fuel = 0.06 * min(ours, cap) / 3
         - 0.25 * 1{our Dragapult ex is within one hit of dying}
```
Weighted lower than in marnie_grimmsnarl (0.06 vs 0.08) because there are two Munkidori here,
not four, and no Froslass generating counters on purpose — the fuel is incidental damage rather
than a manufactured resource.

**KEEP** `Phi_spread` (distance to a Phantom Dive KO, threshold 200), `Phi_energy` (the
{R}{P} split with partial credit for the first attachment), `Phi_line` (Dreepy 0.05 /
Drakloak 0.10 / Dragapult 0.15), `Phi_lock` (Budew, passed in by the rollout), and the shared
`Phi_deny`.

## Why this deck is worth its own adapter anyway

It shares most of its terms with dusknoir, so the argument for a separate LoRA is not the
reward — it is that **this is the build we will actually be played against**. An opponent model
trained on the Dusknoir version would expect a 13-counter placement that never comes and a
prize trade that never happens. The 8 of 8 reconstruction is the evidence.

## Sources

Scouted from replays (`tools/scout_decks.py`, 2026-08-08); the 60/60 reconstruction of team
`RtoABC` shows Dreepy x4 / Drakloak x4 / Dragapult ex x3 / Munkidori x2 / Budew / Dunsparce /
Dudunsparce / Fezandipiti ex / Meowth ex / Dudunsparce ex, with no Duskull line.
Human-meta framing: [[human-meta-vs-kaggle-field]].
