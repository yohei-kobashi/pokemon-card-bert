# Per-deck reward designs

One reward function per deck, written from how the deck is actually piloted, for the
per-deck-LoRA plan: each deck gets its own adapter trained against its own shaped return.

## Why per-deck and not one reward

The first design (`dragapult_dusknoir`) made the case by accident. Its potential rewards
banked damage on the OPPONENT's bench, because Phantom Dive's six counters exist to bring a
body into range. `marnie_grimmsnarl` needs the opposite sign on our OWN board: Froslass puts
counters on every Pokemon with an Ability including ours, and Munkidori then moves them onto
the opponent — so damage on our side is FUEL, not harm, exactly up to the transfer capacity in
play. A single reward that is right for one of these is wrong for the other, and a generic
board evaluator gets both wrong; `shaping-potential-refuted` measured that generic evaluator
picking the better move at chance.

## Method (the same for every deck, in this order)

1. **Read the cards, not the archetype name.** Dump the decklist with attack costs, damages and
   ability texts from the local DB. Half of this session's wrong turns came from reasoning about
   what a deck "is" instead of what its cards say.
2. **Read how humans pilot it** (internet, per `meta-assessment-use-internet`), and where the
   ladder's own agents can be observed, read their replays too (`tools/replay_profile.py`).
   The slowking rebuild failed because the list was copied without the plan that runs it.
3. **Name the loop**: the two or three actions that must chain for the deck to function.
4. **Write Phi as a POTENTIAL**, in units of one prize. The per-step reward is
   `gamma*Phi(s') - Phi(s)`, which leaves the optimal policy unchanged for any Phi
   (Ng, Harada & Russell 1999) — so an opinionated term costs sample efficiency and cannot
   teach a losing line.
5. **Verify every field exists.** `Phi_lock` read `cantPlayItem`, which is not in the
   observation, and scored 0 forever. Anything the board does not expose has to be passed in by
   whoever knows it.
6. **Validate before training**: unstratified separation, then PRIZE-MATCHED separation. If the
   deck-specific terms add nothing at a fixed prize lead, they are decoration.

## Status

| deck | ladder share | design | Phi validated | notes |
|---|---|---|---|---|
| dragapult_dusknoir | 0% (ours) | `docs/rl_dusknoir_design.md`, `tools/dusk_potential.py` | +67.8pt raw / **+22.0pt prize-matched** | the submission candidate |
| marnie_grimmsnarl | 29.4% | [reward_marnie_grimmsnarl.md](reward_marnie_grimmsnarl.md) | not yet | biggest deck on the ladder |
| dudunsparce_box | 15.0% | [reward_dudunsparce_box.md](reward_dudunsparce_box.md) | not yet | tank; the hand-size term is INVERTED vs marnie |
| alakazam_nz | 11.1% | — | — | |
| ogerpon_mono | 9.2% | — | — | |
| dragapult | 5.2% | — | — | shares most terms with dusknoir |
| alakazam | 5.2% | — | — | |
| crustle_geco | 4.6% | — | — | we already beat engine_v2 here (67.3%) |
| crustle | 3.9% | — | — | 53.3% |
| mega_lucario_tr | 3.3% | — | — | |
| cynthia_garchomp | 2.6% | — | — | |

Shares are the 2026-08-08 top-153 scout.
