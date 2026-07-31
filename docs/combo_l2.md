# Combo L2 — per-deck bespoke policies (2026-07-11)

L2 = per-deck subclasses of `ComboPolicy` (engine_v2.py), dispatched by a `l2` key
in `agents/tuning.json`. Generic `combo` L1 = L0 + mild dig (a generic assemble-gate
starves decks — divergence 0.0% vs L0). L2 gives each combo deck a **rigorous
`combo_online` completion check** and **opponent-board-aware execution timing**.

> **⚠ CRITICAL FINDING (2026-07-11, measured after implementation)**: a combo_online
> GATE that WITHHOLDS attacks until the combo completes is **catastrophic** in this
> AI environment — field eval scored the gated L2 at **−8.4% (z −4.3) vs L0** (hydrapple
> −28!). AI opponents punish passivity: combo decks must pressure ASAP. So the gate was
> REMOVED. The rigorous `combo_online` completion-check + opponent-board reads are KEPT,
> but used only for **correct scaling-attack LETHAL detection** (L0 under-rates 0-damage
> scaling attacks) and setup priority (rockets TR-bench, slowking→toolbox), NOT to skip
> attacks. Result: combo L2 ≈ L0 (neutral, −0.1%). Conclusion: **combo does not benefit
> from L2 gating here; it wins like everything else by attacking when able.**


## Shared L2 contract
Each deck overrides:
- `_expected_dmg(ctx)` — the REAL payoff damage (the DB lists 0 for scaling attacks),
  computed from the deck's scaling rule. Used for lethal detection.
- `combo_online(ctx)` — True only when the payoff is genuinely ready (payoff attacker
  active + the deck-specific resource threshold met).
- `decide_attack(ctx)` — **execute** (attack) if: (a) lethal now (`_expected_dmg >=`
  opp active HP), or (b) `combo_online`, or (c) **defensive**: the opponent can KO my
  active next turn (`opp_threat` CAN_KO_ME_NOW) and I have any real attack — don't sit
  and lose the attacker for nothing. Otherwise **withhold** (return None → keep building).
- Opponent info: every execution decision reads `ctx.opp.active.hp`, `ctx.ko_targets`,
  `ctx.opp_threat`, `ctx.opp.bench` (board width) — never fire blind.

Guardrail: withholding only makes sense while a build action exists; the MAIN ladder
runs evolve/ability/attach/trainer BEFORE attack, so a withheld turn still develops.
`turnActionCount>=40` and lethal always break the gate.

---

## alakazam — hand-size nuke (STRONG combo)
- **Payoff**: Alakazam (743) *Powerful Hand* (cost 1 {P}): place **2 damage counters ×
  hand size** on the opponent's Active → `20 × handCount` damage to the Active.
- **Engine**: Kadabra (742) / Alakazam *Psychic Draw* (draw on evolve), Dudunsparce (66)
  *Run Away Draw* (draw 3), Dunsparce (305), Poké Pad / Hilda / Dawn. Enhanced Hammer denial.
- **`_expected_dmg`** = `20 × me.hand_count` if Active is Alakazam, else 0.
- **`combo_online`** = Active is Alakazam (743) AND `hand_count >= 6` (≥120 dmg). Below
  that, keep drawing to grow the hand (a bigger hand = a bigger nuke).
- **Execution / opponent info**: fire early if `20×hand >= opp.active.hp` (lethal on the
  Active) even under 6 cards; also fire if the opponent threatens to KO Alakazam next turn
  (don't waste a loaded nuke). Otherwise build the hand.
- Note: Powerful Hand ignores HP-based walls (it PLACES counters) — good vs high-HP ex.

## doublade — hand-of-swords scaling (STRONG combo)
- **Payoff**: Doublade (1066) *Weaponized Swords* (cost 2 {M}): reveal any # of Honedge
  (1065) / Doublade (1066) / Aegislash (1067) from hand → **60 × revealed**. Alt payoff:
  Aegislash (1067) *Metal Slash* (cost 4, 230).
- **Engine**: Genesect ex (547) *Metallic Signal* (search 2 evolution {M} to hand),
  Hilda, Rare Candy — load the hand with steel-line copies.
- **`_expected_dmg`** = `60 × (# of {1065,1066,1067} in my hand)` if Active is Doublade.
- **`combo_online`** = Active is Doublade AND `steel_in_hand >= 3` (≥180). Below, dig with
  Genesect/Hilda to stock the hand.
- **Execution / opp info**: fire if `60×count >= opp.active.hp` (lethal) or online or
  under KO threat. Prefer to keep 1 line-card to re-evolve if the board is fragile.

## hydrapple — grass-energy scaling (RAMP-combo)
- **Payoff**: Hydrapple ex (150) *Syrup Storm* (cost 2 {G}): **30 + 30 × {G} energy on all
  my Pokémon**. Meganium (710) *Wild Growth* makes each basic {G} count as {G}{G}.
- **Engine**: Teal Mask Ogerpon ex (96) *Teal Dance* + Hydrapple *Ripening Charge* (attach
  a basic {G} each turn), Forest of Vitality, 14 {G}. Celebi/Traverse Time to fetch grass.
- **`_effective_G`** = (basic {G} energy attached across my board) × (2 if a Meganium(710)
  is in play else 1). **`_expected_dmg`** = `30 + 30×_effective_G` if Active is Hydrapple ex.
- **`combo_online`** = Active is Hydrapple ex AND `_effective_G >= 4` (≥150). Keep charging
  grass otherwise.
- **Execution / opp info**: fire on lethal / online / KO threat; grass accumulates each
  turn so patience directly grows the hit.

## rockets_mewtwo — tribal Power-Saver gate (ENGINE-ENFORCED combo)
- **Payoff**: Team Rocket's Mewtwo ex (431) *Erasure Ball* (cost 3, 160; may discard up to
  2 {Energy} from BENCH for more). **Power Saver ability: Mewtwo can't attack unless you
  have ≥4 Team Rocket's Pokémon in play** — the sim will not OFFER the attack before that.
- **Engine**: TR Tarountula (400)/Spidops (401) develop the tribe; Spidops *Charging Up*
  (attach a basic Energy from discard to itself) loads bench energy for Erasure Ball's
  discard-boost; TR Factory / Transceiver consistency; Articuno (414) *Repelling Veil* wall.
- **`_tr_count`** = # of my in-play Pokémon whose name contains "Team Rocket's".
- **`combo_online`** = `_tr_count >= 4` AND Active is Mewtwo ex (431). The gate is mostly
  auto (engine), so L2 mainly **prioritises building 4 TR Pokémon fast** and charging bench
  energy; withholding is redundant (no illegal attack is offered), so decide_attack ≈ L0
  but never chips with Mewtwo when a bench-energy-loaded Erasure Ball would be much bigger.
- **Execution / opp info**: fire Erasure Ball when it KOs the Active (discard bench energy
  up to the amount needed), else standard.

## mamoswine — Stage-2 swarm scaling (NOT a hard gate → light L2)
- **Payoff**: Mamoswine ex (283) *Rumbling March* (cost 2, **180 + 40 × benched Stage-2**).
  180 base is a fine attack any time, so **no strict withholding**.
- **Engine**: *Mammoth Hauler* (search a Pokémon to hand each turn) to stock Stage-2 lines
  (Alakazam 743 / Klinklang 167 / Blaziken 326) on the bench; Blaziken *Seething Spirit*
  energy-accel; Rare Candy.
- **`_expected_dmg`** = `180 + 40 × (# benched Stage-2)` if Active is Mamoswine ex.
- **`combo_online`** = Active is Mamoswine ex AND energy ≥ 2 (can fire). L2 = **prioritise
  Mammoth Hauler + benching Stage-2** for the bonus, but attack whenever loaded (L0-like).

## metagross — energy-engine toolbox (NOT a hard gate → light L2)
- **Payoff**: Steven's Metagross ex (641) *Metal Stomp* (cost 3, 200), fed by *X-Boot*
  (search a basic {P}/{M} energy each turn). 200 is fine any time → no withholding.
- **Engine**: Rare Candy → Metagross ex, then X-Boot every turn to power the toolbox
  (Genesect ex 547, Latias ex 184, Empoleon ex 835, Mega Mawile ex 695). Slow setup.
- **`combo_online`** = a Metagross ex in play (engine online) AND an attacker loaded.
  L2 = **prioritise Rare-Candy-ing Metagross + X-Boot**; attack with the loaded ex (L0-like).

## slowking — multi-attacker toolbox (NOT a combo → treat as toolbox)
- Real cards are a spread/beatdown toolbox: Mega Kangaskhan ex (756) *Rapid-Fire Combo*
  (200+), Kyurem (144) *Trifrost* (110 to 3 = spread), Slowking (163) *Super Psy Bolt* 120,
  Metagross (276). *Seek Inspiration* mills top for setup. There is **no assemble-gate**.
- **L2 = inherit toolbox behaviour** (matchup attacker selection), NOT a combo gate.
  (Recommend re-tagging `slowking` archetype → toolbox after validation.)

---

## Dispatch
`tuning.json` gets `"l2": "<name>"` on each; `engine_v2._PERDECK` maps name→class;
`make_policy` uses it before the archetype class. Non-breaking: decks without `l2` keep
their archetype policy.

---
## Protection (A) — result (2026-07-11)
Implemented per the reframe (protect the combo from disruption): (1) generic **safe
promotion** — after a KO, don't promote an unready payoff/engine into a KO-able Active
if a READY non-payoff attacker is available (keep the payoff safe on bench); (2) rockets
**Team Rocket's Articuno (414)** Repelling-Veil wall benched first to shield the assembly.
**Measured NEUTRAL** (field −0.9%, z −0.43; per-deck swings are noise — divergence only
0.1–0.5%, because promotion-after-KO is rare and L0's best-attacker pick already ≈ the
ready-safe pick). All three combo directions now tested: **gate/withhold −8.4% (bad)**,
fast-assembly +0.2% (neutral), protection −0.9% (neutral). **Combo L2 ≈ L0** — L0 already
assembles, protects and attacks these decks about as well as bespoke logic can here.
