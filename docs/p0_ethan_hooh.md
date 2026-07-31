# P0 Static Analysis — `ethan_hooh`

Blind zero-shot P0. All mechanical claims grounded in the 60-card DB and, where marked
`[VERIFIED]`, in `cg` simulator probes (harness: `tools/_probe_hooh.py`). `[reasoned]` = from
card text + rules, not directly forced in a probe.

## Deck list (60)

Pokémon (13):
- **Ethan's Ho-Oh ex** ×4 — 230 HP, Fire, weak Water(×2). Ability **Golden Flame**: once/turn, attach up to 2 Basic {R} from hand to 1 **Benched Ethan's** Pokémon. Attack **Shining Feathers** [RRRR] 160, *heal 50 from each of your Pokémon*.
- **Ethan's Slugma** ×3 — 80 HP, Fire, weak Water. **Steady Firebreathing** [R] 20.
- **Ethan's Magcargo** ×3 — 130 HP, Fire (Stage 1 ← Slugma). Ability **Melt Away**: no Retreat Cost while it has no Energy. Attack **Lava Burst** [RRR] *display 0*: discard up to 5 {R} from itself, 70 dmg per discarded.
- **Ethan's Pinsir** ×1 — 120 HP, Grass, weak Fire. **Vise Grip** [G] 20; **Rallying Horn** [CCC] 70 (+100 if any of your Ethan's Pokémon was KO'd by an attack last turn).
- **Fezandipiti ex** ×1 — 210 HP, Dark. Ability **Flip the Script**: once/turn, if any of your Pokémon were KO'd last turn, draw 3. **Cruel Arrow** [CCC] *display 0*: 100 to any 1 opponent Pokémon.
- **Latias ex** ×1 — 210 HP, Psychic. Ability **Skyliner**: your Basic Pokémon have no Retreat Cost. **Eon Blade** [PPC] 200, can't attack next turn.

Energy (13): **Basic {R} Energy** ×13 (only energy type in deck).

Trainers (34): Lillie's Determination ×4 (S), Ultra Ball ×4 (I), Buddy-Buddy Poffin ×4 (I),
Ethan's Adventure ×3 (S), Crispin ×3 (S), Boss's Orders ×3 (S), Poké Pad ×3 (I),
Night Stretcher ×3 (I), Pokégear 3.0 ×3 (I), Team Rocket's Watchtower ×2 (Stadium),
Cyrano ×1 (S), Prime Catcher ×1 (I, ACE SPEC).

---

## §1 Win-condition spec

The deck is a **Fire energy-acceleration engine**: only {R} energy, one accelerator (Golden Flame,
2/turn to the bench), and two payoff modes.

### Real-damage formulas (display ≠ effective)
| Attacker | Attack | Display | Real formula | verified |
|---|---|---|---|---|
| Magcargo (356) | Lava Burst | **0** | `dmg = 70 × min(5, R_discarded_from_self)` → 0/70/140/210/280/**350**, ×2 vs Fire-weak | **[VERIFIED]** max 350 seen; 420/560/700 seen = ×2 weakness |
| Fezandipiti (140) | Cruel Arrow | **0** | `dmg = 100` to any 1 target (no bench weakness) | **[VERIFIED]** offered/used |
| Pinsir (338) | Rallying Horn | 70 | `dmg = 70 + (100 if your Ethan's Pkmn KO'd last turn) = 70 or 170` | offered/used at 70; +100 rider [reasoned] |
| Ho-Oh (357) | Shining Feathers | 160 | `dmg = 160`; rider `heal 50 × (#your Pokémon)` | **[VERIFIED]** heal to 50 seen |

`_expected_dmg` implication: **Lava Burst and Cruel Arrow both display 0** — the shared-perception
damage function must be overridden for cids 356 and 140, or L0 will value them at 0 (gust/promotion/
energy-routing/lethal all read display).

### Primary line — *Doctrine A: Ho-Oh sustain + Magcargo finisher* (recommended)
1. **T1–2**: Ho-Oh ex active; bench Slugma → evolve Magcargo. Search with Ultra Ball / Poké Pad / Ethan's Adventure (NOT Poffin — dead, see §2). Manual-attach R to Ho-Oh.
2. **T2+**: Ho-Oh **Shining Feathers 160 + heal 50 all** every turn (standing [RRRR] cost, *does not discard energy* → sustainable). Simultaneously **Golden Flame 2 R/turn onto a benched Magcargo**.
3. **Finisher**: once a benched Magcargo holds **5 R** and the opponent shows a KO target (≥210 HP ex, or lethal), retreat Ho-Oh (cost 2) / Prime Catcher → promote Magcargo → **Lava Burst 350** (700 vs Fire-weak).
4. **Reset**: spent Magcargo has 0 energy → **Melt Away = free retreat** → pivot back to Ho-Oh; rebank the next Magcargo.

Resource the formula tells you to grow: **R energy on one designated benched Magcargo, capped at 5.**

### Alt line — *Doctrine B: Magcargo primary beatdown*
Rush a loaded Magcargo active and Lava Burst repeatedly; Ho-Oh is a bench battery (Golden Flame only).
Weaker: Magcargo is 130 HP glass, must fully reload (5 R) after every shot → low uptime, heavier digging → worse deckout. See `doctrines` A/B discriminator.

### Plan B (main line disrupted)
- **Ho-Oh Shining Feathers** alone is a legitimate 160/turn + full-board heal grind (2-prize liability but 230 HP + heal tanks single hits).
- **Fezandipiti Cruel Arrow** [CCC=3R] snipes 100 to any target (finish a benched threat / spread).
- **Pinsir Rallying Horn** [CCC] 170 as a one-prize revenge attacker the turn after a KO.

---

## §2 Rule-interaction inventory

- **Energy carries up on evolution**: pre-attaching R to a benched **Slugma** is fine — it lifts to Magcargo on evolve. Golden Flame can target Slugma or Magcargo (both "Ethan's"). Front-load Slugma if Magcargo not yet in hand.
- **Damage vs damage-counters**: Lava Burst / Shining Feathers / Cruel Arrow are all **damage** (weakness applies — Lava ×2 vs Fire-weak verified 700). No damage-counter-placement cards in deck.
- **Cost TYPE vs supply** (deck supplies ONLY {R}=Fire; no special energy; Crispin can only pull {R} from this deck):
  | Attack | Cost | Payable by {R}? |
  |---|---|---|
  | Shining Feathers | RRRR | ✅ (Fire) |
  | Steady Firebreathing | R | ✅ |
  | Lava Burst | RRR (+discard R) | ✅ |
  | Rallying Horn | CCC | ✅ (colorless) — **[VERIFIED]** offered/used |
  | Cruel Arrow | CCC | ✅ (colorless) — **[VERIFIED]** offered/used |
  | **Vise Grip** | **G** | ❌ **TYPE-DEAD** — no Grass. **[VERIFIED]** never offered (40 games) |
  | **Eon Blade** | **PPC** | ❌ **TYPE-DEAD** — no Psychic. **[VERIFIED]** never offered (40 games) |
  → **Latias is a Skyliner-only utility Basic; Pinsir attacks only via Rallying Horn.** Any R attached to Latias for "Eon Blade (display 200)" is wasted forever.
- **1-per-turn conflicts**: one Supporter/turn — Boss's Orders (gust) **competes** with the draw/search supporters (Lillie's, Adventure, Crispin, Cyrano). One manual attach/turn (separate from Golden Flame). One Golden Flame/turn.
- **Deckout horizon**: heavy dig — Lillie's Determination (draw 6/8), Ultra Ball ×4 (−1 deck +discard 2 hand), Ethan's Adventure (−3), Cyrano (−3), Poké Pad (−1), Pokégear (−1), Crispin (−2), Fezandipiti draw 3. Only **Night Stretcher** recycles (discard→hand, does not refill deck). **[VERIFIED]** naive over-draw pilot deckouts **29/60 (~48%)**, deck→0 by ~turn 15/player. Deckout is the #1 self-inflicted loss.
- **ACE SPEC / Stadium**: **Prime Catcher** (1 copy) = gust + self-switch (enables promoting a loaded Magcargo without paying retreat). **Team Rocket's Watchtower** ×2 = "{C} (Colorless) Pokémon in play have no Abilities." **None of our Pokémon are Colorless** (Fire/Grass/Dark/Psychic) → it does **not** disable Golden Flame/Flip the Script/Skyliner; one-sided vs opponent's Colorless-ability Pokémon. **[VERIFIED]** our abilities functioned in games where Watchtower was played.

## §2b Self-harm sweep (all 60) — with fire-conditions

| Card | Self-harm | Fire only when (state) | Severity |
|---|---|---|---|
| **Draw/search engine** (Lillie's ×4, Ultra Ball ×4, Adventure ×3, Cyrano, Poké Pad ×3, Pokégear ×3, Crispin ×3, Fezandipiti draw) | **Self-mill → deckout** (~48% verified) | Search/draw only while `deckCount` safe **and** a needed piece is missing; STOP non-essential dig when `deckCount ≤ D` | **catastrophic** (H2) |
| **Ethan's Magcargo — Lava Burst** | Discards up to 5 {R} from itself → 0 energy, cannot re-attack until reloaded (self tempo-lock; explains low post-KO attack rate) | Only when it **secures a KO** (`target_eff_HP ≤ 70×R_on_Magcargo`) or is lethal | major (H1/H3) |
| **Ultra Ball ×4** | Discard 2 from hand — can pitch {R} or a key Pokémon (Magcargo/Ho-Oh) | Only when hand has ≥2 spare non-{R}, non-combo cards **and** target is needed | major (H6) |
| **Lillie's Determination ×4** | Shuffles a curated hand into deck + draws 6/8 (mills) | Only when hand is thin / lacks the line; **never** while holding assembled combo or `deckCount ≤ D` | major (H2/H6) |
| Latias — Eon Blade | "can't attack next turn" self-lock | moot (type-dead, never usable) | none |
| Fezandipiti — Flip the Script | draw 3 → deck resource | Fine unless `deckCount ≤ ~4` | minor |
| Team Rocket's Watchtower | disables Colorless abilities | We have no Colorless Pokémon → **no self-harm** [VERIFIED] | none |

No card returns/removes our own Pokémon from play, no self-damage, no self damage-counter placement.
The catastrophic bucket is **unconditional dig → deckout** (§7, H2).

## §3 Prize arithmetic

| Attacker | HP | Prizes given | Note |
|---|---|---|---|
| Ho-Oh ex | 230 | **2** | +50 heal/turn tanks one ~180 hit; weak Water = ×2 danger |
| Magcargo | 130 | 1 | glass; 350 payload = great 1-for-2 trade if it lands then survives/retreats |
| Slugma | 80 | 1 | early poke only |
| Pinsir | 120 | 1 | Rallying Horn 170 revenge |
| Fezandipiti ex | 210 | **2** | utility (draw); Cruel Arrow snipe |
| Latias ex | 210 | **2** | utility (free retreat) only |

- Environment ~200–330 damage: **Magcargo (130) and Ho-Oh (230−heal) are "taken every turn" targets**; three of six Pokémon are 2-prize exes → careless trades hand the opponent the race.
- Chain requirement: to KO every turn you need a **reloaded** attacker each turn. Magcargo needs ~2–3 turns to rebank 5 R after firing → it is a **periodic nuke, not a per-turn engine**. Ho-Oh Shining Feathers 160 is the per-turn attacker. Keep a 2nd Magcargo banking to preserve post-KO tempo.

## §4 Phase plan (measurable)

| Phase | Transition (state vars) | "On plan" | Recovery on deviation |
|---|---|---|---|
| **Early** | until `356 in play` AND `bench_R[356]≥1` (or Slugma banking) | By ~T3–4: Ho-Oh active, Slugma/Magcargo benched, attacking with Shining Feathers/Steady; `first_attack_turn ≤ 4` | If no Fire attacker: Ultra Ball/Poké Pad/Adventure for Slugma/Ho-Oh (not Poffin); manual-attach to Ho-Oh |
| **Mid** | `bench_R[356]` rising toward 5, `deckCount > 15` | Ho-Oh Shining 160+heal every turn; `energy_attach_share[356]` climbing; `nonattacking_turn_rate` low; not yet firing Lava | If Ho-Oh KO'd: promote Magcargo/Pinsir, use Fezandipiti draw; throttle dig if `deckCount ≤ D` |
| **Late** | `opp_prizes ≤ 3` OR KO target ≥210 HP present OR lethal | Promote loaded Magcargo → Lava Burst 350 KO → Melt-Away retreat; `post_ko_attack_rate` high; `deckCount > 6` | If deck thin: STOP search, attack only; Night Stretcher to rebuy energy/Pokémon |

Cards **used** each phase: early = balls/Poké Pad/Adventure/Crispin, Ho-Oh; mid = Golden Flame, Lillie's (gated), Shining Feathers; late = Boss/Prime Catcher, Lava Burst, Night Stretcher.
Cards **not** used: Buddy-Buddy Poffin (dead everywhere); Latias/Pinsir attacks (type-dead / niche); Lillie's & Ultra Ball once `deckCount ≤ D`.

## §5 Scorecard + tensions

| Axis (deck-specific) | Score | Metric |
|---|---|---|
| Speed (Fire attacker online) | 3 | `first_attack_turn` (Ho-Oh Shining online T2–3; Lava online T4–6) |
| Power curve | 4 | Lava 350 / Shining 160; but Lava periodic |
| Consistency / brick | 2 | Poffin ×4 dead + type-dead Latias/Pinsir attacks + reliance on Golden Flame routing |
| Sustain / rebuild | 4 | Shining heal-all 50; Night Stretcher recycles; Melt-Away pivot |
| Answers (walls/hate) | 3 | Cruel Arrow snipe, Boss/Prime Catcher gust; Watchtower vs Colorless abilities |
| Resource economy | **1** | `loss_share[deckout]` ~0.48; energy discarded 5/shot |
| Disruption resistance | 2 | single accelerator (Golden Flame on Ho-Oh) is a gust/KO chokepoint; no energy in play to survive hammers well |
| Prize-race structure | 2 | three 2-prize exes; must trade Magcargo (1) for exes (2) |

**Tensions (each pairs an engine with its own failure):**
1. **Dig ↔ Deckout** — the draw/search that assembles the combo also mills to 0. Balance: dig only while `deckCount > D (D∈{8..15})` **and** a required piece is missing; *fixing the deckout metric requires simultaneously watching combo-assembly (whiff) so throttling doesn't brick.*
2. **Energy banking ↔ Attacking now** — R sent to bench-Magcargo (Golden Flame) is R not on Ho-Oh's Shining Feathers, and delays damage. Balance: bank to exactly **5 on one** Magcargo, then stop and attack. *Monitor `energy_attach_share[356]` against `nonattacking_turn_rate`.*
3. **Magcargo nuke ↔ prize/resource economy** — Lava Burst spends 5 R + a promotion for one KO; a wasted shot is catastrophic tempo. Balance: fire only when `target_eff_HP ≤ 70×R_on_Magcargo`.

## §6 Card-by-card usage declarations (all 60)

Assumed steady state: hand ~5–8 (heavy draw), bench 3–4, 1–2 Fire attackers, R energy plentiful in hand early.
`fire_if` numeric gates are **provisional params** for P4 calibration; each notes what over-tightening starves.

- **Ethan's Ho-Oh ex ×4** | primary attacker + accelerator | active from T1; Golden Flame every turn `if R_in_hand≥1 and benched Ethan's target exists`; Shining Feathers `if RRRR attached` | ~every turn | dead only if no R ever / all copies prized.
- **Basic {R} Energy ×13** | fuel Shining(4)/Lava(5)/colorless costs | attach to Ho-Oh (attack) or via Golden Flame to bench-Magcargo | ~2–3 placed/turn | never dead (universal fuel).
- **Ethan's Slugma ×3** | Magcargo pre-evo + T1 poke | bench early; evolve ASAP; Steady Firebreathing `if only Fire attacker` | high early | dead late if both Magcargo copies already in play.
- **Ethan's Magcargo ×3** | payoff nuke | evolve on bench; bank to 5 R; promote+Lava `if target_eff_HP ≤ 70×R` | 0.5–1 Lava/game | dead if never reaches ≥3 R (routing failure H1).
- **Ethan's Pinsir ×1** | revenge attacker | promote after a KO; Rallying Horn `if Ethan's KO'd last turn` (170) | ~0.15 | dead unless CCC available; **Vise Grip never** (type-dead).
- **Fezandipiti ex ×1** | draw engine + snipe | bench; Flip the Script `if own Pokémon KO'd last turn and deckCount>4`; Cruel Arrow `if CCC and a 100-KO/snipe target` | ~0.4 | dead if never KO'd (no draw trigger).
- **Latias ex ×1** | free-retreat utility ONLY | bench; Skyliner passive | 0 attacks | **never** attack (Eon Blade type-dead); **never** attach R to it.
- **Lillie's Determination ×4** | hand reset / refuel | `fire_if hand_size ≤ H (H∈{3,4,5}) and deckCount > D (D∈{8,10,12}) and not holding assembled line` | ~1–2 | over-tighten → starves refuel; under-tighten → deckout.
- **Ultra Ball ×4** | any-Pokémon tutor | `fire_if need_pokemon and hand has ≥2 spare non-{R} non-combo` | ~1.5 | over-tighten → can't find Magcargo/Ho-Oh; risk = pitching last R.
- **Buddy-Buddy Poffin ×4** | *(intended basic search)* | **DEAD — 0 legal targets [VERIFIED]** (no ≤70 HP basic; Slugma 80) | plays but whiffs | always dead; ideally don't play (pure tempo loss, no deck thin).
- **Ethan's Adventure ×3** | tutor Ethan's Pkmn + R (up to 3) | `fire_if need (Pokémon or ≥1 R) and deckCount > D` | ~1 | over-tighten → slower setup; mills 3.
- **Crispin ×3** | attach + fetch energy | `fire_if want extra attach` (only pulls {R} here — different-type clause inert) | ~1 | partial waste (2nd energy can't differ); still a free attach.
- **Boss's Orders ×3** | gust KO target | `fire_if can KO/lethal a benched threat this turn (real dmg)` | ~1 | over-tighten → miss lethal; competes with draw supporter.
- **Poké Pad ×3** | tutor non-rule-box Pkmn (Slugma/Magcargo/Pinsir) | `fire_if need the Magcargo line and deckCount > D` | ~1 | over-tighten → slower line; mills 1.
- **Night Stretcher ×3** | recycle Pokémon/{R} from discard | `fire_if key piece (Magcargo/Ho-Oh/R) in discard and needed` — **anti-deckout, does not mill** | ~1 | rarely dead; keep for late rebuild.
- **Pokégear 3.0 ×3** | dig a Supporter | `fire_if no Supporter in hand` | ~1 | low harm; mills 1.
- **Team Rocket's Watchtower ×2** | shut off opp Colorless abilities | `fire_if opp relies on Colorless-ability Pkmn or to overwrite opp stadium` | ~0.5 | no self-harm; often inert vs non-Colorless meta.
- **Cyrano ×1** | tutor up to 3 ex (Ho-Oh/Fezandipiti/Latias) | `fire_if need Ho-Oh and hand lacks it` early | ~0.3 | mills 3; over-tighten slows Ho-Oh access.
- **Prime Catcher ×1 (ACE)** | gust + self-switch (promote loaded Magcargo) | `fire_if lethal gust OR need to bring Magcargo active without retreat` | ~0.3 | single copy — save for the Lava KO turn.

## §7 Loss-mode hypotheses & L2 candidates

Predicted loss distribution: **deckout ~0.45–0.5** (verified naive pilot 0.48), prize-race loss (glass Magcargo / 2-prize exes traded badly), and **"payoff never online"** (Golden Flame feeds Ho-Oh not Magcargo; Lava/Cruel Arrow display 0). Applied general patterns:
- Scale attack undervalued (Lava, Cruel Arrow display 0) → **promote real-damage formula to shared perception** (gust/promotion/energy-routing/lethal read it).
- Energy to wrong Pokémon → **deck-specific feeding plan**: designate one benched Magcargo, cap 5, never feed Latias.
- Deckout → **throttle dig at low deck** (stop-digging side is essential, not just recycle).
- Main attacker KO'd → **2nd Magcargo pre-bank (chain, cap 5)**.
- Type-dead loaded as attacker → **type-aware loading** (Latias/Pinsir-V);
- Self-harm unconditional → **state gate** on Ultra Ball discard / Lillie's / Lava.

### L2 needed?  **Yes — build L2.**
L0's display-0 blindness (Lava & Cruel Arrow = 0), its "attach to highest-display attacker" (feeds Ho-Oh/tries type-dead Latias, starves Magcargo), and its unconditional dig (~48% deckout verified) will systematically break this deck's win condition. The three high-value L2 levers are concrete and testable: (1) shared-perception real-damage for cids 356/140, (2) Golden-Flame/attach routing to a designated benched Magcargo (cap 5), (3) deckout throttle. See `disposition_ledger` — H1 and H2 are catastrophic and must be implemented for P4.
