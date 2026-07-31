# P0 静的解析: `hop_zacian`

Zero-shot / blind static analysis. All mechanical claims are grounded in the 60
actual cards (card DB) and, where marked **[VERIFIED]**, in cg-engine self-play
probes (`/tmp/probe*.py`, `/tmp/harvest.py`). Damage numbers below were read off
the engine's `HP_CHANGE` logs.

---

## 0. Deck contents (60 cards, 24 unique)

Type breakdown: **13 Pokémon, 12 Supporter, 17 Item, 4 Tool (Choice Band), 4 Stadium
(Postwick), 8 Basic {M}, 2 Mist Energy.**

### Pokémon (13)
| id | ×n | name | HP | type | stage | weak/res/retreat | key line |
|---|---|---|---|---|---|---|---|
| 299 | 3 | Hop's Zacian ex | 230 | M | basic ex | F / G / 2 | **Brave Slash** `{M}{M}{M}{C}`=240 (self-lock next turn); **Insta-Strike** `{C}`=30 +30 to a benched |
| 878 | 2 | Hop's Phantump | 70 | P | basic | D / F / 2 | Splashing Dodge `{C}`=10, coin: prevent all dmg/effects next turn |
| 879 | 1 | Hop's Trevenant | 140 | P | stage1←878 | D / F / 2 | Horrifying Revenge `{C}`=30 (+100 if a Hop's KO'd last opp turn); **Corner `{P}{C}{C}`=90 → type-dead** |
| 304 | 2 | Hop's Snorlax | 150 | C | basic | Fight / – / 4 | **Extra Helpings** (ability): your Hop's attacks +30 to opp Active, no-stack; Dynamic Press `{C}{C}{C}`=140 +80 self |
| 309 | 1 | Hop's Wooloo | 70 | C | basic | Fight / – / 1 | Smash Kick `{C}{C}{C}`=50 |
| 310 | 1 | Hop's Dubwool | 120 | C | stage1←309 | Fight / – / 2 | Headbutt `{C}{C}{C}`=80; **Defiant Horn**: on-evolve, gust an opp benched to Active |
| 311 | 1 | Hop's Cramorant | 110 | C | basic | L / F / 1 | Fickle Spitting `{C}`=120 **only if opp has exactly 3 or 4 prizes, else nothing** |
| 140 | 1 | Fezandipiti ex | 210 | D | basic ex | Fight / – / 1 | **Cruel Arrow `{C}{C}{C}`=100 to ANY 1 opp Pokémon** (display 0); Flip the Script: draw 3 if a Pokémon of yours was KO'd |
| 626 | 1 | Patrat | 60 | C | basic | Fight / – / 1 | Procurement `{C}`: search an Item; Gnaw `{C}`=10 |

### Trainers
- **Supporters (12):** Lillie's Determination ×4 (shuffle hand → draw 6, or 8 if you hold exactly 6 prizes), Boss's Orders ×3 (gust), Judge ×3 (both shuffle hand → draw 4), Team Rocket's Petrel ×2 (search Trainer).
- **Items (17):** Pokégear 3.0 ×4 (top-7 → a Supporter), Hop's Bag ×3 (search ≤2 Basic Hop's → Bench), Switch ×3, Night Stretcher ×2 (recover Pokémon/Basic-Energy from discard), Poké Pad ×2 (search a non-rule-box Pokémon), Ultra Ball ×2 (discard 2 → any Pokémon), Secret Box ×1 (discard 3 → Item+Tool+Supporter+Stadium).
- **Tool (4):** Hop's Choice Band — attacks by the attached **Hop's** Pokémon cost **{C} less** and do **+30** to opp Active.
- **Stadium (4):** Postwick — **Hop's** Pokémon (both players') do **+30** to opp Active.

### Energy (10)
- **Basic {M} ×8** — the only energy that pays a {M} requirement. No energy acceleration exists in the deck (hard cap of 1 manual attach/turn).
- **Mist Energy ×2** — provides **{C}** only; grants its holder immunity to opponents' attack *effects* (not damage).

---

## §1 Win-condition spec

**Primary win condition — Hop's Zacian ex beatdown under a stacked "Hop's" damage lattice.**

Setup (procedure): by ~T2–3 have **Postwick** in play (stadium, +30), a **Snorlax** on the
Bench (Extra Helpings, +30), a **Choice Band** on the active Zacian (+30 and −{C}), and **3 Basic
{M}** on that Zacian. Then every "fresh" turn:
- **Brave Slash for 330** → OHKO essentially anything in the format (incl. 2-/3-prize ex/mega).
- Take 6 prizes across ~3–6 KOs, gusting the correct target with **Boss's Orders**.

**Real-damage formulas** (`buff = 30 × [Postwick? + ExtraHelpings-in-play? + ChoiceBand-on-attacker?]`,
max +90; **buffs apply to opponent's ACTIVE only**; Extra Helpings does not stack with itself; weakness
applies **after** buff):
| attack | real formula | display | verified |
|---|---|---|---|
| Brave Slash (413) | `240 + buff` (×2 if def weak M) → **330** at full buff | 240 | **[VERIFIED 330]** (probe2: `HP -330` on a 230-HP Zacian, 3 energy) |
| Insta-Strike (412) active | `30 + buff` → **120** at full buff | 30 | **[VERIFIED 60/90/120]** (harvest) |
| Insta-Strike (412) bench | **30 flat** (no buff, no weakness) | – | **[VERIFIED -30]** (harvest `(-120,-30)`,`(-90,-30)`) |
| Dynamic Press (422) | `140 + buff` (Snorlax IS Hop's) → 230; **80 self** | 140 | **[VERIFIED 200 + self-80]** (harvest `(-200,-80)`) |
| Cruel Arrow (183) | **100 flat, ANY target, ignores W/R** — Fezandipiti is **NOT** a Hop's Pokémon → **no buff** | 0 | **[VERIFIED -100]** (harvest) |
| Fickle Spitting (433) | `120 + buff` if opp prizes ∈ {3,4}; **else 0** | 120 | **[VERIFIED whiff]** (harvest `()` when unmet) |
| Horrifying Revenge (1267) | `30 + (100 if a Hop's was KO'd by attack last opp turn) + buff` → up to 220 | 30 | text-grounded, **unverified** (conditional) |
| Corner (1268) | `90 + buff` but requires `{P}` — **no Psychic supply → never usable** | 90 | type-death, see §2 |

The single variable that drives the whole deck is **`buff` (0/30/60/90) and the count of Basic {M}
on the attacking Zacian (≥3)**. These are the resources to "grow": Postwick down, ≥1 Snorlax in play,
Band on the attacker, 3 Metal on the attacker.

**Sub / plan-B lines** (all get the +buff except Fezandipiti/Patrat):
- **Insta-Strike** (1 energy, 120 active + 30 bench snipe) is the T1–T3 and Brave-Slash-off-turn attack.
- **Cruel Arrow** (Fezandipiti, flat 100 to *any* Pokémon, ignores weakness/resistance on bench) is a
  snipe/finisher that needs **3 colorless-payable** energy and is **display-0** (see H3).
- **Dubwool** (Headbutt 80 + Defiant Horn on-evolve gust) and **Cramorant** (Fickle Spitting 120 in the
  3–4-prize window) are 1-prize supplementary attackers.
- **Fickle Spitting** becomes a 150 (120+30) one-energy nuke exactly in the prize-3/4 window.

**Doctrines** (see JSON `doctrines`):
- **A — single-Zacian + Insta rotation.** One Zacian is the Brave-Slash core; on its self-lock turn it
  Insta-Strikes (120+30) rather than idling. Cheap on energy/bands. Discriminator: `energy_attach_share`
  concentrated on one Zacian serial; high `play_rate[412]` on post-Brave turns.
- **B — two-Zacian tag-team.** Load two Zacians (each with a Band, 3 Metal) and Brave Slash every turn by
  alternating. Needs 6 Metal + 2 Bands → slow with 1 attach/turn. Discriminator: `energy_attach_share`
  split across ≥2 Zacian serials, later `first_attack_turn` but continuous 330 OHKOs.
  A/B: with no acceleration, A is the realistic early plan; B is the grind plan. Measure which correlates
  with wins.

---

## §2 Rule-interaction inventory

- **Energy carries up on evolution.** Relevant lines: Phantump→Trevenant, Wooloo→Dubwool. Pre-attaching
  {C}/{M} to Phantump/Wooloo is legal, but both evolved forms only ever need colorless-payable energy, so
  pre-loading is low value; do **not** pre-load Psychic/anything special. No stage-2 line exists.
- **Damage vs damage-counter placement.** Every attack here deals **damage** (weakness/resistance apply);
  none place counters. So opponents' resistance/effect-immunity matters. **Mist Energy** on a Zacian gives
  that Zacian immunity to opponents' attack *effects* (e.g. gust-lock/ability-shutdown effects), not damage.
- **Attack-cost TYPE vs supply.** Supply = 8 Basic {M} (pays {M} and {C}) + 2 Mist ({C} only). Per attacker:
  | attacker | cost | payable? |
  |---|---|---|
  | Zacian Brave Slash | `{M}{M}{M}{C}` (→ `{M}{M}{M}` with Band) | ✅ needs **3 real Metal**; Mist can be the {C} only in the no-Band case, **Mist never counts toward the 3 {M}** |
  | Zacian Insta-Strike | `{C}` | ✅ any 1 energy (free with Band, see below) |
  | Snorlax Dynamic Press | `{C}{C}{C}` | ✅ 3 Metal |
  | Trevenant Horrifying Revenge | `{C}` | ✅ |
  | **Trevenant Corner** | `{P}{C}{C}` | ❌ **TYPE-DEAD — zero Psychic in deck, never usable** |
  | Dubwool / Wooloo | `{C}{C}{C}` | ✅ |
  | Cramorant Fickle Spitting | `{C}` | ✅ (effect-gated) |
  | Fezandipiti Cruel Arrow | `{C}{C}{C}` | ✅ Metal pays colorless; **not a Hop's Pokémon → no buff** |
  | Patrat | `{C}` | ✅ |
  **Type-death: only Corner.** The subtle trap is that **"loaded" must be counted in {M}, not in energy
  count** — a Zacian carrying 2 Metal + 1 Mist has 3 energy but **cannot Brave Slash** (needs 3 {M}). See H2.
- **Choice Band cost reduction — [VERIFIED].** Probe5 shows the ATTACK option for Brave Slash appears at
  **3 energy** (not 4) once a Band is attached (`en=3, band=True → attacks [412,413]`; `en=2,band → [412]`
  only). So Brave Slash truly costs 3 {M} with a Band, and Insta-Strike's `{C}` reduces to **0** (a free
  attack) — inferred from the same rule; the engine's legality gate honours it. This is a cost the naive
  loader may not model (it may keep attaching a wasteful 4th Metal — H7).
- **1-per-turn caps.** One Supporter, one manual energy, one Stadium, one retreat per turn. The deck is
  **Supporter-heavy (12)** and energy-starved (10 total, 1 attach/turn, no accel) → the binding constraint
  is energy on Zacian, and you routinely draw more Supporters than you can play (tension T2).
- **Self-draw / deckout horizon.** Draw/dig is very deep: Lillie's ×4 (net-neutral, shuffles hand back),
  Judge ×3, Pokégear ×4, Hop's Bag ×3, Poké Pad ×2, Petrel ×2, Ultra Ball ×2 (−2), Secret Box ×1 (−3),
  Night Stretcher ×2. Lillie's/Judge reshuffle the hand so they barely deplete; the true thinning is
  Ultra/Secret discards + searches. Rough horizon: with disciplined play the deck rarely decks before
  ~turn 18–20; games usually end far sooner by OHKO. Deckout is a **tail** risk, not a primary loss mode
  (H10).
- **ACE SPEC / special energy / stadium.** No ACE SPEC in the list. Special energy = Mist ×2 (colorless +
  effect-immunity). Stadium = Postwick ×4 — a buff you rely on; an opponent's stadium removes it and drops
  your ceiling 330→300 (H8, tension T3). Postwick buffs the opponent's Hop's Pokémon too (mirror only).

---

## §2b Self-harm sweep (all 60 cards) ★

Scanned all 24 uniques for effects that can hurt the pilot. **No card removes your own Pokémon to
deck/hand and none mills you**, so there is no literal "use it on your last Pokémon = instant loss"
ability. The real self-harm surface is **hand-shuffle Supporters, Snorlax's self-damage, and discard-cost
Items** — all of which the L0 default ("use abilities/searches unconditionally, fire draw-Supporters at
hand ≤5") can mis-fire.

| card | self-harm | fire-condition (state variables) | severity |
|---|---|---|---|
| **Judge ×3 (1213)** | shuffles **your** hand into deck, redraw only **4** | fire only if `hand_playable_win_pieces == 0` **and** (your hand < 4 useful **or** opp `handCount ≥ 6`). Never on a turn you hold Zacian+Band+Postwick not-yet-in-play. | **catastrophic** (H1) |
| **Lillie's Determination ×4 (1227)** | shuffles **your** hand into deck (redraws 6/8) | fire if `hand` is clogged / lacks a playable win-piece; safe because it refills 6–8, but still buries an about-to-be-played combo | major (H1) |
| **Hop's Snorlax ×2 (304)** | Dynamic Press does **80 to itself**; going Active exposes the Extra-Helpings anchor | attack with Snorlax only as desperation (`no Zacian available AND lethal/near-lethal`); otherwise keep on Bench | major (H6) |
| **Ultra Ball ×2 (1121)** | discard **2** from hand | fire only if `≥2 non-key cards in hand` and a needed Pokémon is missing | minor (H10) |
| **Secret Box ×1 (1092)** | discard **3** from hand | fire only with `≥3 discardable` and you need the toolbox turn | minor (H10) |
| Cramorant (311) | not self-harm but **whiffs (0 dmg)** outside prize-3/4 window | see H4 | major (H4) |
| all others | none | — | — |

Because self-harm cards exist, a **catastrophic** hypothesis is mandatory: **H1** (hand-shuffle Supporter
fired on a hand that holds the assembled win-turn).

---

## §3 Prize arithmetic

- **Zacian ex** 230 HP, gives up **2 prizes**, weak to Fire (×2 → a Fire attacker or a 330-club OHKOs it).
  But Zacian OHKOs back (330), so it usually trades **2-for-2 or better** and often 2-for-1 (it survives a
  non-Fire 200-ish hit and swings again).
- **Fezandipiti ex** 210 HP, **2 prizes** — a liability if it's the pilot's exposed Active for no reason.
- Single-prize bodies: Snorlax/Wooloo/Dubwool/Cramorant/Phantump/Trevenant/Patrat (1 each) — good prize
  "insulation"; leaning on Cramorant/Dubwool for a KO makes the opponent spend removal on 1-prizers.
- **KO chain requirement:** to win by turn ~T5–6 you need a KO nearly every turn from Brave Slash. Because
  Brave Slash **self-locks**, the chain needs either (a) Insta-Strike (120, may not KO) on the off turn, or
  (b) a **second loaded Zacian**. With 1 attach/turn and no accel, (a) is the default; (b) is the grind
  plan. **Not modelling the self-lock = a dropped KO every other turn** (H5).

---

## §4 Phase plan (measurable)

| phase | transition (state vars) | "on plan" | recovery on deviation |
|---|---|---|---|
| **early** | until a Zacian is Active with **≥1** {M} and Postwick down | Snorlax benched, Postwick played, Band on Zacian, **Insta-Strike each turn** (120 active +30 snipe); dig with Pokégear/Bag/Pad | if no Zacian in hand: Hop's Bag / Ultra / Poké Pad to find one; if no Metal: attach whatever, prioritize next Metal |
| **mid** | Zacian has **≥3 {M}** + Band | **Brave Slash 330** on fresh turns; on the self-lock turn Insta-Strike or swap to a 2nd loaded Zacian; Boss's the KO target | if buff pieces missing, still Brave Slash at 240/270/300 (already lethal vs most) — do **not** idle to "assemble all 3" against a low-HP target (T3) |
| **late** | opp prizes ≤ 3 | close with Brave Slash + Boss's on the last-prize target; **in the prize-3/4 window Cramorant Fickle Spitting = 150 for 1 energy**; Cruel Arrow (100, any target) snipes a lethal benched threat | if Zacians exhausted: Night Stretcher a Zacian back / promote a 1-prize attacker; avoid decking (stop Ultra/Secret) |

**Cards used per phase:** early = Insta-Strike, Postwick, Snorlax, Choice Band, all search/draw. Mid =
Brave Slash, Boss's Orders, Switch (rotate Zacians). Late = Boss's, Fickle Spitting, Cruel Arrow, Night
Stretcher. **Not used unless conditions met:** Judge/Lillie's (only on low/clogged non-combo hands),
Fickle Spitting (only prize-3/4), Corner (never — type-dead), Dynamic-Press-as-attack (desperation only).

---

## §5 Scorecard + tensions

| axis | score (1-5) | deck-specific definition / metric |
|---|---|---|
| 1 Speed | **3** | Brave Slash needs 3 manual Metal (no accel) → online ~T3; Insta-Strike chips from T1–2. `first_attack_turn` |
| 2 Power curve | **5** | 330 OHKO ceiling; even stripped to 240 it OHKOs most. Expected dmg by phase |
| 3 Stability | **4** | deep search (Pokégear/Bag/Pad/Petrel/Ultra/Secret), 3 Zacian. Brick rate low. `hand_size_dist` |
| 4 Continuation | **3** | Night Stretcher ×2 + 3 Zacian, but **energy-slow to reload** (1/turn, 10 total). `post_ko_attack_rate` |
| 5 Adaptability | **3** | Boss's ×3 bypass walls, Cruel Arrow snipes; **no answer to energy denial / no accel**. `gust_targets` |
| 6 Resource economy | **3** | only 10 energy, 1/turn; but low demand (Insta 1, Brave 3). Deckout far. `loss_share[deckout]` |
| 7 Disruption resistance | **2** | buff lattice is 3 removable pieces (Stadium war, Snorlax gust, Band removal); Metal denial is crippling (no accel); own hand fragile to opp Judge. `trigger_fire` |
| 8 Side-race | **4** | Zacian OHKOs win the race; two ex (Zacian/Fezandipiti) leak 2-prize turns; 1-prizers insulate. `loss_share[prize]` |

**Tensions (must-monitor pairs):**
- **T1 — tempo vs board development.** Energy is 1/turn: feeding a 2nd Zacian / Snorlax steals Metal from
  the first Zacian. Balance: **all Metal to the first Zacian until it has 3; only then fork.** (monitor
  `energy_attach_share` split vs `first_attack_turn`).
- **T2 — dig vs preservation.** Ultra/Secret/Judge thin/shuffle to find pieces, but discard/bury the win
  combo and edge toward deckout. Balance: **fire hand-shuffle/discard only when hand lacks a playable
  win-piece AND hand-size is low/clogged.** (monitor `play_rate[Judge/Lillie's/Ultra]` vs `loss_share[deckout]`).
- **T3 — buff ceiling vs resilience.** Chasing all 3 buffs for 330 wastes turns when 240–300 already
  kills; but under-buffing risks a non-KO. Balance: **buff to the minimum that OHKOs the current target;
  don't idle to assemble the 4th 30.** (monitor `nonattacking_turn_rate`).

---

## §6 Per-card usage declarations (all 60) + L0-default audit

State-distribution baseline used for calibration: hand ~5–7 (heavy draw), Metal-on-Zacian 0→3 over the
first turns, bench 2–4 Hop's basics, buffs assembled by ~T3.

**L0-default audit (each item checked against this deck):**
1. *Abilities used unconditionally* → **safe here.** Extra Helpings is passive; Flip the Script is
   conditional-beneficial (draw 3 on your KO); Defiant Horn is on-evolve gust (upside). No self-removal
   ability. (§2b's real hazard is Supporters, not abilities.)
2. *Energy → highest-DISPLAY attacker* → **mostly correct:** Zacian's 240 is the max display so Metal flows
   to Zacian. **But** (a) Choice Band drops the true cost to 3 — the loader may over-attach a 4th (H7);
   (b) **Mist counts as an energy but not as {M}** — a Zacian at 2 Metal + Mist reads "loaded" but can't
   Brave Slash (H2); (c) Cruel Arrow/Fickle Spitting show 0/120 with a false face (H3/H4).
3. *Draw-Supporter only at hand ≤5* → **dangerous:** Lillie's/Judge **shuffle your hand away**; firing at
   hand ≤5 on a hand that holds the assembled win-turn buries it (H1).
4. *Search/ball items every turn unconditionally* → Ultra (−2)/Secret (−3) burn cards and thin the deck;
   fine early, edges deckout late (H10).
5. *Evolve toward highest post-evo display* → low stakes (Zacian is a basic, ready immediately). Trevenant's
   90-face (Corner) is **type-dead** — do not funnel energy toward it (H9).
6. *Retreat if "loaded" by energy count* → the self-lock means you sometimes **want** to switch the locked
   Zacian to a fresh one; retreat/Switch gating by count is roughly OK but must respect the {M}-type load.
7. *Gust when own Active display ≥60* → Zacian's Brave Slash face (240) passes; on the Insta-turn the face
   is 30 (<60) so the engine may **not** gust despite Insta doing 120 — minor missed value.
8. *Post-KO promotion = highest display* → promotes a Zacian (240 face). Fine, but may promote an un-Metaled
   Zacian; ensure the promoted body has (or will get) 3 {M}.

**Per-card intents** (`fire_if` values are tunable hypotheses; "starves" = what over-tightening kills):
- **Zacian ex ×3 (299)** — the deck. Intent: primary Brave-Slash core. `fire_if`: Active + Band + ≥3 {M} → Brave Slash; else Insta-Strike. Rate/game ~all attacking turns. Starves: nothing (never dead). Brick only if 0 Zacian drawn (mitigated by Bag/Pad/Ultra).
- **Snorlax ×2 (304)** — buff anchor. Intent: **bench** for Extra Helpings; attack only as desperation. `fire_if` attack: `no Zacian available AND (dmg lethal)`. Rate: 1–2 in play; attack ~0. Starves: emergency offence.
- **Phantump ×2 / Trevenant ×1 (878/879)** — supplementary 1-prize wall/attacker. Intent: Phantump stall (coin dodge) / evolve to Trevenant for Horrifying Revenge (up to 220 after they KO a Hop's). `fire_if` Corner: **never (type-dead)**. Rate: low. Starves: revenge line.
- **Wooloo ×1 / Dubwool ×1 (309/310)** — Defiant-Horn gust package. Intent: evolve Dubwool to gust an opp benched to Active (steal a bad matchup). `fire_if`: evolve when a valuable opp benched target exists. Rate: ~0.3. Starves: free gust.
- **Cramorant ×1 (311)** — prize-window nuke. Intent: Fickle Spitting **only if opp prizes ∈ {3,4}** (150 with buff for 1 energy). `fire_if`: `opp_prizes ∈ {3,4}`. Rate ~0.3. Starves: a huge cheap KO — but firing outside the window = a wasted turn.
- **Fezandipiti ex ×1 (140)** — snipe/draw engine. Intent: Flip-the-Script draw on your KO'd turns; Cruel Arrow (100, any target, ignores W/R) as a finisher. `fire_if` Cruel Arrow: `Fezandipiti has 3 energy AND a 100-snipe is lethal/high-value`. Rate low but nonzero. Starves: bench-snipe finisher (H3). Risk: don't waste 2-prize body / don't over-feed its energy.
- **Patrat ×1 (626)** — Procurement item-tutor. Intent: T1 grab a key Item (Band/Switch/Ball) if desperate; otherwise a spare basic. Rate ~0.2.
- **Lillie's Determination ×4 (1227)** — main refuel. Intent: shuffle+draw 6/8 when hand is clogged/low and holds no about-to-play combo. `fire_if`: `hand_playable_win_pieces == 0 AND hand ≤ 5` (A∈{4,5,6}). Rate ~1/game+. Starves: refuel — but firing on a combo hand buries it (H1).
- **Judge ×3 (1213)** — disruption + draw. Intent: shuffle both hands (you 4) to **reset opponent** when their hand is stocked; **not** as your own dig. `fire_if`: `opp_handCount ≥ 6 AND your hand < 4 useful AND no combo in hand`. Rate ~0.5. Starves: disruption — but reckless use = **catastrophic self-mill of your turn** (H1).
- **Boss's Orders ×3 (1182)** — reach. Intent: gust the lethal/highest-value target (prize math, not face). `fire_if`: `a gust enables a KO or pulls a key support (Snorlax/attacker)`. Rate ~1.5. Starves: reach past walls.
- **Team Rocket's Petrel ×2 (1219)** — trainer tutor. Intent: grab a missing Supporter/Item (Boss's/Band/Postwick). Rate ~0.6.
- **Pokégear 3.0 ×4 (1122)** — Supporter finder. Intent: dig to the Supporter you need each turn (early: Lillie's/Bag; late: Boss's). `fire_if`: no Supporter in hand. Rate ~1.5.
- **Hop's Bag ×3 (1115)** — Hop's-basic engine. Intent: fetch 2 Basic Hop's (Zacian + Snorlax) to Bench T1–2. `fire_if`: bench < 4 OR missing Zacian/Snorlax. Rate ~1.5. Starves: board build.
- **Poké Pad ×2 (1152)** — non-rule-box tutor. Intent: fetch Snorlax/Cramorant/Dubwool/Patrat (NOT Zacian/Fezandipiti — they have rule boxes). Rate ~0.6.
- **Ultra Ball ×2 (1121)** — any-Pokémon tutor (−2). Intent: find Zacian/Snorlax when Bag can't. `fire_if`: `missing key Pokémon AND ≥2 discardable`. Rate ~0.8. Starves: consistency (but discard discipline, H10).
- **Secret Box ×1 (1092)** — toolbox (−3). Intent: one explosive turn assembling Band+Postwick+Supporter+Item. `fire_if`: `≥3 discardable AND it completes the buff lattice`. Rate ~0.4.
- **Night Stretcher ×2 (1097)** — recovery. Intent: return a KO'd Zacian or a Basic {M} from discard. `fire_if`: `Zacian in discard AND <2 Zacian in play/hand` OR `Metal short`. Rate ~0.6. Starves: rebuild loop.
- **Switch ×3 (1123)** — rotation. Intent: swap the self-locked Zacian for a fresh loaded one; escape a bad Active. `fire_if`: `Active is Brave-locked AND a loaded Zacian on bench` OR `Active pinned`. Rate ~1. Starves: the tag-team plan (Doctrine B).
- **Hop's Choice Band ×4 (1171)** — the −{C}/+30 tool. Intent: one on **each** intended Zacian attacker. `fire_if`: `attacker is Hop's AND has no tool`. Rate ~1.5. Starves: both the cost cut and +30 (do not leave a Zacian band-less).
- **Postwick ×4 (1255)** — +30 stadium. Intent: keep one in play; replay after opponent's stadium. `fire_if`: `no Postwick in play`. Rate ~1.3. Starves: 30 of ceiling (H8).
- **Basic {M} ×8 (8)** — fuel. Intent: **all to the first Zacian until 3**, then fork. `fire_if`: attach every turn to the Metal-short intended attacker. Rate ~1/turn. Starves: everything (this is the bottleneck).
- **Mist Energy ×2 (11)** — colorless + effect-immunity tech. Intent: use as a {C} filler on a colorless attacker, or on a Zacian for effect-immunity — **but never treat it as one of the 3 {M}** and never lean on it for Brave Slash. `fire_if`: `attacker's remaining need is {C}` OR `you want effect-immunity on the Active`. Rate ~0.4. Starves: nothing critical; misuse = a false-loaded Zacian (H2).

---

## §7 Loss-mode hypotheses & L2 rule candidates

**Expected loss distribution:** (1) **tempo/whack** — Brave-Slash self-lock turns wasted, or a non-KO from
under-buffing (largest); (2) **disruption** — buff pieces (Postwick/Snorlax/Band) removed or Metal denied
(no accel to recover); (3) **prize** — leaking 2-prize ex turns; (4) **deckout** — tail only.

**Applied patterns (priority):**
1. *Scale/effect attack undervalued* → **promote real-damage to shared perception** for Cruel Arrow (100
   any-target, display 0), Fickle Spitting (120 conditional), Insta-Strike (30→120), Horrifying Revenge.
   Gust/promotion/lethal checks must read the real formula. (H3, H4, H5)
2. *Key card play-rate ~0 / conditional whiff* → gate Fickle Spitting on `opp_prizes ∈ {3,4}`; Cruel Arrow
   on a lethal snipe. (H3, H4)
3. *Energy to the wrong body / false-load* → **{M}-typed load check** (count Metal, not energy) and the
   Choice-Band cost model; don't over-attach past 3. (H2, H7)
4. *Main attacker KO'd → can't swing* → model the Brave-Slash self-lock: off-turn = Insta-Strike or Switch
   to a 2nd loaded Zacian. (H5)
5. *Self-harm unconditional fire* → gate hand-shuffle Supporters (Judge/Lillie's) and Snorlax-attack on the
   §2b conditions. (H1, H6)
6. *Deckout* → stop Ultra/Secret when `deckCount` low. (H10)

**Is L2 needed?** The core plan (Zacian Brave Slash 330) is so dominant, and the L0 defaults align well on
the biggest levers (energy→highest-display = Zacian; gust when Zacian ready; promote a Zacian), that a
**generic aggro L1 very likely already wins the majority of games.** The deck-specific edges are real but
**mostly marginal** (Cruel Arrow finisher, Fickle window, off-turn Insta, buff-piece ordering) with **one
genuinely dangerous default** (unconditional hand-shuffle Supporters — H1) and **one correctness trap**
(Mist/Choice-Band load accounting — H2). **Recommendation: build L1 first; only add L2 rules if telemetry
confirms H1/H2/H5** (the self-mill, the false-load, or the off-turn passivity). If H1/H2/H5 come back
supported by the generic engine, **do not author L2** — calibrate the few gates instead. Symptom invention
is out of scope; every hypothesis below is falsifiable against P1 telemetry.
