# P0 Static Analysis — `mega_feraligatr`

Blind zero-shot analysis from the 60-card list + card DB + simulator probes only. Every mechanical
claim below is tagged **[V]** (verified by simulator probe) or **[T]** (text/type analysis, unprobed).

## 0. Deck list (60 / 27 unique)

**Pokémon (21)**
| n | cid | card | HP | type | wk | role |
|---|---|---|---|---|---|---|
|4|47|Totodile|70|W|L|Basic; Big Bite 10 [W] (opp can't retreat). Rare-Candy / Poffin target|
|3|48|Croconaw|90|W|L|St1; Reverse Thrust 30 [W] + switch self to bench|
|3|49|Feraligatr|180|W|L|**St2 workhorse**; Giant Wave 160 [W,W] (locks next turn) + Torrential Heart skill|
|1|939|Mega Feraligatr ex|**370**|W|L|**St2 megaEx finisher**; Mortal Crunch 200 [W,W,C], +200 vs damaged. **3 prizes**|
|3|112|Munkidori|110|**P**|D|Basic; Adrena-Brain skill (needs {D}). **Cannot attack in this deck**|
|2|305|Dunsparce|70|C|Fi|Basic; Trading Places (free switch) / Ram 20 [C,C]; Poffin target|
|2|66|Dudunsparce|140|C|Fi|St1; Run Away Draw (+3, shuffle self in) / Land Crush 90 [C,C,C]|
|1|1071|Meowth ex|170|C|Fi|Basic; Last-Ditch Catch (tutor Supporter on play) / Tuck Tail. **2 prizes**|
|1|235|Budew|30|G|F|Basic; Itchy Pollen 10 **[free]** — opp can't play Items next turn|
|1|343|Shaymin|80|G|F|Basic; Flower Curtain (block attack dmg to non-rulebox bench)|

**Trainers/Supporters (13):** Lillie's Determination ×4 (1227), Crispin ×2 (1198), Boss's Orders ×2 (1182),
Lucian ×2 (1237), Hilda ×1 (1225), Team Rocket's Petrel ×1 (1219), Rosa's Encouragement ×1 (1240).
**Items (15):** Poké Pad ×4 (1152), Buddy-Buddy Poffin ×4 (1086), Ultra Ball ×2 (1121), Pokégear 3.0 ×2 (1122),
Rare Candy ×2 (1079), Sacred Ash ×1 (1129). **Stadium (1):** Surfing Beach (1262).
**Energy (10):** Basic {W} ×6 (3), Basic {D} ×3 (7), Neo Upper Energy ×1 (10, **ACE SPEC**).

---

## §1 Win-condition spec

**Main win condition — Water Stage-2 beatdown with a pre-damage sub-engine.**

The deck attacks with the Feraligatr line and *manufactures pre-damage* so its two flagship attacks
over-perform their display numbers.

**Verified real-damage formulas** (probe: 30-game combo-seek + 120-game scan of HP_CHANGE logs):

- **Giant Wave (Feraligatr 49, [W,W])** `[V]`
  `dmg = 160 + (120 if Torrential Heart used this turn else 0)` → **280**.
  Torrential Heart (skill) *places 5 damage counters (50) on Feraligatr itself* (probe: HP −50, putDamageCounter=true).
  Giant Wave **self-locks**: "can't use Giant Wave next turn" → one Feraligatr fires every *other* turn.
- **Mortal Crunch (Mega Feraligatr ex 939, [W,W,C])** `[V]`
  `dmg = 200 + (200 if opponent Active has ≥1 damage counter else 0)` → **400** (probe: −200 vs fresh, −400 vs pre-damaged).
  No self-lock → Mega can attack **every** turn.

**The pre-damage engine (why Munkidori is the glue):** Munkidori Adrena-Brain `[V]` (requires ≥1 {D} attached)
*moves up to 3 damage counters (30) from one of YOUR Pokémon to an opponent's Pokémon* — observed as +30 heal on
Feraligatr and −30 placed on the target. This does two jobs at once:
1. **Heals** the self-damage Feraligatr inflicts on itself with Torrential Heart, and
2. **Places** a damage counter on the opponent's Active → arms Mortal Crunch's +200 (200→400).

So the intended loop is: Feraligatr Torrential-Heart-self-damages → Munkidori shuttles that damage onto the
opponent's Active → Mega Feraligatr ex crashes for 400 while Munkidori keeps healing it (370 HP tank).

**Procedure (canonical line):**
- T1–3: Totodile down, Munkidori benched with 1 {D}, dig for Rare Candy + a Stage2.
- T3–5: Rare Candy Totodile → Feraligatr (**verified Rare Candy skips Croconaw straight to either Stage2**);
  2 {W} → Giant Wave 160/280.
- T5+: Mega online (or continuous Giant Waves via Surfing-Beach rotation of a 2nd Feraligatr); Munkidori
  pre-damages every turn → Mortal Crunch 400 finisher.

**Plan B (main body falls / can't assemble):** Regular Feraligatr redundancy (3 copies) as 1-prize workhorses;
Dudunsparce Land Crush (90, [C,C,C]) as a colorless backup body; Sacred Ash + Dudunsparce recycle to rebuild
the line and outlast on prizes.

**Doctrines** (see JSON `doctrines`): **A** regular-Feraligatr engine (cheap, 3 bodies, 1-prize) vs
**B** Mega-tank finisher (400/turn, but 370-HP 3-prize liability, single copy) vs **C** Neo-Upper fast-track
(Neo on a Stage2 = 2 Rainbow, verified — Mega attacks with Neo + 1 energy). Discriminators specified in JSON.

---

## §2 Rules-interaction inventory

- **Energy carries up on evolution [T]** → pre-attaching {W} to Totodile/Croconaw is fine (rises to Feraligatr/Mega).
- **Damage vs damage counters:** Torrential Heart and Adrena-Brain both use **damage counters** (placement),
  which ignore Weakness/Resistance. Giant Wave / Mortal Crunch deal **damage** (subject to Weakness). Mega/Feraligatr
  are Lightning-weak → Lightning attackers double vs them (incl. the 370-HP Mega — a 185+ hit becomes lethal).
- **Energy TYPE vs supply (type-death check, all attackers):**
  | attacker | cost | payable by deck supply? |
  |---|---|---|
  | Feraligatr Giant Wave | [W,W] | ✅ 6 {W} + Crispin + Rosa + Neo(2 Rainbow on St2) |
  | Mega Mortal Crunch | [W,W,C] | ✅ {W}×2 + any (C slot = {D}/Neo/W) |
  | Croconaw / Totodile | [W] | ✅ |
  | Dudunsparce Land Crush | [C,C,C] | ✅ any 3 energy |
  | Dunsparce Ram / Shaymin | [C,C] | ✅ |
  | **Munkidori Mind Bend** | **[P,C]** | ❌ **TYPE-DEAD — no Psychic energy in deck.** `[V]` 0 attack-options offered across 120 games. Munkidori is an **ability-only body**, never an attacker. |
  | Budew Itchy Pollen | free | ✅ |
- **1-per-turn limits:** manual energy attach (1/turn) is the bottleneck for a [W,W]/[W,W,C] deck with only 6 {W};
  Supporter (1/turn) contends between draw (Lillie's) and tutor (Hilda/Boss/Crispin). Surfing Beach gives a *free*
  switch that dodges the 1-retreat cap and the heavy retreat costs (Feraligatr/Mega rc=3).
- **Self-draw / deckout horizon:** very deep engine (Lillie's×4, Lucian×2, Poffin×4, Pad×4, Ultra×2, Pokégear×2)
  vs a **slow clock** (Stage2 + energy + every-other-turn Giant Wave). **Probe (random mirror, 60 games): median
  last turn 54, and 59/60 games someone empties their deck.** Random play overstates length, but it confirms a
  hard **deckout floor** — the draw engine IS the loss engine. Counter-tools: Dudunsparce Run Away Draw (recycle),
  Sacred Ash (+5 Pokémon back), Meowth Tuck Tail.
- **ACE SPEC / stadium / special energy:** exactly 1 ACE SPEC (Neo Upper) — legal. Neo Upper `[V]`: **1 Colorless
  on a Basic, 2 Rainbow (every type) on a Stage2** → save it for a Stage2. Surfing Beach is the only stadium.

## §2b Self-harm sweep (all 60) — ★

| card | self-harm | fire-condition (state variables) | severity |
|---|---|---|---|
| **Feraligatr Torrential Heart** | places **50 self-damage/turn** | fire ONLY when Feraligatr attacks this turn AND (hp−50 > incoming OR Munkidori/Surfing-Beach preserves it). Clear rate (Adrena-Brain 30) < input (50) → net +20/turn. **Never on a Giant-Wave-locked/non-attacking turn.** | **catastrophic** (H1) |
| **Dudunsparce Run Away Draw** | shuffles **itself** into deck | require ≥1 *other* Pokémon in play; net deck −2 (draw 3, +1 back). | **catastrophic** if last body (H2) |
| **Meowth ex Tuck Tail** | bounces **itself** to hand | recycle energy only; never as last body. | catastrophic if last body (H2) |
| **Ultra Ball** | discard 2 from hand | protect Rare Candy / Stage2 / scarce {W}/{D}. | minor (H10) |
| **Lillie's Determination** | shuffles hand into deck | skip when holding assembled combo (Rare Candy + Stage2 + {W}). | major (H6) |
| **Lucian** | hand to bottom (symmetric) | dead-hand only; also helps opponent. | major (H6) |
| Munkidori Adrena-Brain | moves own counters (net **beneficial** heal) | not harmful; used as the pre-damage/heal engine. | — |
| Sacred Ash / Crispin / Rosa / Hilda / Petrel / Poffin / Pad / Pokégear / Boss / Surfing Beach / energies | none | — | none |

Net: **3 self-harm vectors** requiring gates (Torrential Heart, Run-Away-Draw/Tuck-Tail self-removal, Ultra/Lillie's
card loss). Torrential Heart is the standout catastrophic (an ability L0 fires unconditionally → self-KO).

---

## §3 Prize arithmetic

- **Mega Feraligatr ex:** 370 HP (survives almost any single non-Lightning hit) but **gives 3 prizes** — half the
  game if it falls. Weakness Lightning is the OHKO seam. Deploy only when protected + pre-damage guaranteed.
- **Regular Feraligatr:** 180 HP → OHKO'd by ~200 environment damage; **1 prize**. The trade-friendly workhorse.
- **Munkidori 110 / Meowth ex 170 (2 prizes) / Budew 30 / Shaymin 80:** fragile support; Budew is a free late prize.
- **Trade structure:** leading with 1-prize Feraligatr bodies keeps parity; Mega is the tempo swing (400/turn,
  self-healed) but must not be gusted/Lightning'd for its 3-prize bounty. Chain requirement: keep a 2nd Feraligatr
  charging so the post-KO body attacks immediately (else non-attacking turn → deckout drift).

---

## §4 Phase plan (measurable)

| phase | transition (state var) | "on plan" | recovery |
|---|---|---|---|
| **early** | own Stage2 in play w/ ≥1 {W}, or turn≥4 | Totodile down by T2; Munkidori benched w/ 1 {D}; Rare Candy+Stage2 or natural Croconaw in hand; Budew active T1 vs item decks | Poffin/Pad/Hilda to find missing line piece |
| **mid** | a Feraligatr w/ 2 {W} (Giant Wave online) | Giant Wave ≥ every-other-turn; 2nd Feraligatr charging; Munkidori moving self-damage → opp Active | Crispin/Rosa to re-fuel; Surfing Beach to rotate a fresh body |
| **late** | opp prizes ≤2, or Mega online w/ pre-damaged target | Mortal Crunch 400 / Giant Wave 280 each attackable turn; **deckCount > ~6** | Sacred Ash + Dudunsparce recycle; **STOP digging** |

Use-lists: **early** uses Poffin/Pad/Poke/Rare Candy/Budew; does NOT commit Mega. **mid** uses Feraligatr/Crispin/
Rosa/Surfing Beach; slows searching. **late** uses Boss (gust)/Mega/Sacred Ash; does NOT fire Lillie's/Lucian/search
into a thin deck.

---

## §5 Scorecard + tensions

Speed 2 · Power-curve 4 · Consistency 3 · Recovery 4 · Adaptability 3 · Resource-economy 2 ·
Disruption-resistance 2 · Prize-race 3 · **(invented) Self-harm-management 2**. (Definitions + verification
metrics in JSON `axes_notes`.) The invented axis derives from the win-con variable "Feraligatr self-damage vs
Adrena-Brain clear rate."

**Tensions (★):**
1. **dig/consistency ↔ deckout survival** — the tutor/draw suite is the consistency engine *and* the deck-out clock.
   Balance: dig freely while `deckCount>12 AND combo not assembled`; else stop and recycle.
2. **Torrential-Heart burst (+120) ↔ Feraligatr survival** — +50 self/turn vs −30 clear/turn (net +20). Fire only
   on attacking turns with a survival margin.
3. **Mega clock (400) ↔ 3-prize liability** — commit Mega only with a guaranteed pre-damaged target and protection.

P3 note: any fix to a tension member must co-monitor its pair (dig-rate ↔ deckout; Torrential-Heart-rate ↔
Feraligatr-KO) to avoid whack-a-mole.

---

## §6 Per-card use declarations & L0-default audit

Full 27-card intent table (with fire-conditions + play-rates + brick conditions) is in JSON `card_intents`.
Highlights vs the deck's steady-state (typical: hand 4–6, 1–2 {W} in play, Munkidori benched with 1 {D}):

**L0 default audit — item by item:**
1. *Abilities used unconditionally* → **Torrential Heart self-KOs Feraligatr** (H1, catastrophic); Run-Away-Draw
   self-removal (H2). Adrena-Brain auto-use is mostly fine but must target opp **Active** (H8).
2. *Energy to display-max attacker* → Mega(200)/Feraligatr(160) show real numbers, but **Munkidori (Mind Bend 60)
   attracts energy it can never use** and {W} is scarce (6) → **mis-routing to Munkidori starves the line** (H3).
   Also display hides the real 280/400 (H4).
3. *Draw supporter at hand≤5* → **Lillie's/Lucian shuffle away the assembled combo** (H6). Gate on possession.
4. *Search every turn* → **deckout** (H5). Add a dig-stop at low deckCount.
5. *Evolve to display-max* → prefers Mega(200) — may rush the 3-prize body before pre-damage exists (doctrine B risk).
6. *Won't retreat "charged" body* → a {D}-loaded **Munkidori is treated as charged and stuck Active** though it
   can't attack (H7). Use Surfing Beach / type-aware charged detection.
7. *Gust when own display ≥60* → fine trigger, but target selection must use real 280/400 + prize value (H4/H8).
8. *Post-KO promote display-max* → can promote an un-charged Mega or type-dead Munkidori (H9).

---

## §7 Loss-cause hypotheses & L2 candidates

Expected loss distribution (consistent with the reported ~24% WR / high deckout / Munkidori-energy report):
**deckout (dominant)** > self-KO/tempo (Torrential Heart, Munkidori mis-route) > prize (Mega gusted for 3) > board.

12 hypotheses in JSON (2 catastrophic: H1 Torrential-Heart self-KO, H2 last-body self-removal). Pattern mapping:
- H1/H2 → *board-state gate on self-harm effect* (new template: gate self-damage/self-removal abilities on
  survival + non-last-body).
- H3 → *deck-specific fueling* (cap Munkidori at 1 {D}, {W}→line by type).
- H4/H8 → *promote real-damage formula to shared perception* (Giant Wave 280, Mortal Crunch 400-when-pre-damaged).
- H5 → *deck-thin dig-stop + recycle activation*.
- H6/H10 → *possession-based firing* (skip draw-supporter / protect discard when combo held).
- H7/H9 → *type-aware charged/wall + prize-aware promotion*.
- H11 → *free-switch attack-uptime*. H12 → *phase-gated tech deployment*.

**L2 recommendation: RECOMMENDED.** H1/H3/H4 are structural deck-specific mis-defaults, not threshold calibration —
they need self-harm gates, type-aware fueling and real-damage perception that L0 cannot express. If P1 confirms
deckout-dominant losses with H1/H3 firing, an L2 rule set is justified over L1.

## Verification log
- Mortal Crunch 200/400 conditional on opp damage — **[V]** (HP_CHANGE −200 vs −400).
- Giant Wave 160→280 + Torrential Heart self −50 (damage counter) — **[V]**.
- Adrena-Brain moves 3 counters (heal own +30 / place opp −30) — **[V]**.
- Rare Candy Totodile → Feraligatr AND → Mega Feraligatr ex directly — **[V]** (EVOLVE_DONE both).
- Munkidori Mind Bend [P,C] unpayable → 0 attacks offered / never used — **[V]** (120-game scan).
- Neo Upper: 1 C on Basic, 2 Rainbow on Stage2 — **[V]**.
- Deckout floor: 59/60 random-mirror games empty a deck — **[V]** (direction only; magnitude needs L0 telemetry).
- Adrena-Brain {D}-gate exactness, Land Crush 90, energy carry-up — **[T]** (unprobed, text/type).
