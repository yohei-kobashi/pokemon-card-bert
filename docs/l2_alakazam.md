# alakazam — card-by-card role analysis (L2 foundation, 2026-07-11)

**Win condition**: get **Alakazam (743, Stage 2)** into the Active Spot with 1 {P},
grow the HAND as large as possible, then **Powerful Hand** = *place 2 damage counters
per card in hand on the opponent's Active* = **20 × handCount** to their Active. It
PLACES counters, so it ignores HP-based damage reduction / walls, but hits ONE target
(the Active) — pair with Boss's Orders to nuke the right Pokémon. Everything else in the
deck exists to (a) assemble the Abra→Kadabra→Alakazam line and (b) make the hand huge.

## Payoff
| card | role | detail |
|---|---|---|
| **Alakazam 743** ×4 | **PAYOFF** | Powerful Hand (1 {P}) = 20×hand to Active. *Psychic Draw*: on evolving into it, **draw 3** — evolving is itself a hand-grow + arms the payoff. |

## Evolution line (assemble the payoff)
| card | role | detail |
|---|---|---|
| Abra 741 ×4 | seed / pivot | basic. *Teleportation Attack* (1) = switch to bench → a threatened Abra escapes to safety. |
| Kadabra 742 ×4 | mid-evo / draw | *Psychic Draw*: on evolve, **draw 2**. Super Psy Bolt 30 (minor). |
| Rare Candy 1079 ×4 | evo accel | Abra→Alakazam skipping Kadabra = FAST payoff (still triggers Alakazam's draw 3, but skips Kadabra's draw 2 → tempo vs cards trade-off). |

## Hand-growth engine (the heart — every held card = +20 damage)
| card | net hand | detail |
|---|---|---|
| **Dawn 1231** ×4 | **+3** | search a Basic + Stage1 + Stage2 to HAND = fetches the whole Abra/Kadabra/Alakazam line at once. Best enabler. |
| **Hilda 1225** ×4 | **+2** | search an Evolution Pokémon + an Energy to hand. |
| Poké Pad 1152 ×4 | +1 | search a non-rulebox Pokémon to hand. |
| Enriching Energy 13 ×1 | +4 | special {C} energy; on attach, **draw 4** (big burst; also gives Dudunsparce/backup a colorless). |
| Telepath Psychic Energy 19 ×4 | search | provides {P}; on attach to a {P} Pokémon, search deck (thin + find). |
| Dudunsparce 66 / Dunsparce 305 ×3/3 | draw loop | *Run Away Draw*: **draw 3**, then shuffle Dudunsparce back → re-fetch + re-evolve for another draw 3 (repeatable engine). |
| Psychic Draw (742/743) | +2 / +3 | draw on each evolution. |

## Backup attacker (do NOT neglect)
| card | role | detail |
|---|---|---|
| **Dudunsparce 66** | **backup** | *Land Crush* 90 (cost 3). When Alakazam is KO'd this is the follow-up — **L0's habit of keeping energy on the board loads it; a "hand-preservation" build that stopped attaching left NO backup and lost (−1 to −4% vs L0).** |
| Dunsparce 305 | seed | basic for Dudunsparce; Trading Places (1) = pivot. |

## Disruption
| card | role | detail |
|---|---|---|
| Enhanced Hammer 1081 ×4 | energy denial | discard a **Special** Energy from an opponent's Pokémon (tech vs special-energy decks; dead vs basic-energy decks). |
| Boss's Orders 1182 ×3 | gust | pull a benched opponent to the Active — **Powerful Hand hits the Active, so Boss chooses WHAT you nuke** (a low-HP threat, or their key setup piece). |

## Recovery / consistency
| card | role | detail |
|---|---|---|
| Night Stretcher 1097 ×3 | recovery | a Pokémon or basic Energy from discard → hand. |
| Sacred Ash 1129 ×1 | recovery / anti-deckout | shuffle up to 5 Pokémon from discard into deck (recycle the line, avoid deckout in a long grind). |
| Lana's Aid 1184 ×1 | recovery (+hand) | up to 3 non-rulebox Pokémon / basic Energy from discard → hand. |
| Buddy-Buddy Poffin 1086 ×4 | setup | search 2 basics ≤70HP → **Bench** (Abra/Dunsparce). Develops board (does not grow the attacking hand). |

## Protection
| card | role | detail |
|---|---|---|
| Battle Cage 1264 ×1 | bench shield | Stadium: prevent damage counters on **Benched** Pokémon (both sides) from the opponent's attacks/abilities → shields the Abra/backup on the bench from spread/snipe. |

## Energy
| card | role | detail |
|---|---|---|
| Basic {P} 5 ×4 | energy | Powerful Hand needs 1 {P}. |
| Telepath {P} 19 ×4 | energy + search | see engine. |
| Enriching 13 ×1 | energy + draw 4 | see engine. |
(Only 1 energy powers the payoff; surplus energy arms Dudunsparce's cost-3 Land Crush.)

---

## L2 design implications (from the roles)
1. **Maximise hand BEFORE Powerful Hand**: sequence the net-positive growers
   (Dawn +3, Hilda +2, Poké Pad +1, Enriching +4, evolution draws) and *hold* the
   result — but this is exactly what L0 already does (it plays draw/search); the gain
   over L0 is small and easily eaten by noise.
2. **Never sacrifice the backup**: keep loading Dudunsparce (Land Crush) — proven that
   suppressing energy to "preserve the hand" loses (no follow-up after Alakazam KO).
3. **Boss targets the nuke**: Powerful Hand hits the Active, so Boss's Orders should pull
   the opponent's most valuable KO-able / most-threatening Pokémon into the Active before
   the nuke — a genuine per-deck decision (L0 gusts for a direct KO with the ACTIVE's
   damage, not for Powerful-Hand placement value).
4. **Enhanced Hammer** is dead vs basic-energy decks — only play it when the opponent
   actually has Special Energy attached (read the opponent board).
5. **Powerful Hand ignores HP walls** (places counters) → prioritise this line vs
   high-HP ex / damage-reduction decks; it's single-target, so it's weak vs wide boards
   (there a spread deck would be better — a matchup fact, not a piloting lever).
