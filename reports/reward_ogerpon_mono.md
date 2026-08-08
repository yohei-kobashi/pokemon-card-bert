# Reward design: ogerpon_mono

9.2% of the top-153. The simplest deck in the set — four copies of one attacker and seventeen
basic {G} — and the one whose damage formula points at the opponent's board in the wrong
direction for every other design here.

## The cards that matter

```
Teal Mask Ogerpon ex (210 HP) x4  -- the entire Pokemon line
  ABL Teal Dance          once per turn, attach a basic {G} from hand TO THIS POKEMON,
                          and if you did, draw a card
  ATK Myriad Leaf Shower  {G}{G}{G}  30 base, +30 FOR EACH ENERGY ATTACHED TO **BOTH**
                          ACTIVE POKEMON
17x basic {G} | Tera Orb x4 | Bug Catching Set x4 | Judge x4 | Crushing Hammer x4
Jumbo Ice Cream x3 | Lively Stadium x2 | Grow Grass Energy x2 | Hero's Cape x1
```

## The loop

```
every Ogerpon in play uses Teal Dance on its own turn -- each attaches to ITSELF and draws
  -> with the manual attachment that is 2+ energy a turn onto the Active, and every BENCHED
     Ogerpon is charging itself into the next attacker at the same time
  -> Myriad Leaf Shower: 30 + 30 x (our Active's energy + THEIR Active's energy)
```

At 5 energy on ours and 2 on theirs that is 30 + 30x7 = **240 for three {G}**.

**The opponent's energy is part of our damage.** Every other design in this set treats their
energy as something to strip; here it is a multiplier we are paid for. That is now the fourth
sign inversion across five decks:

| deck | their energy | their hand | our hand | our own damage |
|---|---|---|---|---|
| ogerpon_mono | **more = our damage** | — | — | — |
| marnie_grimmsnarl | strip | small | — | **fuel** |
| dudunsparce_box | — | **large = our damage** | — | — |
| alakazam_nz | strip (special only) | small | **large = our damage** | — |
| dragapult_dusknoir | strip | — | — | bad |

**A tension in our own list, flagged not resolved:** we run Crushing Hammer x4, which discards
their energy and therefore *lowers* Myriad Leaf Shower. Against a deck that attacks off two
energy the tempo is probably worth more than the 60 damage; against one that stacks six it is
not. The potential below scores the board, not the card, so it will price this correctly — but
if the policy learns to hold Crushing Hammer against big-energy decks, that is a signal the
DECKLIST wants revisiting, not a bug.

## Phi, in units of one prize

```
Phi_prize  = 1.00 x (their prizes left - ours)

Phi_damage = 0.40 * clip(shower_dmg / max(1, their_active_hp), 0, 1)
             where shower_dmg = 30 + 30 * (energy on our Active + energy on their Active)
           + 0.10 * clip(energy_on_our_active / 5, 0, 1)

Phi_next   = 0.15 * min(#benched Ogerpon with >= 2 {G}, 2) / 2 * 2
             # each benched Ogerpon self-charges; the deck's resilience is the NEXT attacker
             # already being paid for when the 210 HP Active falls

Phi_ready  = 0.15 * 1{our Active can pay {G}{G}{G}}
Phi_heal   = 0.08 * min(#Jumbo Ice Cream in hand, 2) / 2      # 210 HP needs the heal
Phi_deny   = the shared block, but WITHOUT the "their energy" term -- it has the wrong sign
             for this deck. Keep only: their Active cannot attack (0.12), their hand (0.06),
             asleep/paralysed (0.06).
```

`Phi_damage` reads their energy through the damage formula rather than as a separate term, so
the inversion is expressed once, in the place it actually comes from.

## What to verify before training this one

* **Does Teal Dance's attach show up as an ABILITY option or an ATTACH option?** The potential
  does not care, but a pilot that never fires it starves the deck, and this is exactly the
  shape of the Seek Inspiration failure (an ability whose displayed value is 0).
* **`Grow Grass Energy` and `Tera Orb`**: check whether either provides more than one {G} for
  the count, since the whole formula is "number of Energy attached", which for a card
  providing double is not the same as its printed count.
* Prize-matched separation before any training.

## Sources

- [Teal Mask Ogerpon ex Deck Guide and Deck List - Deltia's Gaming](https://deltiasgaming.com/pokemon-tcg-teal-mask-ogerpon-ex-deck-guide-and-deck-list/)
- [Best Teal Mask Ogerpon ex Deck List Guide | AlcastHQ](https://alcasthq.com/pokemontcg-ogerpon-ex-deck-list-guide/)
- [Teal Mask Ogerpon ex – Limitless](https://limitlesstcg.com/cards/TWM/25)
