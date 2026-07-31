"""Fleet audit: cards that can NEVER do their job in the deck they are sitting in.

These are DECKBUILDING bugs, not engine bugs -- no amount of piloting fixes a card whose
effect has no legal target in its own 60. They surfaced while authoring card_roles
(docs/card_roles_guide.md), because tiering a card forces you to read what it needs.

Each check answers one question: "is there anything in THIS deck this card can resolve
for?" A check only fires when the answer is a hard NO from the deck list alone -- board
state can never rescue it.

Usage:
    python tools/audit_dead_cards.py            # all decks
    python tools/audit_dead_cards.py crustle    # one deck
"""
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cg-lib"))

from cg.api import CardType, EnergyType, all_attack, all_card_data  # noqa: E402

_CARDS = {c.cardId: c for c in all_card_data()}
_ATTACKS = {a.attackId: a for a in all_attack()}


def _norm(s):
    """The DB mixes both apostrophes (53 names use U+2019). Never match a name raw --
    see tools/check_name_matching.py."""
    return (s or "").replace("’", "'").lower()

_ENERGY_TYPES = (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
_WILD = {EnergyType.RAINBOW, EnergyType.TEAM_ROCKET}


def _is_pokemon(c):
    return c.cardType == CardType.POKEMON


def _has_rule_box(c):
    return bool(c.ex or c.megaEx)


def _stage(c):
    return 0 if c.basic else 1 if c.stage1 else 2 if c.stage2 else -1


def _evolution(c):
    return _is_pokemon(c) and not c.basic


def check_deck(name):
    deck = [int(l) for l in open(os.path.join(ROOT, "decks", name + ".csv")) if l.strip()]
    counts = Counter(deck)
    cards = {cid: _CARDS[cid] for cid in counts if cid in _CARDS}
    pokes = [c for c in cards.values() if _is_pokemon(c)]
    names = {cid: c.name for cid, c in cards.items()}
    findings = []

    def dead(cid, why):
        findings.append((counts[cid], cid, names.get(cid, "?"), why))

    # ---- Trainers whose search class is empty in this deck --------------------
    if 1086 in cards and not [c for c in pokes if c.basic and (c.hp or 0) <= 70]:
        lo = min(((c.hp or 0), c.name) for c in pokes if c.basic) if any(c.basic for c in pokes) else (0, "-")
        dead(1086, f"Buddy-Buddy Poffin fetches Basics with HP<=70; this deck's smallest Basic is {lo[1]} HP{lo[0]}")

    if 1225 in cards and not [c for c in pokes if _evolution(c)]:
        dead(1225, "Hilda searches an Evolution Pokemon + an Energy; the deck has ZERO Evolution Pokemon "
                   "-> degrades to a worse Energy Search that also costs the turn's Supporter")

    if 1152 in cards:
        targets = [c for c in pokes if not _has_rule_box(c)]
        if not targets:
            dead(1152, "Poke Pad searches a Pokemon with no Rule Box; every Pokemon here has one")

    if 1119 in cards and not [c for c in cards.values() if c.cardType == CardType.BASIC_ENERGY]:
        dead(1119, "Energy Search fetches a Basic Energy; the deck runs none")

    # ---- Trainers/Energy whose REQUIRED partner class is absent ---------------
    # The first pass only checked "does this searcher have a target". Plenty of cards
    # instead require a PROPERTY of the deck: a Tera body, a Team Rocket's body, a
    # non-Rule-Box body, a Mega ex, or one specific basic energy type. Same failure
    # mode, same silence.
    def _etypes():
        return {c.energyType for c in cards.values() if c.cardType == CardType.BASIC_ENERGY}

    def _poke_etypes():
        return {c.energyType for c in pokes}

    has_tera = any(c.tera for c in pokes)
    has_tr = any("team rocket's" in _norm(c.name) for c in pokes)
    has_nonrb = any(not _has_rule_box(c) for c in pokes)
    has_megaex = any(c.megaEx for c in pokes)
    basic_e = _etypes()
    poke_e = _poke_etypes()

    TERA_HARD = {1098: "Glass Trumpet is playable ONLY with a Tera Pokemon in play",
                 1127: "Tera Orb searches a Tera Pokemon",
                 1165: "Sparkling Crystal only discounts a TERA Pokemon's attack",
                 1250: "Area Zero Underdepths only widens the bench for a player with a Tera Pokemon",
                 1201: "Briar's extra Prize only triggers on a KO by YOUR Tera Pokemon"}
    for cid, why in TERA_HARD.items():
        if cid in cards and not has_tera:
            dead(cid, why + "; this deck has ZERO Tera Pokemon")
    if 15 in cards and not has_tr:
        dead(15, "Team Rocket's Energy can ONLY be attached to a Team Rocket's Pokemon "
                 "(and is discarded otherwise); this deck has none")
    if 1175 in cards and not has_nonrb:
        dead(1175, "Brave Bangle only boosts a Pokemon with NO Rule Box; every Pokemon here has one")
    if 1229 in cards and not has_megaex:
        dead(1229, "Wally's Compassion heals a Mega Evolution Pokemon ex; this deck runs none")
    TYPED = {1142: (EnergyType.FIGHTING, "Fighting Gong fetches a Basic {F} Energy or Basic {F} Pokemon"),
             1146: (EnergyType.PSYCHIC, "Wondrous Patch attaches a Basic {P} Energy to a Benched {P} Pokemon"),
             1254: (EnergyType.LIGHTNING, "Levincia recovers Basic {L} Energy"),
             1094: (EnergyType.GRASS, "Bug Catching Set reveals {G} Pokemon / Basic {G} Energy"),
             1141: (EnergyType.FIGHTING, "Premium Power Pro boosts your {F} Pokemon")}
    for cid, (et, why) in TYPED.items():
        if cid not in cards:
            continue
        if et not in basic_e and et not in poke_e:
            dead(cid, f"{why}; this deck has neither a Basic {et.name} Energy nor a {et.name} Pokemon")

    if 1079 in cards:
        # Rare Candy needs a Stage 2 whose chain bottoms out at a Basic in the deck
        ok = False
        for c in pokes:
            if _stage(c) != 2:
                continue
            mid = next((m for m in _CARDS.values() if m.name == c.evolvesFrom), None)
            base = mid.evolvesFrom if mid else None
            if any(b.name == base for b in pokes if b.basic):
                ok = True
                break
        if not ok:
            dead(1079, "Rare Candy needs a Stage 2 in hand evolving from a Basic in play; "
                       "no complete Basic->Stage2 pair exists in this deck")

    # ---- copy attacks: three kinds, and they are NOT interchangeable ----------
    # Lumping them together (any "use it as this attack") is wrong in both directions:
    #  * DECK-MILL -- Slowking's Seek Inspiration (213) "discard the top card of your
    #    deck ... choose 1 of its attacks and use it as this attack". The payload lives
    #    in the DECK and is never played, so it needs NEITHER an evolution path NOR its
    #    own energy. This is why slowking runs Metagross with no Metang/Beldum/Rare
    #    Candy ON PURPOSE -- it is ammo. The first version of this audit called that a
    #    dead card because the copy exemption was only wired into the energy check.
    #  * BENCH -- N's Zoroark ex's Night Joker (403) copies "your Benched N's Pokemon's
    #    attacks". The payload must be IN PLAY, so it still needs an evolution path; only
    #    its energy cost is somebody else's problem.
    #  * OPPONENT -- Foul Play / Metronome / Gemstone Mimicry / Try to Imitate / Haughty
    #    Order copy the OPPONENT's attacks. They say nothing about our own cards, and
    #    treating them as a blanket exemption would SILENCE real findings.
    def _copy_kind(a):
        t = (a.text or "")
        if "as this attack" not in t:
            return None
        if "opponent" in t:
            return "opponent"
        if "top card of your deck" in t:
            return "deck_mill"
        if "Bench" in t:
            return "bench"
        return "opponent"                  # unknown flavour: assume no exemption
    kinds = {_copy_kind(a) for c in pokes for x in (c.attacks or [])
             for a in (_ATTACKS.get(x),) if a}
    mill_ammo = "deck_mill" in kinds       # our non-Rule-Box Pokemon are DECK ammo
    bench_ammo = "bench" in kinds          # our benched bodies are payloads someone pays for

    # ---- Pokemon that can never be PUT INTO PLAY ------------------------------
    has_candy = 1079 in cards
    for c in pokes:
        st = _stage(c)
        if st <= 0:
            continue
        if mill_ammo and not _has_rule_box(c):
            continue                       # ammo: milled from the deck, never played
        pre = c.evolvesFrom
        if any(p.name == pre for p in pokes):
            continue                       # normal evolution path exists
        if st == 2 and has_candy:
            mid = next((m for m in _CARDS.values() if m.name == pre), None)
            if mid is not None and any(p.name == mid.evolvesFrom for p in pokes):
                continue                   # Rare Candy skips the missing Stage 1
        if (c.skills or []):
            continue                       # may have a self-put Ability; not a hard NO
        dead(c.cardId, f"Stage {st}: evolves from '{pre}', which is NOT in the deck"
                       + (" (and Rare Candy cannot bridge it)" if has_candy else " (no Rare Candy either)"))

    # ---- Attackers whose costs this deck's energy can never pay ---------------
    # Three ways to be wrong here, all of which the first draft of this check WAS:
    #  1. A special energy's ``energyType`` field LIES about what it provides. Prism (16)
    #     is energyType=COLORLESS yet "provides every type" on a BASIC; Neo Upper (10)
    #     does the same on a STAGE 2; Ignition (17) gives {C}{C}{C} on an Evolution.
    #     Reading energyType alone declared comfey_yveltal's own win condition dead.
    #  2. A Pokemon with an ABILITY is not here to attack (Munkidori, Froslass, Latias,
    #     Articuno, Dusknoir, Cornerstone Ogerpon...). Being unable to attack is the plan.
    #  3. A pre-evolution's job is to evolve, not to attack.
    def _supply(card):
        """(types this energy can provide, predicate on the Pokemon holding it)."""
        txt = " ".join((s.text or "") for s in (card.skills or []))
        if card.energyType in _WILD:
            return set(EnergyType), lambda c: True
        if "every type of Energy" in txt:                    # conditional wild
            if "attached to a Basic" in txt:
                return set(EnergyType), lambda c: c.basic
            if "attached to a Stage 2" in txt:
                return set(EnergyType), lambda c: _stage(c) == 2
            if "attached to an Evolution" in txt:
                return set(EnergyType), lambda c: _evolution(c)
            return set(EnergyType), lambda c: True
        return {card.energyType}, lambda c: True

    #  4. A COPY attacker makes our own bodies legitimate payloads that somebody ELSE
    #     pays for -- but only the deck-mill and bench flavours (see _copy_kind above).
    #     N's Reshiram's Fire+Lightning cost is paid by N's Zoroark ex's {D}{D} Night
    #     Joker; Slowking's Seek Inspiration fires Kyurem's 5-energy Trifrost for {P}{C}.
    #     An OPPONENT-copy attack (Foul Play, Metronome...) grants no such exemption.
    supply = [_supply(c) for c in cards.values() if c.cardType in _ENERGY_TYPES]
    for c in pokes:
        if c.skills:
            continue                       # played for its Ability, not its attack
        if any(p.evolvesFrom == c.name for p in pokes):
            continue                       # a pre-evolution: its job is to evolve
        if (mill_ammo or bench_ammo) and not _has_rule_box(c):
            continue                       # copy payload: someone else pays its cost
        atks = [a for a in (_ATTACKS.get(x) for x in (c.attacks or [])) if a]
        if not atks:
            continue
        avail = set()
        for types, holder_ok in supply:
            if holder_ok(c):
                avail |= types
        payable = any(
            {e for e in (getattr(a, "energies", None) or []) if e != EnergyType.COLORLESS} <= avail
            for a in atks)
        if not payable:
            costs = {e for a in atks for e in (getattr(a, "energies", None) or [])
                     if e != EnergyType.COLORLESS}
            dead(c.cardId, f"no Ability and no payable attack: needs {sorted(costs)}, "
                           f"this deck can supply it {sorted(avail)}")
    return findings


def self_test():
    """Prove every check CAN fire. A check that never fires is indistinguishable from
    "no bugs found" -- which is the exact failure mode this tool exists to catch, and
    which the first version of the name-matching guard actually had. Build a synthetic
    deck that violates each rule and assert the finding appears.
    """
    import tempfile
    fails = []
    # (card under test, partner cards that make it LIVE, expectation)
    CASES = [
        (1086, [305], "Poffin + Dunsparce HP70 -> fine"),           # control: must NOT fire
        (1086, [756], "Poffin + only HP300 Basics -> DEAD"),
        (1225, [756], "Hilda + zero Evolutions -> DEAD"),
        (1225, [741, 742], "Hilda + Abra/Kadabra -> fine"),         # control
        (1127, [756], "Tera Orb + zero Tera -> DEAD"),
        (15, [756], "Team Rocket's Energy + zero TR Pokemon -> DEAD"),
        (1175, [756], "Brave Bangle + only Rule Box bodies -> DEAD"),
        (1229, [305], "Wally's + zero Mega ex -> DEAD"),
        (1146, [305], "Wondrous Patch + no {P} at all -> DEAD"),
    ]
    d = tempfile.mkdtemp()
    for cid, partners, label in CASES:
        deck = [cid] * 4 + partners * 4
        deck += [1] * (60 - len(deck))                  # pad with Basic {G} Energy
        name = f"_selftest_{cid}_{'_'.join(map(str, partners))}"
        with open(os.path.join(ROOT, "decks", name + ".csv"), "w") as f:
            f.write("\n".join(str(x) for x in deck))
        try:
            found = {x[1] for x in check_deck(name)}
            fired = cid in found
            want = "DEAD" in label
            if fired != want:
                fails.append(f"{label}: check {'fired' if fired else 'did NOT fire'}, expected "
                             f"{'a finding' if want else 'none'}")
        finally:
            os.remove(os.path.join(ROOT, "decks", name + ".csv"))
    for f in fails:
        print("SELF-TEST FAIL " + f)
    print(f"self-test: {len(CASES) - len(fails)}/{len(CASES)} checks behave as specified")
    return not fails


def main(only=None):
    decks = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "decks")) if f.endswith(".csv"))
    total = slots = 0
    for name in decks:
        if only and name not in only:
            continue
        f = check_deck(name)
        if not f:
            continue
        print(f"\n{name}")
        for n, cid, nm, why in sorted(f, reverse=True):
            print(f"   [{n}x] {cid:5d} {nm:28s} {why}")
            total += 1
            slots += n
    print(f"\n{total} dead card(s) across the fleet, {slots} deck slots")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    main(only=set(a for a in sys.argv[1:] if not a.startswith("-")) or None)
