"""Print a deck's cards with full rules text, for authoring tuning.json card_roles.

card_roles cannot be inferred mechanically -- copy count was measured to carry no signal
(a finished deck is almost all 4-ofs: alakazam 73%, crustle 87%) and the card TYPE is
exactly the global hierarchy that was measured wrong. It takes reading what each card
actually does in THIS deck. This dumps that reading material.

Usage:
    python tools/dump_deck.py crustle            # one deck
    python tools/dump_deck.py crustle alakazam   # several
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cg-lib"))

from collections import Counter  # noqa: E402

from cg.api import CardType, all_attack, all_card_data  # noqa: E402

_CARDS = {c.cardId: c for c in all_card_data()}
_ATTACKS = {a.attackId: a for a in all_attack()}

_TYPE = {
    CardType.POKEMON: "POKEMON", CardType.BASIC_ENERGY: "BASIC_ENERGY",
    CardType.SPECIAL_ENERGY: "SPECIAL_ENERGY", CardType.SUPPORTER: "SUPPORTER",
    CardType.ITEM: "ITEM", CardType.TOOL: "TOOL", CardType.STADIUM: "STADIUM",
}


def _wrap(text, indent=" " * 10, width=96):
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return ("\n" + indent).join(lines)


def dump(name):
    path = os.path.join(ROOT, "decks", name + ".csv")
    deck = [int(l) for l in open(path) if l.strip()]
    counts = Counter(deck)
    print(f"\n{'='*100}\nDECK {name}  ({len(deck)} cards, {len(counts)} unique)\n{'='*100}")

    def key(cid):
        c = _CARDS.get(cid)
        # Pokemon first, then by evolution stage so a line reads top-down
        order = {CardType.POKEMON: 0, CardType.SUPPORTER: 1, CardType.ITEM: 2,
                 CardType.TOOL: 3, CardType.STADIUM: 4}.get(c.cardType, 5) if c else 9
        stage = (0 if c.basic else 1 if c.stage1 else 2) if c and c.cardType == CardType.POKEMON else 0
        return (order, stage, cid)

    for cid in sorted(counts, key=key):
        c = _CARDS.get(cid)
        n = counts[cid]
        if not c:
            print(f"\n  [{n}x] id={cid}  <UNKNOWN CARD>")
            continue
        tags = []
        if c.megaEx:
            tags.append("megaEx(3 prizes)")
        elif c.ex:
            tags.append("ex(2 prizes)")
        if c.aceSpec:
            tags.append("ACE_SPEC(1-of by rule)")
        if c.tera:
            tags.append("tera")
        head = f"\n  [{n}x] id={cid:<5d} {c.name}   <{_TYPE.get(c.cardType, c.cardType)}>"
        if tags:
            head += "  " + " ".join(tags)
        print(head)
        if c.cardType == CardType.POKEMON:
            stage = "Basic" if c.basic else "Stage1" if c.stage1 else "Stage2" if c.stage2 else "?"
            ev = f" <- evolves from {c.evolvesFrom}" if c.evolvesFrom else ""
            print(f"          {stage} HP{c.hp} retreat{c.retreatCost} energyType={c.energyType}"
                  f" weak={c.weakness} resist={c.resistance}{ev}")
        for s in (c.skills or []):
            print(f"          ABILITY {s.name}: {_wrap(s.text)}")
        for aid in (c.attacks or []):
            a = _ATTACKS.get(aid)
            if not a:
                continue
            cost = getattr(a, "energies", None) or getattr(a, "cost", None)
            print(f"          ATTACK  {a.name} (id={aid}) dmg={a.damage} cost={cost}")
            if a.text:
                print(f"                  {_wrap(a.text, ' ' * 18)}")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(
        f[:-4] for f in os.listdir(os.path.join(ROOT, "decks")) if f.endswith(".csv"))
    for n in names:
        dump(n)
