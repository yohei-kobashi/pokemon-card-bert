"""Make `need:N` TYPE-AWARE. The count-only version was wrong in a way that would mislead.

Attack costs are typed: `cost=[6,6,0]` is FIGHTING+FIGHTING+COLORLESS. Colorless accepts any
energy, a typed symbol does not. The first implementation compared LENGTHS only, so a Pokemon
holding two Psychic against an [FFC] attack came out as `need:1` when the truth is `need:2` --
the Psychics can only ever pay the single colorless. Since the entire point of `need` is to
supply the fact the prompt was missing, a wrong value is worse than no value.

Also redefines the quantity from "cheapest attack by card count" to "the SMALLEST shortfall over
all damaging attacks": with types, a longer cost whose symbols are already satisfied can be
closer to payable than a shorter one that needs a type we do not hold.

RAINBOW is treated as a wildcard against typed symbols; every other type must match exactly.
"""
import os

P = os.path.join(os.getcwd(), "lm/serialize.py")
s = open(P).read()

OLD_START = "def _cheapest_attack_cost(cid):"
i = s.index(OLD_START)
j = s.index("def _board_facts(p):")
NEW = '''def _shortfall(cost, attached):
    """How many MORE energies this cost needs, honouring types.

    Typed symbols must be paid by their own type (RAINBOW counts as any); whatever is left over
    -- of any type -- pays the colourless symbols.
    """
    from cg.api import EnergyType
    col = int(EnergyType.COLORLESS)
    rainbow = int(EnergyType.RAINBOW)
    req = Counter(int(x) for x in (cost or []))
    have = Counter(int(x) for x in (attached or []))
    short = 0
    for t, k in req.items():
        if t == col:
            continue
        use = min(k, have.get(t, 0))
        have[t] -= use
        k -= use
        if k and have.get(rainbow, 0):          # rainbow pays any typed symbol
            w = min(k, have[rainbow])
            have[rainbow] -= w
            k -= w
        short += k
    left = sum(v for v in have.values() if v > 0)
    short += max(0, req.get(col, 0) - left)
    return short


def _need_energy(cid, attached):
    """Smallest number of extra energies that makes ANY damaging attack payable.

    Not "the cheapest attack by card count": with types, a longer cost whose symbols are
    already covered can be closer to payable than a shorter one demanding a type we lack.
    """
    c = vocab._CARDS.get(cid)
    if not c or not c.attacks:
        return None
    at = _attack_table()
    shorts = [_shortfall(at[a].energies, attached)
              for a in c.attacks if at.get(a) and at[a].damage]
    return min(shorts) if shorts else None


'''
s = s[:i] + NEW + s[j:]

OLD_BF = '''    need = _cheapest_attack_cost(cid)
    if need is not None:
        out.append("need:%d" % max(0, need - len(p.get("energies") or [])))'''
NEW_BF = '''    need = _need_energy(cid, p.get("energies") or [])
    if need is not None:
        out.append("need:%d" % need)'''
assert s.count(OLD_BF) == 1, "board_facts anchor"
s = s.replace(OLD_BF, NEW_BF)

open(P, "w").write(s)
print("patched: need is now type-aware")
