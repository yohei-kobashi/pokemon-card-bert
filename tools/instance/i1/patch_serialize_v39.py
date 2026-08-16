"""Prompt format v39: role-grouped DECK, no ID ME, and the board facts the model was missing.

Measured motivation, all from today:

* `glossary="none"` removed EVERY card semantic: across 40,000 prompts, attack costs appear 0
  times, retreat costs 0 times, attack damage 0 times. The model sees card ids and HP only.
* Ceiling-adjusted per-kind accuracy puts `attach` at 20.3% of its ceiling and `retreat` at
  33.6% (chance 31.2%) -- the two worst by a wide margin, and the ONLY two whose decisive input
  (cheapest attack cost, retreat cost) is absent from the prompt. Everything else runs 52-100%.
* Role labels carry real signal: a role-only predictor scores 43.4% against 20.7% random on
  card-bearing options, and 20% of cards hold a different role in different decks.

Changes, each behind an explicit parameter so v37 stays reproducible for A/B:

  deck_mode="roles"   DECK is grouped by role, groups in a FIXED order, ids sorted WITHIN a
                      group. That kills the file-order fingerprint structurally, so shuffling
                      is neither needed nor applied. (Under deck_mode="remaining" the list was
                      already id-sorted, so the shuffle was randomising an order that carried
                      no fingerprint anyway.)
  board_facts=True    each in-play Pokemon gains `need:N` (energies short of its cheapest
                      damaging attack) and `rt:N` (retreat cost), on BOTH sides -- the opponent's
                      numbers are what retreat and target choice depend on. Attached energy
                      switches to the count form `|G3` from `|GGG`.
  identify="op"       drop `ID ME`; keep the opponent prediction. Recorded judgment: "ID ME
                      really is redundant given DECK[]", and engine_v2 is 0.0% sensitive to its
                      own decklist, so the label side barely encodes deck identity.
"""
import os
import re

P = os.path.join(os.getcwd(), "lm/serialize.py")
s = open(P).read()

if "board_facts" in s:
    print("already patched")
    raise SystemExit(0)

# ---------------------------------------------------------------- helpers + attack table
ANCH = "def _pk(p):"
assert s.count(ANCH) == 1, "_pk anchor"
HELP = '''_ATTACKS = None


def _attack_table():
    global _ATTACKS
    if _ATTACKS is None:
        from cg.api import all_attack
        _ATTACKS = {a.attackId: a for a in all_attack()}
    return _ATTACKS


def _cheapest_attack_cost(cid):
    """Energies needed by the cheapest DAMAGING attack, or None if the card has none.

    Mirrors agents/_engine._cheapest_cost so the number the prompt shows is the number the
    engine acts on. Non-damaging attacks are excluded: they are not what attaching is for.
    """
    c = vocab._CARDS.get(cid)
    if not c or not c.attacks:
        return None
    at = _attack_table()
    costs = [len(at[a].energies or []) for a in c.attacks if at.get(a) and at[a].damage]
    return min(costs) if costs else None


def _board_facts(p):
    """` need:N rt:N` for one in-play Pokemon -- absent from v37 prompts entirely."""
    cid = p.get("id")
    out = []
    need = _cheapest_attack_cost(cid)
    if need is not None:
        out.append("need:%d" % max(0, need - len(p.get("energies") or [])))
    c = vocab._CARDS.get(cid)
    if c is not None and c.retreatCost is not None:
        out.append("rt:%d" % c.retreatCost)
    return (" " + " ".join(out)) if out else ""


'''
s = s.replace(ANCH, HELP + ANCH)

# ---------------------------------------------------------------- _pk gains the facts
OLD_PK = '''    e = p.get("energies") or []
    if e:
        s += f"|{vocab.energy_letters(e)}"'''
NEW_PK = '''    e = p.get("energies") or []
    if e:
        # count form: |G3 not |GGG. Same tokens on average, but the QUANTITY is a number the
        # model reads instead of a run of characters it has to count against a cost.
        s += "|" + (_energy_counts(e) if board_facts else vocab.energy_letters(e))'''
assert s.count(OLD_PK) == 1, "_pk energy anchor"
s = s.replace(OLD_PK, NEW_PK)

OLD_SIG = "def _pk(p):"
NEW_SIG = "def _pk(p, board_facts=False):"
s = s.replace(OLD_SIG, NEW_SIG, 1)

OLD_RET = '''    tools = _ids(p.get("tools"))
    if tools:                                   # tool CARDS (Cape/Belt change HP/damage)
        s += "|" + ",".join(vocab.card_tok(t) for t in tools)
    return s'''
NEW_RET = '''    tools = _ids(p.get("tools"))
    if tools:                                   # tool CARDS (Cape/Belt change HP/damage)
        s += "|" + ",".join(vocab.card_tok(t) for t in tools)
    if board_facts:
        s += _board_facts(p)
    return s'''
assert s.count(OLD_RET) == 1, "_pk tail anchor"
s = s.replace(OLD_RET, NEW_RET)

# energy count helper
s = s.replace("_ATTACKS = None\n", '''def _energy_counts(elist):
    """`GG C` -> `G2C` : one letter per DISTINCT type, with its count when above 1."""
    seen = []
    cnt = Counter(vocab.energy_letters(elist))
    for ch in vocab.energy_letters(elist):
        if ch not in seen:
            seen.append(ch)
    return "".join(ch + (str(cnt[ch]) if cnt[ch] > 1 else "") for ch in seen)


_ATTACKS = None
''', 1)

# ---------------------------------------------------------------- _side passes it through
OLD_SIDE = "def _side(pl, me):"
NEW_SIDE = "def _side(pl, me, board_facts=False):"
assert s.count(OLD_SIDE) == 1, "_side anchor"
s = s.replace(OLD_SIDE, NEW_SIDE)
s = s.replace('    s = f"A[{_pk(active)}]"', '    s = f"A[{_pk(active, board_facts)}]"')
s = s.replace('        s += " B[" + ",".join(_pk(b) for b in bench) + "]"',
              '        s += " B[" + ",".join(_pk(b, board_facts) for b in bench) + "]"')

open(P, "w").write(s)
print("patched _pk/_side with board_facts;", P)
