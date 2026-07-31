"""Shared ID / enum maps + token strings (component C).

Used by the serializer, the action decoder, and the tokenizer extension so every
part of the pipeline speaks the same finite symbol space. `special_tokens()` lists
the candidate special tokens to add to the tokenizer (whether they actually shorten
sequences vs. the base 248k BPE is to be measured before committing — see plan §4-3).
"""
from cg.api import (  # noqa: E402  (lm/__init__ has set sys.path)
    OptionType, SelectContext, AreaType, EnergyType,
    all_card_data, all_attack,
)

_CARDS = {c.cardId: c for c in all_card_data()}
_ATTACKS = {a.attackId: a for a in all_attack()}

# Evolution reverse index. ``evolvesFrom`` in the DB is a NAME string; map it to an id
# and build the forward (evolves-TO) index by name so a card block can name both ends.
_NAME2ID = {}
for _c in _CARDS.values():
    _NAME2ID.setdefault(_c.name, _c.cardId)
_EVOLVES_TO = {}
for _c in _CARDS.values():
    _ef = getattr(_c, "evolvesFrom", None)
    if _ef:
        _EVOLVES_TO.setdefault(_ef, []).append(_c.cardId)

# EnergyType -> single-letter tag (C=colorless, N=dragon, *=rainbow, TR=team rocket)
_ENERGY_LETTER = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P",
                  6: "F", 7: "D", 8: "M", 9: "N", 10: "*", 11: "TR"}


def evolves_from_id(cid):
    c = _CARDS.get(cid)
    ef = getattr(c, "evolvesFrom", None) if c else None
    return _NAME2ID.get(ef) if ef else None


def evolves_to_ids(cid):
    c = _CARDS.get(cid)
    return _EVOLVES_TO.get(c.name, []) if c else []


def etype_letter(t):
    return _ENERGY_LETTER.get(t, "?") if t is not None else "-"


def card(cid):
    return _CARDS.get(cid)


def card_name(cid):
    c = _CARDS.get(cid)
    return c.name if c else f"?{cid}"


def card_tok(cid):
    return f"c{cid}"


def attack_tok(aid):
    return f"a{aid}"


def attack_dmg(aid):
    a = _ATTACKS.get(aid)
    return a.damage if a else 0


def energy_letters(elist):
    return "".join(_ENERGY_LETTER.get(e, "?") for e in (elist or []))


def _enum_name(cls, v):
    try:
        return cls(v).name
    except Exception:
        return str(v)


def area_name(a):
    return _enum_name(AreaType, a) if a is not None else ""


def opt_name(t):
    return _enum_name(OptionType, t)


def ctx_name(c):
    return _enum_name(SelectContext, c)


def _fleet_names():
    """Deck names + archetypes, for the opponent-identification segment."""
    import json, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        tun = json.load(open(os.path.join(root, "agents", "tuning.json")))
    except Exception:
        return [], []
    decks = sorted(k for k, v in tun.items() if isinstance(v, dict) and v.get("archetype"))
    arches = sorted({tun[k]["archetype"] for k in decks})
    return decks, arches


def deck_tok(name):
    return f"d_{name}"


def arch_tok(name):
    return f"a_{name}"


def special_tokens():
    """Candidate special tokens for tokenizer extension (card/attack ids + enums).

    Deck and archetype names are included so the opponent-identification segment
    (`OP d_alakazam_xero:9 a_combo:9`) costs ONE token per name instead of being
    shredded into subwords by the base BPE."""
    toks = [f"c{cid}" for cid in _CARDS]
    toks += [f"a{aid}" for aid in _ATTACKS]
    toks += [f"ctx_{c.name}" for c in SelectContext]
    toks += [f"opt_{o.name}" for o in OptionType]
    toks += [f"area_{a.name}" for a in AreaType]
    decks, arches = _fleet_names()
    toks += [deck_tok(d) for d in decks]
    toks += [arch_tok(a) for a in arches]
    return toks
