"""Option (semantic) encoding + action decoding with legality check (component B).

`encode_option` renders one legal option as an index-independent semantic string
(e.g. ``attack:982``, ``play:c678``, ``attach:c6@ACTIVE0``). `decode_action` maps an
LM output back to ``select.option`` indices, accepting either raw indices ("0,3") or
semantic strings, and returns None on anything illegal — the AGENT then falls back to
the heuristic engine (never forfeits). The same `encode_option` is used to build the
menu the model sees, so encoding and decoding are consistent.
"""
from cg.api import OptionType
from lm import vocab


def _hand_card_id(o, obs):
    if obs is None:
        return o.get("cardId")
    try:
        cur = obs["current"]
        me = cur["players"][cur["yourIndex"]]
        hand = me.get("hand") or []
        i = o.get("index")
        if i is not None and 0 <= i < len(hand):
            return hand[i]["id"]
    except Exception:
        pass
    return o.get("cardId")


# AreaType value -> PlayerState key for CARD-option resolution
_AREA_KEY = {2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}


def _card_at(o, obs):
    """Resolve a CARD option's card id from its (area, index[, playerIndex]) in obs."""
    if obs is None:
        return o.get("cardId")
    try:
        cur = obs["current"]
        pi = o.get("playerIndex")
        if pi is None:
            pi = cur["yourIndex"]
        area, idx = o.get("area"), o.get("index")
        if area == 1:                       # DECK -> select.deck
            lst = (obs.get("select") or {}).get("deck") or []
        elif area == 7:                     # STADIUM
            lst = cur.get("stadium") or []
        elif area == 12:                    # LOOKING
            lst = cur.get("looking") or []
        else:
            lst = (cur["players"][pi].get(_AREA_KEY.get(area)) or []) if _AREA_KEY.get(area) else []
        if idx is not None and 0 <= idx < len(lst) and lst[idx]:
            return lst[idx].get("id")
    except Exception:
        pass
    return o.get("cardId")


def _ability_card_id(o, obs):
    """Which card's Ability is this? Not necessarily one in hand.

    _hand_card_id only indexes ``me["hand"]``, so an Ability on a Pokemon already in play
    resolved to None and rendered "ability:cNone" (measured: 14 in a 60-game sweep). An
    Ability option carries either an explicit cardId or an inPlayArea/inPlayIndex pointing
    at the body that owns it."""
    if o.get("cardId") is not None:
        return o["cardId"]
    try:
        cur = obs["current"]
        pi = o.get("playerIndex")
        if pi is None:
            pi = cur["yourIndex"]
        # An Ability option points at its owner either via inPlayArea/inPlayIndex OR via
        # plain area/index (observed: {"type":10,"area":5,"index":4} = bench[4]). Accept both.
        for area, idx in ((o.get("inPlayArea"), o.get("inPlayIndex")),
                          (o.get("area"), o.get("index"))):
            key = {4: "active", 5: "bench"}.get(area)
            if not key:
                continue
            lst = cur["players"][pi].get(key) or []
            if idx is not None and 0 <= idx < len(lst) and lst[idx]:
                return lst[idx].get("id")
    except Exception:
        pass
    return _hand_card_id(o, obs)


def _attached_energy_id(o, obs):
    """OptionType.ENERGY: area/index -> the Pokemon, energyIndex -> its attached energy."""
    if obs is None:
        return None
    try:
        cur = obs["current"]
        pi = o.get("playerIndex")
        if pi is None:
            pi = cur["yourIndex"]
        key = {4: "active", 5: "bench"}.get(o.get("area"))
        idx, ei = o.get("index"), o.get("energyIndex")
        if key is None or idx is None or ei is None:
            return None
        lst = cur["players"][pi].get(key) or []
        if not (0 <= idx < len(lst)) or not lst[idx]:
            return None
        ecs = lst[idx].get("energyCards") or []
        return ecs[ei].get("id") if 0 <= ei < len(ecs) else None
    except Exception:
        return None


def _target(o):
    a = o.get("inPlayArea")
    if a is None:
        return ""
    i = o.get("inPlayIndex")
    return f"@{vocab.area_name(a)}{i if i is not None else ''}"


def encode_option(o, obs=None):
    t = o.get("type")
    try:
        ot = OptionType(t)
    except Exception:
        return f"opt{t}#{o.get('index')}"

    if ot == OptionType.ATTACK:
        return f"attack:{o.get('attackId')}"
    if ot == OptionType.RETREAT:
        return "retreat"
    if ot == OptionType.END:
        return "end"
    if ot == OptionType.YES:
        return "yes"
    if ot == OptionType.NO:
        return "no"
    if ot == OptionType.NUMBER:
        return f"num:{o.get('number')}"
    if ot == OptionType.SKILL:
        # OptionType.SKILL (15) had no branch, so it rendered as the opaque "opt15#None"
        # (measured: 30 in a 60-game sweep) -- the model saw a blob instead of a move.
        # It carries cardId directly (e.g. {"type":15,"cardId":1260,"serial":108}).
        return f"skill:c{o.get('cardId')}{_target(o)}"
    if ot == OptionType.ABILITY:
        # _target() only fires on inPlayArea. When the owner is addressed by area/index
        # instead, two copies of the same body (e.g. bench [c506, c506, ...]) would BOTH
        # render "ability:c506" and the model could not say which one it means.
        loc = _target(o)
        if not loc and o.get("area") in (4, 5) and o.get("index") is not None:
            loc = f"@{vocab.area_name(o['area'])}{o['index']}"
        return f"ability:c{_ability_card_id(o, obs)}{loc}"
    if ot == OptionType.PLAY:
        return f"play:c{_hand_card_id(o, obs)}"
    if ot == OptionType.ATTACH:
        return f"attach:c{_hand_card_id(o, obs)}{_target(o)}"
    if ot == OptionType.EVOLVE:
        return f"evolve:c{_hand_card_id(o, obs)}{_target(o)}"
    if ot in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD):
        cid = _card_at(o, obs)
        if cid is None:
            # A face-DOWN card: prize piles are [null,...] for both seats, and `looking`
            # may hold None ("None if the card is facedown"). Unresolvable BY DESIGN --
            # the real game makes you pick a prize blind. Rendering it "card:cNone@PRIZE0"
            # (965 of these in a 60-game sweep) implies a lookup failure and gives the
            # model a token it can never ground; the slot index is the only real
            # information, so say exactly that.
            return f"facedown:{vocab.area_name(o.get('area'))}{o.get('index')}"
        return f"card:c{cid}@{vocab.area_name(o.get('area'))}{o.get('index')}"
    if ot == OptionType.ENERGY:
        # Two-step reference: area/index -> the Pokemon, energyIndex -> the energy CARD
        # attached to it. Emitting only the index ("energy:0") told the model nothing
        # about WHICH energy it was about to strip -- and 52% of these menus offer a real
        # choice. Name the card and where it sits.
        cid = _attached_energy_id(o, obs)
        where = f"@{vocab.area_name(o.get('area'))}{o.get('index')}" if o.get("area") is not None else ""
        return f"energy:c{cid}{where}#{o.get('energyIndex')}" if cid else f"energy:{o.get('energyIndex')}"
    if ot == OptionType.SPECIAL_CONDITION:
        return f"cond:{o.get('specialConditionType')}"
    return f"opt{t}#{o.get('index')}"


def decode_action(text, obs):
    """LM output -> list[int] option indices, or None if unparseable / illegal."""
    if text is None:
        return None
    sel = obs["select"]
    opts = sel["option"]
    enc = [encode_option(o, obs) for o in opts]

    picks = []
    for raw in str(text).replace(";", ",").replace("[", "").replace("]", "").split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok.lstrip("-").isdigit():          # (a) raw index
            j = int(tok)
            if 0 <= j < len(opts):
                picks.append(j)
                continue
            return None
        if tok in enc:                          # (b) semantic match (first hit)
            picks.append(enc.index(tok))
            continue
        return None                             # unknown token

    seen, uniq = set(), []
    for j in picks:
        if j in seen:
            return None
        seen.add(j)
        uniq.append(j)
    if not (sel["minCount"] <= len(uniq) <= sel["maxCount"]):
        return None
    return uniq
