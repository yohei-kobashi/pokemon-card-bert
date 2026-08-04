"""Live costs the observation does not carry.

The board renderer prints `rt:N` from the card table -- the card's PRINTED retreat cost -- and the
observation exposes no live one. Effects change it, so on decks built around such an effect the
prompt tells the model a legal, free retreat costs energy it does not have: measured 43% of the
offered retreats on ns_zoroark (N's Castle) and 40% on ethan_hooh (Latias ex), against 0% on decks
with no modifier. 24 of 63 decks run one. See [[prompt-lies-about-retreat-cost]].

The menu is the ORACLE this module is checked against: if `retreat` is on the menu then the live
cost is payable, so `effective_retreat_cost <= attached energy` must hold. tools/audit_costs.py
runs that check over whole games, which is why this can be trusted without hand-verifying cards.

What it cannot see, and does not pretend to:
  * ability nullification (Iron Thorns ex / Gastrodon / Team Rocket's Watchtower, 7 of 63 decks)
    turns the ability-based zeroing off; we would still report 0.
  * `c91 Rillaboom` raises the cost during the opponent's next turn -- a temporal effect no
    snapshot carries. NO deck in the pool runs it, which is the only reason this is acceptable.
  * "can't retreat" effects (31 cards) are NOT a cost question: the engine simply omits retreat
    from the menu, so the availability flag covers them for free.
"""

from lm import vocab

_STADIUM_NS_CASTLE = 1253       # N's Pokemon (both players) have no Retreat Cost
_ABILITY_ZERO = {
    184: "basic",               # Latias ex: your Basic Pokemon have no Retreat Cost
    170: "metal",               # Archaludon: your Pokemon with M energy have no Retreat Cost
}
_SELF_ZERO_IF_NO_ENERGY = {356, 788}        # Ethan's Magcargo, Charmander
_TOOL_AIR_BALLOON = 1174        # -2
_TOOL_RESCUE_BOARD = 1157       # -1, and 0 when remaining HP <= 30
_TOOL_GRAVITY_GEMSTONE = 1166   # +1 to BOTH Active Pokemon while its holder is Active
_METAL = 8


def _ids(seq):
    out = []
    for x in (seq or []):
        if isinstance(x, dict):
            if x.get("id") is not None:
                out.append(x["id"])
        elif isinstance(x, int):
            out.append(x)
    return out


def _is_ns(cid):
    n = getattr(vocab._CARDS.get(cid), "name", "") or ""
    return n.startswith("N's") or n.startswith("N’s")


def _in_play(cur, pi):
    p = (cur.get("players") or [])[pi]
    return _ids(p.get("active")) + _ids(p.get("bench"))


def effective_retreat_cost(obs, pi, mon):
    """Live retreat cost of `mon` (an in-play Pokemon dict belonging to player `pi`).

    Returns None when the card is unknown. Zeroing effects are ABSOLUTE ("has no Retreat Cost"),
    so they win over the additive tools rather than stacking with them; additive results floor
    at 0.
    """
    if not mon:
        return None
    cid = mon.get("id")
    card = vocab._CARDS.get(cid)
    if card is None or card.retreatCost is None:
        return None
    cur = obs.get("current") or {}
    tools = set(_ids(mon.get("tools")))
    energies = mon.get("energies") or []

    # --- absolute zeroing ---------------------------------------------------------------
    if _STADIUM_NS_CASTLE in set(_ids(cur.get("stadium"))) and _is_ns(cid):
        return 0
    if cid in _SELF_ZERO_IF_NO_ENERGY and not energies:
        return 0
    if _TOOL_RESCUE_BOARD in tools and (mon.get("hp") or 0) <= 30:
        return 0
    mine = set(_in_play(cur, pi))
    for holder, cond in _ABILITY_ZERO.items():
        if holder not in mine:
            continue
        if cond == "basic" and getattr(card, "basic", False):
            return 0
        if cond == "metal" and _METAL in energies:
            return 0

    # --- additive -----------------------------------------------------------------------
    cost = card.retreatCost
    if _TOOL_AIR_BALLOON in tools:
        cost -= 2
    if _TOOL_RESCUE_BOARD in tools:
        cost -= 1
    # Gravity Gemstone raises BOTH Actives, so it counts from either side's Active slot.
    for q in range(len(cur.get("players") or [])):
        act = ((cur.get("players") or [])[q].get("active") or [None])[0]
        if act and _TOOL_GRAVITY_GEMSTONE in set(_ids(act.get("tools"))):
            cost += 1
            break
    return max(0, cost)
