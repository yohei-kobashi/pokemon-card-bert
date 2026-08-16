#!/usr/bin/env python3
"""Phantom Dive's six damage counters, allocated exactly instead of learned.

WHY THIS IS NOT LEFT TO THE MODEL. The engine does not ask for the allocation; it asks for one
target at a time with a counter budget remaining:

    SEL DAMAGE_COUNTER_ANY dmg:6 n1-1 :: 0=card:c121@BENCH0 1=card:c121@BENCH1 ...

A cross-encoder scores each candidate independently, so "three counters here and three there"
is not something it can express -- the joint plan is outside its action space, and training
`spread_aim` was asking it to learn a decision it cannot represent. It sits at 48.2%, which is
about what choosing between two or three targets at random gives.

The decision is also arithmetic, which is the one thing this model is measurably worst at:
`attach-decisions-at-chance` measured attach top1 at 16-29% against a 14% chance line, and
`prompt-lies-about-retreat-cost` found it could not even read affordability off the prompt.

The allocation itself is small enough to solve exactly: at most five bodies, at most six
counters, and a clear objective from the deck's plan --

    1. take a prize NOW if a body is already within the remaining counters
    2. otherwise bring the most valuable body INTO Phantom Dive range (hp <= 200), preferring
       an ex (two prizes) over a non-ex, and preferring the cheapest such conversion
    3. never spend counters on a body that cannot be finished with what is left; the plan is
       to bank damage that CONVERTS, not to spread it thin

This is not engine_v2 and does not import it. It is deck-specific inference-time arithmetic,
and it is barred from nothing: the constraint was that engine_v2 must not appear in TRAINING.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PD_RANGE = 200          # Phantom Dive's damage to the Active


def _prize_value(card):
    """How many prizes this body is worth when knocked out."""
    if card is None:
        return 1
    if getattr(card, "megaEx", False):
        return 3
    return 2 if getattr(card, "ex", False) else 1


def choose(obs, seat=None):
    """Index of the option this counter should go on, or None to leave it to the caller.

    Called once per counter; the engine re-asks with the budget decremented, so a greedy
    choice per call composes into the allocation. Ties are broken toward the cheapest
    conversion so the remaining counters stay useful.
    """
    from agents._engine import _CARDS
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    pl = cur.get("players") or []
    if not opts or len(pl) < 2:
        return None
    yi = cur.get("yourIndex", 0) if seat is None else seat
    opp = pl[1 - yi] or {}
    remain = int(sel.get("remainDamageCounter") or 0)
    if remain <= 0:
        return None

    best, best_key = None, None
    for i, o in enumerate(opts):
        if not isinstance(o, dict) or o.get("playerIndex") in (None, yi):
            continue
        area = o.get("inPlayArea", o.get("area"))
        idx = o.get("inPlayIndex", o.get("index"))
        try:
            pk = ((opp.get("active") or [None])[0] if area == 1
                  else (opp.get("bench") or [])[idx])
        except (IndexError, TypeError):
            continue
        if not isinstance(pk, dict) or not pk.get("id"):
            continue
        hp = pk.get("hp") or 0
        val = _prize_value(_CARDS.get(pk.get("id")))
        if hp <= remain * 10:
            # (1) a prize right now: most prizes first, then the cheapest kill
            key = (0, -val, hp)
        else:
            need = hp - PD_RANGE
            if 0 < need <= remain * 10:
                # (2) convertible into a Phantom Dive KO with the counters in hand
                key = (1, -val, need)
            else:
                # (3) cannot be finished -- spending here banks damage that never converts
                key = (2, -val, need)
        if best_key is None or key < best_key:
            best, best_key = i, key
    # Refuse the decision only when nothing is reachable at all; then the caller (the model)
    # keeps it, which is the honest fallback rather than a silent arbitrary pick.
    if best is not None and best_key is not None and best_key[0] <= 1:
        return best
    return None


def is_spread_select(obs):
    """True when this decision is a Phantom Dive counter placement."""
    sel = obs.get("select") or {}
    return (sel.get("context") in (13, 14)          # DAMAGE_COUNTER / DAMAGE_COUNTER_ANY
            and int(sel.get("remainDamageCounter") or 0) > 0)
