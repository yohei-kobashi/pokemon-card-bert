#!/usr/bin/env python3
"""Where this turn's energy goes, decided exactly instead of learned.

The same argument as `dusk_spread`: the answer is a lookup on types, the model is at 23.0%
(energy_line) and 20.5% (energy_focus) on it, and `attach-decisions-at-chance` already measured
this exact family of decision at 16-29% against a 14% chance line -- attaching is the single
worst thing this model does, and it is the thing the deck cannot function without.

The rule, from the deck's own costs:

    Phantom Dive is {R}{P}. Only c2 (basic {R}) and c5 (basic {P}) pay it, and they must land
    on the SAME body -- two half-charged bodies can neither of them attack. Drakloak's Dragon
    Headbutt costs the same {R}{P}, so energy put on a Drakloak works immediately and survives
    the evolution, which is why the plan attaches before evolving rather than after.

    c7 (basic {D}) pays NOTHING toward Phantom Dive. Its only job is switching on Munkidori's
    Adrena-Brain, so a {D} anywhere else is a wasted attachment for the turn.

Priority: complete {R}{P} on a body that then attacks > start the highest stage that can carry
it > {D} onto a Munkidori that has none. Anything else is declined and left to the model.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DREEPY, DRAKLOAK, PULT, MUNKIDORI = 119, 120, 121, 112
FIRE, PSY, DARK = 2, 5, 7
STAGE = {PULT: 3, DRAKLOAK: 2, DREEPY: 1}          # higher = closer to attacking


def _can_pd(types):
    e = list(types)
    for w in (FIRE, PSY):
        if w in e:
            e.remove(w)
        elif 0 in e:
            e.remove(0)
        else:
            return False
    return True


def _giving(obs, yi, o, text):
    """Which energy type this attach option would move."""
    cid = o.get("cardId") if isinstance(o, dict) else None
    if cid is None and text.startswith("attach:c"):
        try:
            cid = int(text.split("attach:c", 1)[1].split("@", 1)[0])
        except (ValueError, IndexError):
            cid = None
    if cid is None:
        cur = obs.get("current") or {}
        pl = cur.get("players") or []
        try:
            h = (pl[yi] or {}).get("hand") or []
            cid = (h[o.get("index")] or {}).get("id")
        except (IndexError, TypeError):
            cid = None
    return {2: FIRE, 5: PSY, 7: DARK}.get(cid)


def choose(obs, seat=None):
    """Index of the attach option to take, or None to leave the decision to the model."""
    from lm.actions import encode_option
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    pl = cur.get("players") or []
    if not opts or len(pl) < 2:
        return None
    yi = cur.get("yourIndex", 0) if seat is None else seat
    me = pl[yi] or {}

    best, best_key = None, None
    for i, o in enumerate(opts):
        if not isinstance(o, dict):
            continue
        try:
            t = encode_option(o, obs)
        except Exception:                                      # noqa: BLE001
            continue
        if not t.startswith("attach:"):
            continue
        giving = _giving(obs, yi, o, t)
        if giving is None:
            continue
        area = o.get("inPlayArea", o.get("area"))
        idx = o.get("inPlayIndex", o.get("index"))
        try:
            pk = ((me.get("active") or [None])[0] if area == 1
                  else (me.get("bench") or [])[idx])
        except (IndexError, TypeError):
            continue
        if not isinstance(pk, dict) or not pk.get("id"):
            continue
        tid = pk.get("id")
        have = list(pk.get("energies") or [])

        if giving == DARK:
            # Only Munkidori can spend it, and only one is needed to switch the ability on.
            key = (0, 0, 0) if (tid == MUNKIDORI and DARK not in have) else None
        elif tid in STAGE:
            if _can_pd(have + [giving]) and not _can_pd(have):
                key = (0, -STAGE[tid], 0)          # completes the cost: the attack turns on
            elif giving in (FIRE, PSY) and not any(x in (FIRE, PSY) for x in have):
                key = (1, -STAGE[tid], len(have))  # first useful energy, highest stage first
            else:
                key = None                          # a third energy, or a duplicate type
        else:
            key = None                              # {R}/{P} on a body that cannot use it
        if key is None:
            continue
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def is_attach_select(obs):
    """True when this decision includes attaching an energy from hand."""
    from lm.actions import encode_option
    sel = obs.get("select") or {}
    for o in (sel.get("option") or []):
        try:
            if encode_option(o, obs).startswith("attach:"):
                return True
        except Exception:                                      # noqa: BLE001
            continue
    return False
