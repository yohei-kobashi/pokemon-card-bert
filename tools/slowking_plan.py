#!/usr/bin/env python3
"""The slowking game plan as checkable rules -- WITH A GROUND TRUTH the dusknoir spec lacked.

The #1 and #2 ladder teams play this deck, and their replays are on disk. So every rule here
can be validated the way a rule should be: run it against THEIR games. If the rule is right
they score high on it; if they score low, the rule is wrong and our pilot must not be trained
or written toward it. For dragapult_dusknoir no such reference existed and six rules had to be
corrected by inspection alone.

The rules come from what those 49 games (2,977 decisions) actually show, not from the card
text read in isolation:

    their attacks   Seek Inspiration 36 | Delightful Kiss 28 | Destined Fight 8 | Trifrost 7
                    Gutsy Swing 4 | Super Psy Bolt 2 | **Rapid-Fire Combo 1**
    their Active    Slowking 38.4% | Mega Kangaskhan 35.7% | Smoochum 13.2%

Mega Kangaskhan ex holds the Active Spot for a third of all decisions and attacks ONCE in 49
games: it is a 300 HP wall running Run Errand, and it is kept empty of energy on purpose.
Smoochum's Delightful Kiss costs nothing and fetches two basic {P} onto the Bench -- 30% of
their attacks -- which is how an 8-10 energy list functions at all.

    PYTHONPATH=cg-lib python3 tools/slowking_plan.py --replays        # the reference agents
    PYTHONPATH=cg-lib python3 tools/slowking_plan.py --traces X.gz    # our pilot
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = os.path.join(ROOT, "scratchpad_replays")

SLOWKING, SLOWPOKE, KANGA = 163, 162, 756
SMOOCHUM, CONKELDURR, ANNIHILAPE, KYUREM = 183, 115, 224, 144
LATIAS, MEOWTH, FEZ = 184, 1071, 140
ACADEMY, CIPHER = 1248, 1188
SEEK, PSYBOLT, KISS = 213, 214, 242
PSY = 5
PAYLOADS = (CONKELDURR, ANNIHILAPE, SMOOCHUM, KYUREM)

RULES = {
    "front_slowking": ("stand a CHARGED Slowking in the Active Spot", 2.0),
    "seek":           ("attack with Seek Inspiration, not Super Psy Bolt", 2.0),
    "kiss_for_energy": ("attack with Delightful Kiss while energy is short", 1.5),
    "kanga_no_energy": ("do NOT put energy on Mega Kangaskhan -- it is the wall", 1.5),
    "energy_slowking": ("energy goes on Slowking (it pays {P}{C} for Seek)", 1.5),
    "run_errand":     ("use Run Errand every turn Kangaskhan is Active", 1.0),
    "stack_payload":  ("put a Seek payload on top (Academy / Ciphermaniac's)", 1.5),
    "evolve_slowking": ("evolve Slowpoke into Slowking", 1.5),
}


def _slots(ps):
    return list(ps.get("active") or []) + list(ps.get("bench") or [])


def _e(pk):
    return list((pk or {}).get("energies") or [])


def opportunities(obs, seat=None):
    """{rule: (conformant indices, SCOPE indices)} -- same contract as dusk_plan."""
    from lm.actions import encode_option
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    pl = cur.get("players") or []
    if not opts or len(pl) < 2:
        return {}
    yi = cur.get("yourIndex", 0) if seat is None else seat
    if yi >= len(pl):
        return {}
    me = pl[yi] or {}
    mine = _slots(me)
    ids = [p.get("id") for p in mine if isinstance(p, dict)]
    out = {}

    def texts():
        for i, o in enumerate(opts):
            try:
                yield i, o, encode_option(o, obs)
            except Exception:                                   # noqa: BLE001
                yield i, o, ""
    T = list(texts())

    def _pk_of(o):
        area = o.get("inPlayArea", o.get("area"))
        idx = o.get("inPlayIndex", o.get("index"))
        try:
            return ((me.get("active") or [None])[0] if area == 1
                    else (me.get("bench") or [])[idx])
        except (IndexError, TypeError):
            return None

    # --- attacks: WHICH attack, not whether to attack ----------------------------------------
    # The first version scored "Seek was legal and not chosen" as a miss, and the reference
    # agents came out at 11.4% on their own signature play. The reason is arithmetic, not
    # judgement: a turn offers the attack in the menu at every decision but ends after ONE of
    # them, so a ten-decision turn produces nine "misses" by construction. 94% of the triggers
    # had development still available. The question a rule can actually ask is: GIVEN that the
    # turn is being ended with an attack, which attack. Scope is the attacks alone, and `end`
    # and `retreat` are deliberately NOT in it -- "attack or keep building" is a different
    # question that this data cannot answer.
    atk = {i: o for i, o in enumerate(opts) if isinstance(o, dict) and o.get("attackId")}
    if len(atk) > 1:
        scope = set(atk)
        seek = {i for i, o in atk.items() if o["attackId"] == SEEK}
        if seek and len(scope) > len(seek):
            out["seek"] = (seek, scope)
        kiss = {i for i, o in atk.items() if o["attackId"] == KISS}
        if kiss and sum(len(_e(p)) for p in mine if isinstance(p, dict)) < 3 \
                and len(scope) > len(kiss):
            out["kiss_for_energy"] = (kiss, scope)

    # --- energy: onto Slowking, never onto the wall ------------------------------------------
    att = [(i, o, t) for i, o, t in T if t.startswith("attach")]
    if att:
        scope = {i for i, _o, _t in att}
        good, kanga = set(), set()
        for i, o, _t in att:
            pk = _pk_of(o)
            if not isinstance(pk, dict):
                continue
            if pk.get("id") == SLOWKING:
                good.add(i)
            elif pk.get("id") == KANGA:
                kanga.add(i)
        if good and len(scope) > len(good):
            out["energy_slowking"] = (good, scope)
        # Keeping the wall empty is a rule in its own right: it is WHY Kangaskhan never
        # attacks in their games, and it is expressed by what they decline to do.
        if kanga and len(kanga) < len(scope):
            out["kanga_no_energy"] = (scope - kanga, scope)

    # --- abilities: Run Errand while Active, and the placers -----------------------------
    abl = [(i, o, t) for i, o, t in T if t.startswith("abl") or t.startswith("ability")]
    if abl:
        scope = {i for i, _o, _t in abl}
        a = me.get("active") or [None]
        if isinstance(a[0], dict) and a[0].get("id") == KANGA:
            r = {i for i, o, _t in abl if (_pk_of(o) or {}).get("id") == KANGA}
            if r and len(scope) > len(r):
                out["run_errand"] = (r, scope)
        acad = {i for i, o, _t in abl if (_pk_of(o) or {}).get("id") == ACADEMY}
        if acad and any(h.get("id") in PAYLOADS for h in (me.get("hand") or [])):
            out["stack_payload"] = (acad, scope | acad)

    # --- put a PAYLOAD on top, not our worst card --------------------------------------------
    if sel.get("context") in (9, 10):        # TO_DECK / TO_DECK_BOTTOM
        pay = set()
        for i, o, _t in T:
            cid = o.get("cardId") if isinstance(o, dict) else None
            if cid is None:
                try:
                    cid = ((me.get("hand") or [])[o.get("index")] or {}).get("id")
                except (IndexError, TypeError):
                    cid = None
            if cid in PAYLOADS:
                pay.add(i)
        if pay and len(pay) < len(opts):
            out["stack_payload"] = (pay, set(range(len(opts))))

    # --- promoting / evolving ----------------------------------------------------------------
    ev = [(i, o, t) for i, o, t in T if t.startswith("evolve")]
    if ev:
        scope = {i for i, _o, _t in ev}
        e = set()
        for i, o, _t in ev:
            cid = o.get("cardId")
            if cid is None:
                try:
                    cid = ((me.get("hand") or [])[o.get("index")] or {}).get("id")
                except (IndexError, TypeError):
                    cid = None
            if cid == SLOWKING:
                e.add(i)
        if e and len(scope) > len(e):
            out["evolve_slowking"] = (e, scope)

    # --- who stands in the Active Spot -------------------------------------------------------
    if sel.get("context") in (1, 2, 3) and len(opts) > 1:
        cand = {}
        for i, o, _t in T:
            pk = _pk_of(o)
            if isinstance(pk, dict) and pk.get("id"):
                cand[i] = pk
        charged = {i for i, pk in cand.items()
                   if pk.get("id") == SLOWKING and len(_e(pk)) >= 2}
        if charged and len(charged) < len(cand):
            out["front_slowking"] = (charged, set(cand))
    return out


def score(obs, picks, seat=None):
    live = opportunities(obs, seat)
    chosen = set(picks if isinstance(picks, (list, tuple)) else [picks])
    return ({r: (1 if (c & chosen) else 0) for r, (c, _s) in live.items()},
            {r: 1 for r in live})


def _from_replays(limit, team=""):
    """Measure the REFERENCE agents: the teams that hold #1 and #2 with this deck."""
    hit, n = collections.Counter(), collections.Counter()
    files = sorted(glob.glob(os.path.join(CACHE, "episode-*-replay.json")),
                   key=os.path.getmtime, reverse=True)[:limit]
    games = 0
    for fn in files:
        try:
            d = json.load(open(fn))
        except Exception:                                       # noqa: BLE001
            continue
        names = ((d.get("info") or {}).get("TeamNames")) or []
        steps = d.get("steps") or []
        seen = [set(), set()]
        for pair in steps:
            for mi, ag in enumerate(pair[:2]):
                o = (ag.get("observation") or {}).get("current") or {}
                pls = o.get("players") or []
                yi = o.get("yourIndex", 0)
                if yi < len(pls) and pls[yi]:
                    ps = pls[yi]
                    for z in ("hand", "discard", "prize"):
                        for c in (ps.get(z) or []):
                            if isinstance(c, dict) and c.get("id"):
                                seen[mi].add(c["id"])
                    for s in _slots(ps):
                        if isinstance(s, dict) and s.get("id"):
                            seen[mi].add(s["id"])
        who = [mi for mi in (0, 1) if SLOWKING in seen[mi]]
        if team:
            who = [mi for mi in who if mi < len(names) and names[mi] == team]
        if not who:
            continue
        mi = who[0]
        games += 1
        for pair in steps:
            if mi >= len(pair):
                continue
            ag = pair[mi]
            act, obs = ag.get("action"), ag.get("observation") or {}
            if not act or not isinstance(act, list) or not (obs.get("select") or {}).get("option"):
                continue
            h, k = score(obs, act, (obs.get("current") or {}).get("yourIndex", 0))
            hit.update(h)
            n.update(k)
    return hit, n, games


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", action="store_true", help="measure the ladder's own agents")
    ap.add_argument("--team", default="")
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--traces", default="", help="measure OUR pilot from trace files")
    ap.add_argument("--mirror-so", default="")
    a = ap.parse_args()

    if a.replays:
        hit, n, games = _from_replays(a.limit, a.team)
        label = "REFERENCE AGENTS (%d games)" % games
    else:
        import library
        from mirror_env import DEFAULT_SO, MirrorEngine
        eng = MirrorEngine(a.mirror_so or DEFAULT_SO)
        ids = {}

        def d_ids(x):
            if x not in ids:
                ids[x] = [int(v) for v in open(library.deck_path(x)) if v.strip()]
            return ids[x]
        hit, n = collections.Counter(), collections.Counter()
        games = 0
        for path in [p for p in a.traces.split(",") if p]:
            for line in gzip.open(path, "rt"):
                d = json.loads(line)
                if d.get("header"):
                    continue
                d0 = d.get("deck0") or d.get("deck")
                d1 = d.get("deck1") or d.get("deck")
                if "slowking" not in (d0, d1):
                    continue
                seat = 0 if d0 == "slowking" else 1
                obs = eng.start(d_ids(d0), d_ids(d1), d["seed"], mirror=1)
                games += 1
                try:
                    for pick in d["picks"]:
                        if obs is None:
                            break
                        cur = obs.get("current") or {}
                        if cur.get("result", -1) != -1 or obs.get("select") is None:
                            break
                        if cur.get("yourIndex") == seat and pick is not None:
                            h, k = score(obs, pick, seat)
                            hit.update(h)
                            n.update(k)
                        if pick is None:
                            break
                        obs = eng.select(pick)
                finally:
                    eng.finish()
        label = "OUR PILOT (%d games)" % games

    print("%s\n" % label)
    print("%-16s %8s %9s %9s  %s" % ("rule", "taken", "chances", "rate", "what it is"))
    for r, (name, w) in sorted(RULES.items(), key=lambda kv: -kv[1][1]):
        k = n.get(r, 0)
        if not k:
            print("%-16s %8s %9d %9s  %s" % (r, "-", 0, "NO TRIGGER", name))
        else:
            print("%-16s %8d %9d %8.1f%%  %s" % (r, hit.get(r, 0), k, 100.0 * hit[r] / k, name))


if __name__ == "__main__":
    main()
