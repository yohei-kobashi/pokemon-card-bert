#!/usr/bin/env python3
"""What separates the decisions where the reference agents DO the play from the ones where
they do not?

`slowking_plan.py` measured the ladder's own agents against rules written from their action
DISTRIBUTION and they scored 7.9-32.9% -- so the rules are wrong, not the agents. Seek
Inspiration is 38% of their attacks and yet fires on 11.4% of the decisions where it is legal;
"the attack they use most" and "the attack they always use when they can" are different
statements, and only the first was in the data.

This tool does not assume the condition. For each trigger it records a small set of board
features and reports the take rate conditioned on each, so the split shows itself.

    PYTHONPATH=cg-lib python3 tools/slowking_conditions.py --limit 2500
"""

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = os.path.join(ROOT, "scratchpad_replays")
SLOWKING, KANGA, SMOOCHUM = 163, 756, 183
SEEK, PSYBOLT, KISS = 213, 214, 242
PAYLOADS = (115, 224, 183, 144)


def _slots(ps):
    return list(ps.get("active") or []) + list(ps.get("bench") or [])


def features(obs, yi):
    """Small, readable board facts -- each one a candidate condition."""
    from lm.actions import encode_option
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    pl = cur.get("players") or []
    me, opp = pl[yi] or {}, pl[1 - yi] or {}
    texts = []
    for o in opts:
        try:
            texts.append(encode_option(o, obs))
        except Exception:                                       # noqa: BLE001
            texts.append("")
    kinds = {t.split(":")[0] for t in texts if t}
    act = (me.get("active") or [None])[0]
    oact = (opp.get("active") or [None])[0]
    f = {}
    # the hypothesis carried over from dragapult_dusknoir: attacking ends the turn, so an
    # attack rule that fires on a menu still offering development is asking for the turn to be
    # thrown away
    f["menu"] = ("attack_only" if not (kinds - {"attack", "end", "retreat"})
                 else "has_development")
    f["turn"] = "t<=4" if (cur.get("turn") or 0) <= 4 else ("t<=10" if (cur.get("turn") or 0) <= 10 else "late")
    f["active"] = ({SLOWKING: "slowking", KANGA: "kanga", SMOOCHUM: "smoochum"}
                   .get((act or {}).get("id"), "other") if isinstance(act, dict) else "none")
    f["payload_in_hand"] = ("yes" if any(h.get("id") in PAYLOADS
                                         for h in (me.get("hand") or [])) else "no")
    n_e = sum(len(p.get("energies") or []) for p in _slots(me) if isinstance(p, dict))
    f["our_energy"] = "0-1" if n_e <= 1 else ("2-3" if n_e <= 3 else "4+")
    f["opp_hp"] = ("low<=120" if isinstance(oact, dict) and (oact.get("hp") or 0) <= 120
                   else "high")
    f["prizes"] = ("ahead" if len(opp.get("prize") or []) < len(me.get("prize") or [])
                   else ("even" if len(opp.get("prize") or []) == len(me.get("prize") or [])
                         else "behind"))
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--rule", default="seek", choices=("seek", "front_slowking", "kiss"))
    a = ap.parse_args()

    from lm.actions import encode_option
    tally = collections.defaultdict(lambda: [0, 0])       # (feature, value) -> [took, n]
    joint = collections.Counter()
    files = sorted(glob.glob(os.path.join(CACHE, "episode-*-replay.json")),
                   key=os.path.getmtime, reverse=True)[:a.limit]
    games = trig = 0
    for fn in files:
        try:
            d = json.load(open(fn))
        except Exception:                                       # noqa: BLE001
            continue
        steps = d.get("steps") or []
        seen = [set(), set()]
        for pair in steps:
            for mi, ag in enumerate(pair[:2]):
                o = (ag.get("observation") or {}).get("current") or {}
                pls = o.get("players") or []
                yi = o.get("yourIndex", 0)
                if yi < len(pls) and pls[yi]:
                    for s in _slots(pls[yi]):
                        if isinstance(s, dict) and s.get("id"):
                            seen[mi].add(s["id"])
                    for c in (pls[yi].get("hand") or []):
                        if isinstance(c, dict) and c.get("id"):
                            seen[mi].add(c["id"])
        who = [mi for mi in (0, 1) if SLOWKING in seen[mi]]
        if not who:
            continue
        mi = who[0]
        games += 1
        for pair in steps:
            if mi >= len(pair):
                continue
            ag = pair[mi]
            act, obs = ag.get("action"), ag.get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if not act or not isinstance(act, list) or not opts:
                continue
            cur = obs.get("current") or {}
            yi = cur.get("yourIndex", 0)
            want = None
            if a.rule == "seek":
                want = {i for i, o in enumerate(opts)
                        if isinstance(o, dict) and o.get("attackId") == SEEK}
            elif a.rule == "kiss":
                want = {i for i, o in enumerate(opts)
                        if isinstance(o, dict) and o.get("attackId") == KISS}
            else:
                pl = cur.get("players") or []
                me = pl[yi] or {}
                want = set()
                for i, o in enumerate(opts):
                    area = o.get("inPlayArea", o.get("area")) if isinstance(o, dict) else None
                    idx = o.get("inPlayIndex", o.get("index")) if isinstance(o, dict) else None
                    try:
                        pk = ((me.get("active") or [None])[0] if area == 1
                              else (me.get("bench") or [])[idx])
                    except (IndexError, TypeError):
                        continue
                    if isinstance(pk, dict) and pk.get("id") == SLOWKING:
                        want.add(i)
            if not want or len(want) == len(opts):
                continue
            trig += 1
            took = 1 if (want & set(act)) else 0
            f = features(obs, yi)
            for k, v in f.items():
                t = tally[(k, v)]
                t[0] += took
                t[1] += 1
            joint[(f["menu"], f["active"], took)] += 1

    print("rule=%s | %d games | %d triggers\n" % (a.rule, games, trig))
    print("%-16s %-16s %7s %8s" % ("feature", "value", "n", "take rate"))
    last = None
    for (k, v), (took, n) in sorted(tally.items()):
        if n < 10:
            continue
        if k != last:
            print()
            last = k
        print("%-16s %-16s %7d %7.1f%%" % (k, v, n, 100.0 * took / n))
    print("\njoint (menu, active) -> take rate:")
    keys = {(m, ac) for (m, ac, _t) in joint}
    for m, ac in sorted(keys):
        yes, no = joint[(m, ac, 1)], joint[(m, ac, 0)]
        if yes + no >= 10:
            print("   %-16s %-10s n=%4d  %5.1f%%" % (m, ac, yes + no, 100.0 * yes / (yes + no)))


if __name__ == "__main__":
    main()
