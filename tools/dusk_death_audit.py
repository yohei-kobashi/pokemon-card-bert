#!/usr/bin/env python3
"""WHO dies against ogerpon_mono, carrying WHAT -- the 1.27 energy-with-body attribution, itemised.

The hammer audit split the line-energy loss 1.27 body-death vs 0.42 hammer, which redirects the
question from "stop the hammers" to "stop feeding charged bodies". This names the feed: for
every line body of ours that leaves play, record its stage, the {R}/{P} it carried, and whether
it was our ACTIVE when it happened (Myriad Leaf Shower one-shots the front; a body that dies
from the BENCH was gusted out -- two different remedies).

Also the tempo side: the first turn a single body of ours has {R}{P} banked, the first turn a
Dragapult ex exists, and the first dive -- the three arrows whose gaps are the matchup.
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EPISODE_START = {"current": None, "logs": [], "remainingOverageTime": 600.0,
                  "search_begin_input": None, "select": None, "step": 1}
PHANTOM_DIVE = 154
DREEPY, DRAKLOAK, PULT = 119, 120, 121
DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
BUDEW, MUNKIDORI, FEZ, MEOWTH = 235, 112, 140, 1071
FIRE, PSY = 2, 5
NAME = {119: "Dreepy", 120: "Drakloak", 121: "Dragapult", 131: "Duskull", 132: "Dusclops",
        133: "Dusknoir", 235: "Budew", 112: "Munkidori", 140: "Fez", 1071: "Meowth"}


def _slots(ps):
    a = list((ps or {}).get("active") or [])
    b = list((ps or {}).get("bench") or [])
    return [(p, i == 0) for i, p in enumerate(a) if isinstance(p, dict)] + \
           [(p, False) for p in b if isinstance(p, dict)]


def _rp(p):
    return sum(1 for e in (p.get("energies") or []) if e in (FIRE, PSY))


def _can_pd(p):
    e = [x for x in (p.get("energies") or [])]
    for w in (FIRE, PSY):
        if w in e:
            e.remove(w)
        elif 0 in e:
            e.remove(0)
        else:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--opp", default="ogerpon_mono")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fmt", default="prompt", choices=("prompt", "dusk"))
    a = ap.parse_args()

    import mirror_match as mm
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

    eng = MirrorEngine(DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    my_ids, opp_ids = mm.load_deck(a.deck), mm.load_deck(a.opp)
    mm._FMT = a.fmt
    agent, _ = mm.make_agent(a.spec, a.deck, my_ids, tuning.get(a.deck, {}))
    opp_agent, _ = mm.make_agent("engine", a.opp, opp_ids, tuning.get(a.opp, {}))

    deaths = collections.Counter()     # (name, energy_carried, was_active) -> count
    firsts = collections.defaultdict(list)   # metric -> [turn, ...] one per game where it happened
    C = collections.Counter()
    g = {}

    def reset():
        g.update(turn=None, prev=None, prev_prizes=None, ourturn=0,
                 saw_payable=None, saw_pult=None, saw_dive=None)

    def watched(obs):
        cur = obs.get("current") or {}
        sel = obs.get("select") or {}
        yi = cur.get("yourIndex", 0)
        pl = cur.get("players") or []
        me = (pl[yi] or {}) if yi < len(pl) else {}
        turn = cur.get("turn")

        if turn != g["turn"]:
            if g["turn"] is not None:
                g["ourturn"] += 1
            now = _slots(me)
            # deaths since our last decision: previous snapshot bodies that no longer exist.
            # Bodies are matched greedily by (id, energy) -- identical twins can swap, which is
            # fine: what is counted is the multiset difference, not identity.
            if g["prev"] is not None:
                prev_prizes = g["prev_prizes"]
                mine_now = len(me.get("prize") or [])
                lost_prizes = (mine_now < prev_prizes) if prev_prizes is not None else False
                remaining = [(p.get("id"), _rp(p)) for p, _ in now]
                for p, was_active in g["prev"]:
                    key = (p.get("id"), _rp(p))
                    if key in remaining:
                        remaining.remove(key)
                    elif p.get("id") in (DRAKLOAK, PULT, DUSCLOPS, DUSKNOIR) \
                            and (p.get("id") + 1, _rp(p)) in remaining:
                        remaining.remove((p.get("id") + 1, _rp(p)))   # it evolved, not died
                    elif lost_prizes or p.get("id") not in (DREEPY, DRAKLOAK, PULT):
                        deaths[(NAME.get(p.get("id"), p.get("id")), _rp(p), was_active)] += 1
                        if p.get("id") in (DREEPY, DRAKLOAK, PULT):
                            C["line_deaths"] += 1
                            C["line_energy_buried"] += _rp(p)
            g["prev"] = now
            g["prev_prizes"] = len(me.get("prize") or [])
            g["turn"] = turn
            # tempo firsts
            if g["saw_payable"] is None and any(_can_pd(p) for p, _ in now
                                                if p.get("id") in (DREEPY, DRAKLOAK, PULT)):
                g["saw_payable"] = g["ourturn"]
            if g["saw_pult"] is None and any(p.get("id") == PULT for p, _ in now):
                g["saw_pult"] = g["ourturn"]

        pick = agent(obs)
        opts = sel.get("option") or []
        for i in (pick or []):
            if 0 <= i < len(opts) and isinstance(opts[i], dict) \
                    and opts[i].get("attackId") == PHANTOM_DIVE and g["saw_dive"] is None:
                g["saw_dive"] = g["ourturn"]
        return pick

    wins = 0
    for i in range(a.games):
        reset()
        watched(dict(_EPISODE_START))
        opp_agent(dict(_EPISODE_START))
        seed, mine = a.seed + i // 2, i % 2
        r = (play(eng, watched, opp_agent, my_ids, opp_ids, seed, mirror=1) if mine == 0
             else play(eng, opp_agent, watched, opp_ids, my_ids, seed, mirror=1))
        wins += int(r == mine)
        for k in ("saw_payable", "saw_pult", "saw_dive"):
            if g[k] is not None:
                firsts[k].append(g[k])

    n = float(a.games)
    print("=== %s vs %s, %d games: %d wins (%.1f%%) ===" % (a.deck, a.opp, a.games, wins,
                                                            100.0 * wins / n))
    print("  spec %s" % a.spec)
    for k, lbl in (("saw_payable", "first {R}{P} banked on one line body"),
                   ("saw_pult", "first Dragapult ex in play"),
                   ("saw_dive", "first Phantom Dive")):
        v = firsts[k]
        print("  %-38s our turn %5.2f mean, in %d/%d games"
              % (lbl, (sum(v) / len(v)) if v else -1, len(v), a.games))
    print("  line deaths/game %.2f, line {R}/{P} buried with them/game %.2f"
          % (C["line_deaths"] / n, C["line_energy_buried"] / n))
    print("  deaths by (body, energy carried, was our ACTIVE):")
    for (name, e, act), cnt in sorted(deaths.items(), key=lambda kv: -kv[1])[:14]:
        print("      %-10s energy=%d  %-6s %5d   (%.2f/game)"
              % (name, e, "ACTIVE" if act else "bench", cnt, cnt / n))


if __name__ == "__main__":
    main()
