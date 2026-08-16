#!/usr/bin/env python3
"""How often does each plan rule actually FIRE, and how often does it change the answer?

A rule that never triggers reads as "not measured", never as "not done" -- the silent-zero
failure that has cost this project three separate results. So before any A/B of a new rule,
count its opportunities. Reported per GAME and per OUR-TURN, because a card sits in hand across
every menu of a turn and per-menu rates have been misread four times.

    PYTHONPATH=cg-lib:tools DUSK_FRONT_DIVE=1 python3 tools/dusk_rule_fire.py \
        --rules lethal_now,...,front_dive --opp ogerpon_mono --games 60
"""
import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EPISODE_START = {"current": None, "logs": [], "remainingOverageTime": 600.0,
                  "search_begin_input": None, "select": None, "step": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--opp", default="ogerpon_mono")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import json
    import mirror_match as mm
    import dusk_plan
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

    rules = [r for r in a.rules.split(",") if r]
    fire = collections.Counter()      # rule -> menus where it fired
    narrow = collections.Counter()    # ... and actually removed an option
    turns = set()                     # (game, turn) seen, for a per-turn denominator
    fire_turn = collections.defaultdict(set)
    state = {"g": 0}

    # Wrap `opportunities` rather than the agent: the counter then sees exactly what
    # plan_filter sees, including the rules the arm does not name.
    orig = dusk_plan.opportunities

    def counted(obs, seat=None):
        live = orig(obs, seat)
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        cur = obs.get("current") or {}
        key = (state["g"], cur.get("turn"))
        if len(opts) >= 2:
            turns.add(key)
            for r in rules:
                hit = live.get(r)
                if not hit:
                    continue
                good = set(hit[0])
                if not good:
                    continue
                fire[r] += 1
                fire_turn[r].add(key)
                if len(good) < len(opts):
                    narrow[r] += 1
        return live
    dusk_plan.opportunities = counted

    eng = MirrorEngine(DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    my_ids, opp_ids = mm.load_deck(a.deck), mm.load_deck(a.opp)
    mm._FMT = "prompt"
    agent, _ = mm.make_agent("planfilter:%s:engine" % a.rules, a.deck, my_ids,
                             tuning.get(a.deck, {}))
    opp_agent, _ = mm.make_agent("engine", a.opp, opp_ids, tuning.get(a.opp, {}))

    wins = 0
    for g in range(a.games):
        state["g"] = g
        agent(dict(_EPISODE_START)), opp_agent(dict(_EPISODE_START))
        seed, mine = a.seed + g // 2, g % 2
        r = (play(eng, agent, opp_agent, my_ids, opp_ids, seed, mirror=1) if mine == 0
             else play(eng, opp_agent, agent, opp_ids, my_ids, seed, mirror=1))
        wins += int(r == mine)

    nt = max(1, len(turns))
    print("%s vs %s, %d games: %d wins (%.1f%%), %d turns with a real menu"
          % (a.deck, a.opp, a.games, wins, 100.0 * wins / a.games, len(turns)))
    print("%-16s %8s %8s %9s %9s" % ("rule", "menus", "narrowed", "turns", "turns/game"))
    for r in rules:
        print("%-16s %8d %8d %9d %9.2f"
              % (r, fire[r], narrow[r], len(fire_turn[r]), len(fire_turn[r]) / a.games))


if __name__ == "__main__":
    main()
