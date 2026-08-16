#!/usr/bin/env python3
"""Does the item lock actually stop the hammers, and are we using it?

WHY. ogerpon_mono runs 4 Crushing Hammer -- "Flip a coin. If heads, discard an Energy from 1 of
your opponent's Pokemon" -- and it is the one matchup where our energy arrow collapses: the
forensics measured our line energy FALLING 1.58 times a game there against 0.50-1.08 everywhere
else, and "Dragapult ex in play -> can pay {R}{P}" dropping 20 points instead of 7-9.

Crushing Hammer is an ITEM. Budew's Itchy Pollen reads "During your opponent's next turn, they
can't play any Item cards from their hand" -- so one Budew attack turns off every hammer in
their deck for a turn, at zero energy, from a body with free retreat. That is the format's
recognised answer (Pokemon.com's own Budew article describes exactly this role: attack "for a
few turns while the primary attackers get set up", then use the free retreat to pivot away).

So this measures the three things that decide whether the lock is the lever it looks like:

  1. do WE use it            -- Itchy Pollen turns per game
  2. does it WORK            -- hammers played on a locked turn should be exactly 0. If it is
                                not 0 the engine does not implement the lock and the whole idea
                                is dead; this is the falsification test and it runs first.
  3. is it WORTH it          -- hammers landed and line energy lost, per game, per arm

Counting is from OUR decision points only, which is all a pilot can see. The opponent's discard
pile is visible, so hammers PLAYED are counted by watching copies of Crushing Hammer accumulate
in it between our turns -- not by assuming.

    PYTHONPATH=cg-lib:tools python3 tools/dusk_hammer_audit.py --games 200 \
        --spec 'planfilter:lethal_now,...,search_bottom,setup_search:engine'
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
HAMMER = 1120
ITCHY_POLLEN = 323
BUDEW = 235
DREEPY, DRAKLOAK, PULT = 119, 120, 121
FIRE, PSY = 2, 5


def _discard(ps):
    for k in ("discard", "discardPile", "trash"):
        v = (ps or {}).get(k)
        if isinstance(v, list):
            return v
    return []


def _line_energy(me):
    n = 0
    for p in list((me or {}).get("active") or []) + list((me or {}).get("bench") or []):
        if isinstance(p, dict) and p.get("id") in (DREEPY, DRAKLOAK, PULT):
            n += sum(1 for e in (p.get("energies") or []) if e in (FIRE, PSY))
    return n


def _line_bodies(me):
    return sum(1 for p in list((me or {}).get("active") or []) + list((me or {}).get("bench") or [])
               if isinstance(p, dict) and p.get("id") in (DREEPY, DRAKLOAK, PULT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--opp", default="ogerpon_mono")
    ap.add_argument("--games", type=int, default=200)
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

    C = collections.Counter()
    g = {"hammers": 0, "locked_next": False, "turn": None, "energy": None,
         "bodies": None, "budew_turns": set()}

    def reset_game():
        g.update(hammers=0, locked_next=False, turn=None, energy=None, bodies=None)
        g["budew_turns"] = set()

    def watched(obs):
        cur = obs.get("current") or {}
        sel = obs.get("select") or {}
        pl = cur.get("players") or []
        yi = cur.get("yourIndex", 0)
        pick = agent(obs)
        if not pl or yi >= len(pl):
            return pick
        me, opp = pl[yi] or {}, pl[1 - yi] or {}
        turn = cur.get("turn")
        if turn != g["turn"]:
            # a NEW turn of ours: settle what happened since the last one
            hm = sum(1 for c in _discard(opp) if (c or {}).get("id") == HAMMER)
            if g["turn"] is not None:
                played = hm - g["hammers"]
                if played > 0:
                    C["hammers_played"] += played
                    if g["locked_next"]:
                        # THE FALSIFICATION TEST: an item played on a locked turn
                        C["hammers_while_locked"] += played
                e_now = _line_energy(me)
                if g["energy"] is not None and e_now < g["energy"]:
                    lost = g["energy"] - e_now
                    C["line_energy_lost"] += lost
                    # ATTRIBUTION: a hammer discards energy WITHOUT removing the body; a
                    # knockout (or retreat) removes body and energy together. Splitting the
                    # loss by whether a line body vanished in the same window separates
                    # "hammered" from "the charged body died", which need different remedies.
                    if _line_bodies(me) < (g["bodies"] if g["bodies"] is not None else 0):
                        C["lost_with_body"] += lost
                    elif played > 0:
                        C["lost_to_hammer"] += lost
                    else:
                        C["lost_other"] += lost
            g["hammers"] = hm
            g["turn"] = turn
            g["locked_next"] = False
            C["our_turns"] += 1
            act = (me.get("active") or [None])[0]
            if isinstance(act, dict) and act.get("id") == BUDEW:
                C["budew_active_turns"] += 1
        g["energy"] = _line_energy(me)
        g["bodies"] = _line_bodies(me)

        opts = sel.get("option") or []
        for i in (pick or []):
            if 0 <= i < len(opts) and isinstance(opts[i], dict):
                if opts[i].get("attackId") == ITCHY_POLLEN:
                    C["itchy_used"] += 1
                    g["locked_next"] = True
        if any(isinstance(o, dict) and o.get("attackId") == ITCHY_POLLEN for o in opts):
            C["itchy_offered_menus"] += 1
        return pick

    wins = 0
    for i in range(a.games):
        reset_game()
        watched(dict(_EPISODE_START))
        opp_agent(dict(_EPISODE_START))
        seed, mine = a.seed + i // 2, i % 2
        r = (play(eng, watched, opp_agent, my_ids, opp_ids, seed, mirror=1) if mine == 0
             else play(eng, opp_agent, watched, opp_ids, my_ids, seed, mirror=1))
        wins += int(r == mine)

    n = float(a.games)
    print("=== %s vs %s, %d games: %d wins (%.1f%%) ===" % (a.deck, a.opp, a.games, wins,
                                                            100.0 * wins / n))
    print("  spec %s" % a.spec)
    print("  our turns/game            %6.2f" % (C["our_turns"] / n))
    print("  Budew ACTIVE turns/game   %6.2f" % (C["budew_active_turns"] / n))
    print("  Itchy Pollen used/game    %6.2f   (offered on %.2f menus/game)"
          % (C["itchy_used"] / n, C["itchy_offered_menus"] / n))
    print("  hammers PLAYED/game       %6.2f" % (C["hammers_played"] / n))
    print("  ... on a LOCKED turn      %6.2f   <- must be 0.00 if the lock works"
          % (C["hammers_while_locked"] / n))
    print("  line {R}/{P} lost/game    %6.2f" % (C["line_energy_lost"] / n))
    print("    with the body (KO)      %6.2f" % (C["lost_with_body"] / n))
    print("    to a hammer             %6.2f" % (C["lost_to_hammer"] / n))
    print("    other (retreat cost...) %6.2f" % (C["lost_other"] / n))


if __name__ == "__main__":
    main()
