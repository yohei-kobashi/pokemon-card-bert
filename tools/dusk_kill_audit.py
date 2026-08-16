#!/usr/bin/env python3
"""Why don't the dives convert into prizes against ogerpon_mono?

THE ARITHMETIC PROBLEM. Phantom Dive does 200 into a 210 HP body, so every dive leaves the
Active at exactly 10 unless something adds a counter -- and the deck carries three closers for
precisely this: Munkidori's Adrena-Brain (move up to 3 of our counters across, needs {D} on
Munkidori), Dusclops' Cursed Blast (5 counters, costs the body), and the six bench counters of
a PREVIOUS dive (a benched Ogerpon that already carries 10+ dies to the next 200). The human
line "3 Phantom Dives + 1 Cursed Bomb usually ends the game" is this arithmetic.

So the question is not "do we dive" (the forensics closed that: promoted = dives) but what
happens AFTER each dive:

    dive -> did it KO?
         -> if not, how much was left, and did we have a closer available?
         -> by our next turn, was the wounded body still there, healed, capped, or hidden?

Everything is measured from OUR decision points. Their side is read from the observation
(active id / hp / maxHp are visible), so "healed" and "grew maxHp" (Hero's Cape) are facts,
not guesses.

    PYTHONPATH=cg-lib:tools python3 tools/dusk_kill_audit.py --games 200 \
        --spec 'planfilter:...:engine'
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
MUNKIDORI, DUSCLOPS, DUSKNOIR = 112, 132, 133
DREEPY, DRAKLOAK, PULT = 119, 120, 121
DARK = 7


def _side(cur, i):
    pl = cur.get("players") or []
    return (pl[i] or {}) if i < len(pl) else {}


def _slots(ps):
    return list((ps or {}).get("active") or []) + list((ps or {}).get("bench") or [])


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
    H = collections.Counter()          # remaining-HP histogram after a non-KO dive
    g = {}

    def reset():
        g.update(turn=None, dived=False, tgt=None, my_prizes=None, opp_prizes=None,
                 prizes_when=collections.Counter(), our_turn_no=0,
                 first_prize_turn=None, my_active_e=[])

    def watched(obs):
        cur = obs.get("current") or {}
        sel = obs.get("select") or {}
        yi = cur.get("yourIndex", 0)
        me, opp = _side(cur, yi), _side(cur, 1 - yi)
        turn = cur.get("turn")

        if turn != g["turn"] and g["turn"] is not None:
            g["our_turn_no"] += 1
            # settle the PREVIOUS dive: what is their board like one full rotation later?
            if g["dived"]:
                C["dives"] += 1
                opp_now = {}
                for p in _slots(opp):
                    if isinstance(p, dict):
                        opp_now.setdefault(p.get("id"), []).append(p)
                op_prizes_now = len(opp.get("prize") or [])
                took = (g["opp_prizes"] - op_prizes_now) if g["opp_prizes"] is not None else 0
                # crude but honest: their prize count is OUR takings mirror -- we read our own
                mine_now = len(me.get("prize") or [])
                took = (g["my_prizes"] - mine_now) if g["my_prizes"] is not None else 0
                if took > 0:
                    C["dive_converted"] += 1
                    C["prizes_from_dives"] += took
                else:
                    C["dive_unconverted"] += 1
                    t = g["tgt"]
                    if t is not None:
                        # is the wounded body still visible, and in what state?
                        cands = [p for p in _slots(opp) if isinstance(p, dict)
                                 and p.get("id") == t["id"]]
                        best = None
                        for p in cands:
                            dmgd = (p.get("maxHp") or 0) - (p.get("hp") or 0)
                            if best is None or dmgd > best[0]:
                                best = ((p.get("maxHp") or 0) - (p.get("hp") or 0), p)
                        if best is None:
                            C["unconv_gone"] += 1          # scooped / shuffled away
                        else:
                            dmgd, p = best
                            if dmgd <= 0:
                                C["unconv_healed_full"] += 1
                            elif (p.get("maxHp") or 0) > t["maxHp"]:
                                C["unconv_grew_cap"] += 1  # Hero's Cape landed after the dive
                            elif dmgd < t["dmg_after"]:
                                C["unconv_healed_part"] += 1
                            else:
                                C["unconv_still_wounded"] += 1
                g["dived"] = False
                g["tgt"] = None
            g["my_prizes"] = len(me.get("prize") or [])
            g["opp_prizes"] = len(opp.get("prize") or [])
        if g["turn"] is None:
            g["my_prizes"] = len(me.get("prize") or [])
            g["opp_prizes"] = len(opp.get("prize") or [])
        g["turn"] = turn

        pick = agent(obs)

        opts = sel.get("option") or []
        for i in (pick or []):
            if not (0 <= i < len(opts) and isinstance(opts[i], dict)):
                continue
            if opts[i].get("attackId") == PHANTOM_DIVE:
                g["dived"] = True
                oa = (_slots(opp) or [None])[0]
                if isinstance(oa, dict):
                    hp, mx = oa.get("hp") or 0, oa.get("maxHp") or 0
                    g["tgt"] = {"id": oa.get("id"), "hp": hp, "maxHp": mx,
                                "dmg_after": (mx - hp) + 200}
                    if hp <= 200:
                        C["dive_would_ko"] += 1
                    else:
                        H[hp - 200] += 1
                        C["dive_left_alive"] += 1
                        # closers AVAILABLE at that moment
                        have_munki = any(isinstance(p, dict) and p.get("id") == MUNKIDORI
                                         and DARK in (p.get("energies") or [])
                                         for p in _slots(me))
                        have_clops = any(isinstance(p, dict)
                                         and p.get("id") in (DUSCLOPS, DUSKNOIR)
                                         for p in _slots(me))
                        if have_munki:
                            C["closer_munki_ready"] += 1
                        if have_clops:
                            C["closer_clops_onboard"] += 1
                        if not have_munki and not have_clops:
                            C["closer_none"] += 1
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

    n = float(a.games)
    print("=== %s vs %s, %d games: %d wins (%.1f%%) ===" % (a.deck, a.opp, a.games, wins,
                                                            100.0 * wins / n))
    print("  spec %s" % a.spec)
    print("  dives/game                     %6.2f" % (C["dives"] / n))
    print("  ... would KO on the spot       %6d  (target hp <= 200 when we pressed it)"
          % C["dive_would_ko"])
    print("  ... left the target ALIVE      %6d" % C["dive_left_alive"])
    print("      remaining-HP histogram      %s" % dict(sorted(H.items())))
    print("      closer ready: Munkidori+{D} %5d   Dusclops/Dusknoir on board %d   NEITHER %d"
          % (C["closer_munki_ready"], C["closer_clops_onboard"], C["closer_none"]))
    print("  one rotation later (unconverted dives only):")
    for k, lbl in (("unconv_still_wounded", "still wounded (we just didn't finish)"),
                   ("unconv_healed_part", "partially healed"),
                   ("unconv_healed_full", "healed to FULL"),
                   ("unconv_grew_cap", "maxHp grew (Cape landed after)"),
                   ("unconv_gone", "body gone (scooped/shuffled)")):
        print("      %-38s %5d" % (lbl, C[k]))
    print("  dives followed by a prize for us within one rotation: %d of %d (%.0f%%)"
          % (C["dive_converted"], C["dives"], 100.0 * C["dive_converted"] / max(1, C["dives"])))


if __name__ == "__main__":
    main()
