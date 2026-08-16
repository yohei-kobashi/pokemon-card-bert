#!/usr/bin/env python3
"""How often is a GAME-WINNING attack available and not taken?

`lethal_now` is not a matter of taste like "attack with Phantom Dive now vs develop first" --
it fires only when the attack on the menu takes our LAST prizes (Active knock-out, or Phantom
Dive's six counters finishing bench bodies, or both; see tools/dusk_plan.py). Missing it does not
cost tempo, it throws away a won game.

WHY NOT MEASURE IT WITH A WIN-RATE GATE. It fired on 5 turns across 19 live games -- about 0.26
per game. A rule that rare cannot move a win rate out of the noise even at 1200 games, so the
2026-08-12 deferral gate (r5 25.6% vs prohibitions-only 29.6%) could not attribute anything to
it: lethal_now was bundled with three frequent rules and the +4.0 belongs to dropping those.
The right statistic is the DIRECT one -- opportunities offered, and how many were taken.

    PYTHONPATH=cg-lib:tools python tools/lethal_check.py --games 60 \\
        --spec 'planfilter:clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace:hf:/root/out/mrl2_r5b'
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EPISODE_START = {"current": None, "logs": [], "remainingOverageTime": 600.0,
                  "search_begin_input": None, "select": None, "step": 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="agent spec for the protagonist")
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--opp", default="marnie_grimmsnarl,alakazam_nz,dragapult,ogerpon_mono")
    ap.add_argument("--games", type=int, default=60, help="per opponent")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--fmt", default="dusk", choices=("prompt", "dusk"))
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import dusk_plan
    import mirror_match as mm
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

    eng = MirrorEngine(a.mirror_so or DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    my_ids = mm.load_deck(a.deck)
    mm._FMT = a.fmt
    agent, _sc = mm.make_agent(a.spec, a.deck, my_ids, tuning.get(a.deck, {}))

    stat = {"offered": 0, "taken": 0, "games": 0, "games_with_miss": set()}
    cur_game = [0]

    def watched(obs):
        try:
            sel = obs.get("select") or {}
            if len(sel.get("option") or []) >= 2:
                live = dusk_plan.opportunities(obs)
                hit = live.get("lethal_now")
                if hit and hit[0]:
                    stat["offered"] += 1
                    pick = agent_inner(obs)
                    if set(pick if isinstance(pick, (list, tuple)) else [pick]) & set(hit[0]):
                        stat["taken"] += 1
                    else:
                        stat["games_with_miss"].add(cur_game[0])
                    return pick
        except Exception:                                   # noqa: BLE001
            pass                                            # a diagnostic must never pilot
        return agent_inner(obs)

    agent_inner = agent
    for opp in [o for o in a.opp.split(",") if o]:
        opp_ids = mm.load_deck(opp)
        opp_agent, _ = mm.make_agent("engine", opp, opp_ids, tuning.get(opp, {}))
        for g in range(a.games):
            cur_game[0] = (opp, g)
            watched(_EPISODE_START)
            mine = g % 2
            s = a.seed + g // 2
            if mine == 0:
                play(eng, watched, opp_agent, my_ids, opp_ids, s, mirror=1)
            else:
                play(eng, opp_agent, watched, opp_ids, my_ids, s, mirror=1)
            stat["games"] += 1
        print("  %-22s cumulative: %d offered / %d taken" % (opp, stat["offered"], stat["taken"]),
              flush=True)

    miss = stat["offered"] - stat["taken"]
    print("\nspec: %s" % a.spec)
    print("  games                    %d" % stat["games"])
    print("  lethal decisions offered %d  (%.2f per game)"
          % (stat["offered"], stat["offered"] / max(1, stat["games"])))
    print("  taken                    %d" % stat["taken"])
    print("  MISSED WINS              %d  (%.1f%% of chances, in %d games)"
          % (miss, 100.0 * miss / max(1, stat["offered"]), len(stat["games_with_miss"])))
    if a.out:
        json.dump({"spec": a.spec, "games": stat["games"], "offered": stat["offered"],
                   "taken": stat["taken"], "missed": miss,
                   "games_with_miss": len(stat["games_with_miss"])}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
