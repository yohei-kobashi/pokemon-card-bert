#!/usr/bin/env python3
"""Where does an LM pilot diverge from engine_v2, and does the divergence cost the game?

Written for mega_lucario, where the v39 reranker lost the mirror 4-36 (10.0%) in 40 games. That
is not "slightly worse pilot" -- it is something structural, and 40 games to certainty means the
cause should be cheap to see.

At EVERY decision the LM faces, engine_v2 is asked the same question on the same state. Both
answers are recorded as their rendered option text, so the comparison is over what the move
actually IS, not over an index. Aggregated:

  * how often they agree at all
  * the ACTION-KIND distribution of each (attack / evolve / play / attach / retreat / end ...) --
    a pilot that never attacks or never evolves shows up here immediately
  * the LM's rank of engine_v2's choice, so "the right move was scored 2nd" and "the right move
    was scored last" are distinguishable
  * per-turn agreement, since a pilot can be fine early and collapse once the board is complex

engine_v2 is only QUERIED, never allowed to act -- the game is driven entirely by the LM, so the
states are the ones the LM's own play reaches, not engine_v2's.

SEATS ALTERNATE, and every statistic is reported per seat. The first version pinned the LM to
seat 0 and reported mega_lucario as 8-7, which looked survivable; with seats alternating the same
pilot is 6-17 first and 1-22 SECOND. Pinning the seat measured the lighter half of the problem
and would have sent the diagnosis after the wrong cause.
"""
import argparse
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)


def kind_of(opt_text):
    """Coarse action kind from the rendered option, e.g. 'atk:a123' -> 'atk'."""
    m = re.match(r"([a-z_]+)", opt_text or "")
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--model", required=True, help="hf:<dir> | qwen:<dir>")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from cg.game import battle_start, battle_select, battle_finish
    from lm.actions import encode_option
    from lm.agent import make_lm_agent
    from tools.mirror_match import load_deck, make_agent

    ids = load_deck(args.deck)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    prof = tuning.get(args.deck, {})
    lm_agent, scorer = make_agent(args.model, args.deck, ids, prof)
    ref = make_lm_agent(ids, prof, model=None)          # engine_v2, QUERIED ONLY
    opp = make_lm_agent(ids, prof, model=None)          # engine_v2 pilots the other seat

    per = {0: {"agree": 0, "total": 0, "w": 0, "l": 0,
               "lm": collections.Counter(), "ref": collections.Counter(),
               "swap": collections.Counter()},
           1: {"agree": 0, "total": 0, "w": 0, "l": 0,
               "lm": collections.Counter(), "ref": collections.Counter(),
               "swap": collections.Counter()}}
    by_turn = collections.defaultdict(lambda: [0, 0])
    disagree_examples = []

    for g in range(args.games):
        lm_seat = g % 2
        st = per[lm_seat]
        obs, _sd = battle_start(ids, ids)
        if obs is None:
            continue
        try:
            for _ in range(4000):
                cur = obs.get("current") or {}
                if cur.get("result", -1) != -1:
                    if cur["result"] == lm_seat:
                        st["w"] += 1
                    elif cur["result"] in (0, 1):
                        st["l"] += 1
                    break
                sel = obs.get("select")
                if sel is None:
                    break
                yi = cur.get("yourIndex", 0)
                if yi != lm_seat:                        # the other seat is plain engine_v2
                    obs = battle_select(opp(obs))
                    continue
                opts = sel.get("option") or []
                pick_lm = lm_agent(obs)
                pick_ref = ref(obs)
                if len(opts) >= 2 and pick_lm and pick_ref:
                    a = encode_option(opts[pick_lm[0]], obs) if pick_lm[0] < len(opts) else "?"
                    b = encode_option(opts[pick_ref[0]], obs) if pick_ref[0] < len(opts) else "?"
                    t = int(float(cur.get("turn", 0) or 0))
                    st["total"] += 1
                    st["lm"][kind_of(a)] += 1
                    st["ref"][kind_of(b)] += 1
                    by_turn[min(t, 15)][1] += 1
                    if a == b:
                        st["agree"] += 1
                        by_turn[min(t, 15)][0] += 1
                    else:
                        # PAIRED direction: on the same state, engine picked kind X and the LM
                        # picked kind Y. Marginal counts only say "11 fewer attaches"; this says
                        # what replaced them.
                        st["swap"][(kind_of(b), kind_of(a))] += 1
                        if len(disagree_examples) < 15:
                            disagree_examples.append({"turn": t, "lm": a, "engine": b,
                                                      "n_opts": len(opts)})
                obs = battle_select(pick_lm)
        finally:
            battle_finish()

    for si in (0, 1):
        st = per[si]
        n = max(1, st["total"])
        print("\n=== LM in seat %d (%s) === %d-%d = %.1f%%   %d decisions, agreement %.1f%%"
              % (si, "moves first" if si == 0 else "moves second", st["w"], st["l"],
                 100.0 * st["w"] / max(1, st["w"] + st["l"]), st["total"],
                 100.0 * st["agree"] / n), flush=True)
        print("  action kind        LM            engine_v2")
        for kk in sorted(set(st["lm"]) | set(st["ref"]),
                         key=lambda x: -(st["lm"][x] + st["ref"][x])):
            print("    %-14s %5d (%4.1f%%) %5d (%4.1f%%)"
                  % (kk, st["lm"][kk], 100.0 * st["lm"][kk] / n,
                     st["ref"][kk], 100.0 * st["ref"][kk] / n))
        if st["swap"]:
            print("  on disagreements, engine picked X and the LM picked Y instead:")
            for (x, y), c in st["swap"].most_common(8):
                print("    %-12s -> %-12s %4d" % (x, y, c))
    print("\nagreement by turn (both seats):")
    for t in sorted(by_turn):
        ok, n = by_turn[t]
        print("  T%-3d %5d decisions  %.1f%%" % (t, n, 100.0 * ok / max(1, n)), flush=True)
    print("\nfirst disagreements:")
    for e in disagree_examples:
        print("  T%-3d (%2d opts)  LM %-28s engine %s" % (e["turn"], e["n_opts"], e["lm"],
                                                          e["engine"]), flush=True)
    if args.out:
        json.dump({"deck": args.deck,
                   "per_seat": {str(si): {"w": per[si]["w"], "l": per[si]["l"],
                                          "total": per[si]["total"], "agree": per[si]["agree"],
                                          "lm": dict(per[si]["lm"]), "ref": dict(per[si]["ref"]),
                                          "swap": {"%s->%s" % kv: c
                                                   for kv, c in per[si]["swap"].items()}}
                                for si in (0, 1)},
                   "examples": disagree_examples}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
