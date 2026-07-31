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

    agree = total = 0
    kinds_lm, kinds_ref, rank_of_ref = collections.Counter(), collections.Counter(), collections.Counter()
    by_turn = collections.defaultdict(lambda: [0, 0])
    disagree_examples = []
    results = collections.Counter()

    for g in range(args.games):
        obs, _sd = battle_start(ids, ids)
        if obs is None:
            continue
        try:
            for _ in range(4000):
                cur = obs.get("current") or {}
                if cur.get("result", -1) != -1:
                    results[cur["result"]] += 1
                    break
                sel = obs.get("select")
                if sel is None:
                    break
                yi = cur.get("yourIndex", 0)
                if yi != 0:                              # seat 1 is plain engine_v2
                    obs = battle_select(opp(obs))
                    continue
                opts = sel.get("option") or []
                pick_lm = lm_agent(obs)
                pick_ref = ref(obs)
                if len(opts) >= 2 and pick_lm and pick_ref:
                    a = encode_option(opts[pick_lm[0]], obs) if pick_lm[0] < len(opts) else "?"
                    b = encode_option(opts[pick_ref[0]], obs) if pick_ref[0] < len(opts) else "?"
                    t = int(float(cur.get("turn", 0) or 0))
                    total += 1
                    kinds_lm[kind_of(a)] += 1
                    kinds_ref[kind_of(b)] += 1
                    by_turn[min(t, 15)][1] += 1
                    if a == b:
                        agree += 1
                        by_turn[min(t, 15)][0] += 1
                    else:
                        if scorer is not None:
                            cands = [encode_option(o, obs) for o in opts]
                            try:
                                s = scorer.score(
                                    __import__("lm.serialize", fromlist=["x"]).serialize_stateless(
                                        obs, deck_ids=ids, deck_name=args.deck,
                                        **__import__("tools.rl_config", fromlist=["x"]).PROMPT_FMT),
                                    cands, obs)
                                order = sorted(range(len(s)), key=lambda i: -s[i])
                                rank_of_ref[order.index(pick_ref[0]) if pick_ref[0] < len(s)
                                            else -1] += 1
                            except Exception:
                                pass
                        if len(disagree_examples) < 15:
                            disagree_examples.append({"turn": t, "lm": a, "engine": b,
                                                      "n_opts": len(opts)})
                obs = battle_select(pick_lm)
        finally:
            battle_finish()

    print("games %d | LM(seat0) wins %d, losses %d, other %s"
          % (args.games, results.get(0, 0), results.get(1, 0),
             {k: v for k, v in results.items() if k not in (0, 1)}), flush=True)
    print("decisions %d | agreement with engine_v2 %.1f%%"
          % (total, 100.0 * agree / max(1, total)), flush=True)
    print("\naction kinds        LM      engine_v2")
    for k in sorted(set(kinds_lm) | set(kinds_ref), key=lambda x: -(kinds_lm[x] + kinds_ref[x])):
        print("  %-16s %5d (%4.1f%%) %5d (%4.1f%%)"
              % (k, kinds_lm[k], 100.0 * kinds_lm[k] / max(1, total),
                 kinds_ref[k], 100.0 * kinds_ref[k] / max(1, total)))
    if rank_of_ref:
        n = sum(rank_of_ref.values())
        print("\nwhen they disagree, the LM ranked engine_v2's move:")
        for r in sorted(rank_of_ref):
            print("  rank %-3s %5d (%4.1f%%)" % (r if r >= 0 else "?", rank_of_ref[r],
                                                 100.0 * rank_of_ref[r] / n))
    print("\nagreement by turn:")
    for t in sorted(by_turn):
        ok, n = by_turn[t]
        print("  T%-3d %5d decisions  %.1f%%" % (t, n, 100.0 * ok / max(1, n)))
    print("\nfirst disagreements:")
    for e in disagree_examples:
        print("  T%-3d (%2d opts)  LM %-28s engine %s" % (e["turn"], e["n_opts"], e["lm"],
                                                          e["engine"]))
    if args.out:
        json.dump({"agree": agree, "total": total, "kinds_lm": dict(kinds_lm),
                   "kinds_ref": dict(kinds_ref), "rank_of_ref": {str(k): v for k, v
                                                                 in rank_of_ref.items()},
                   "examples": disagree_examples}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
