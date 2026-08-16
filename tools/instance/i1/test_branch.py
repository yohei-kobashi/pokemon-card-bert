"""Exercise rl_branch over real games.

Answers the three questions that decide whether decision-level grouping is buildable:
  1. does the determinization invariant hold on real decisions, and where does it fail?
  2. do branches resolve to a terminal result?
  3. what does a branch point cost?

Run:  CUDA_VISIBLE_DEVICES="" python /root/test_branch.py [games] [K]
"""
import collections
import os
import sys
import time

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import library                                                   # noqa: E402
from cg.game import battle_start, battle_select, battle_finish    # noqa: E402
from lm.agent import make_lm_agent                                # noqa: E402
import rl_branch                                                  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NPLAY = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PAIRS = [("alakazam", "dragapult"), ("crustle_stall", "alakazam"),
         ("rockets_mewtwo", "dragapult")]


def main():
    stats = collections.Counter()
    fail_reasons = collections.Counter()
    qdist = collections.Counter()
    spreads = []
    t_branch = 0.0
    n_branch = 0

    for gi in range(GAMES):
        pilot, opp = PAIRS[gi % len(PAIRS)]
        d_me = library.read_deck(pilot)
        d_op = library.read_deck(opp)
        a_me = make_lm_agent(pilot, None, None)
        a_op = make_lm_agent(opp, None, None)
        pilot_i = gi % 2
        d0, d1 = (d_me, d_op) if pilot_i == 0 else (d_op, d_me)
        obs, _ = battle_start(d0, d1)
        if obs is None:
            continue
        try:
            for _ in range(4000):
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) != -1:
                    break
                sel = obs.get("select")
                if sel is None:
                    break
                yi = cur["yourIndex"]
                opts = sel.get("option") or []
                agent = a_me if yi == pilot_i else a_op

                # branch only on the pilot's real single-pick choices
                if (yi == pilot_i and len(opts) >= 2
                        and sel.get("minCount", 1) == 1 and sel.get("maxCount", 1) == 1):
                    stats["decisions_eligible"] += 1
                    try:
                        rl_branch.unseen_multisets(obs, d_me, d_op)
                        stats["determinization_ok"] += 1
                        ok = True
                    except rl_branch.DeterminizationError as e:
                        msg = str(e)
                        fail_reasons[msg.split(":")[0][:60]] += 1
                        ok = False
                    if ok and stats["branched"] < 40:
                        cands = [[i] for i in range(min(K, len(opts)))]
                        t0 = time.time()
                        try:
                            qs = rl_branch.branch_values(obs, d_me, d_op, pilot_i,
                                                         cands, a_me, a_op,
                                                         n_playouts=NPLAY)
                            t_branch += time.time() - t0
                            n_branch += 1
                            stats["branched"] += 1
                            stats["branch_playouts"] += len(cands) * NPLAY
                            stats["branch_resolved"] += sum(1 for q in qs if q is not None)
                            if all(q is not None for q in qs):
                                spread = max(qs) - min(qs)
                                spreads.append(spread)
                                qdist["spread<=%.2f" % (round(spread * 4) / 4.0)] += 1
                        except Exception as e:
                            fail_reasons["branch_values:" + type(e).__name__] += 1
                obs = battle_select(agent(obs))
        finally:
            battle_finish()

    print("=== determinization invariant ===")
    el = stats["decisions_eligible"]
    ok = stats["determinization_ok"]
    print("  eligible decisions : %d" % el)
    print("  invariant HOLDS    : %d  (%.1f%%)" % (ok, 100.0 * ok / max(1, el)))
    if fail_reasons:
        print("  failures:")
        for r, c in fail_reasons.most_common(10):
            print("    %4d  %s" % (c, r))
    print("\n=== branches ===")
    print("  branch points run  : %d" % stats["branched"])
    print("  playouts issued    : %d" % stats["branch_playouts"])
    print("  playouts resolved  : %d  (%.1f%%)"
          % (stats["branch_resolved"],
             100.0 * stats["branch_resolved"] / max(1, stats["branch_playouts"])))
    if n_branch:
        print("  cost per branch pt : %.3f s  (%d playouts each)"
              % (t_branch / n_branch, K))
    print("\n=== do candidates actually differ? (Q spread within a group) ===")
    print("  playouts per candidate : %d" % NPLAY)
    if spreads:
        import math
        mean_spread = sum(spreads) / len(spreads)
        nonzero = sum(1 for s_ in spreads if s_ > 1e-9)
        # if every candidate had the SAME true value, spread would still be nonzero from
        # sampling alone: K order statistics of a mean of NPLAY +/-1 draws
        noise_sd = 1.0 / math.sqrt(NPLAY)
        print("  groups                 : %d" % len(spreads))
        print("  mean Q spread          : %.3f" % mean_spread)
        print("  groups with any spread : %d (%.0f%%)"
              % (nonzero, 100.0 * nonzero / len(spreads)))
        print("  per-Q sampling sd      : %.3f  (spread must beat this to be signal)"
              % noise_sd)
    for k, v in sorted(qdist.items()):
        print("  %-14s %d" % (k, v))


if __name__ == "__main__":
    main()
