#!/usr/bin/env python3
"""Is engine_v2's answer a function of the observation alone, or of its own history?

DAgger labels come from asking engine_v2 what it would do in a state the LM's play produced --
a state engine_v2 never walked toward. If its policy object carries per-episode state (what it
has already done this turn, cached plans, counters), that query returns an answer conditioned on
a history that did not happen, and every label built this way is quietly wrong.

Three answers are compared on the SAME observation:
  playing   the instance that has actually been driving the game
  fresh     a brand-new instance that has seen nothing
  reused    a second instance that has been QUERIED at every step but never allowed to act
            -- exactly the role it plays in tools/diag_pilot.py

playing vs fresh disagreeing means the policy is stateful and DAgger must rebuild the labeller
per decision. fresh vs reused disagreeing means the act of querying itself corrupts it, which is
worse: the diagnostics already gathered would be affected too.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    deck_name = sys.argv[1] if len(sys.argv) > 1 else "crustle_stall"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    import library

    ids = [int(x) for x in open(library.deck_path(deck_name)) if x.strip()]
    prof = json.load(open(os.path.join(ROOT, "agents", "tuning.json"))).get(deck_name, {})
    mk = lambda: make_lm_agent(ids, prof, model=None)          # noqa: E731

    playing = mk()
    reused = mk()
    opp = mk()
    n = same_fresh = same_reused = 0
    for g in range(games):
        obs, _ = battle_start(ids, ids)
        if obs is None:
            continue
        try:
            for _ in range(4000):
                cur = obs.get("current") or {}
                if cur.get("result", -1) != -1 or obs.get("select") is None:
                    break
                yi = cur.get("yourIndex", 0)
                if yi != 0:
                    obs = battle_select(opp(obs))
                    continue
                a = playing(obs)
                b = mk()(obs)                                   # fresh, never seen anything
                c = reused(obs)                                 # queried every step, never acts
                n += 1
                same_fresh += int(a == b)
                same_reused += int(a == c)
                obs = battle_select(a)
        finally:
            battle_finish()
    print("deck %s | %d decisions over %d games" % (deck_name, n, games))
    print("  playing == fresh   %5d / %d = %.2f%%" % (same_fresh, n, 100.0 * same_fresh / max(1, n)))
    print("  playing == reused  %5d / %d = %.2f%%" % (same_reused, n, 100.0 * same_reused / max(1, n)))
    if same_fresh == n:
        print("  -> stateless w.r.t. the observation: DAgger labels are safe to collect this way")
    else:
        print("  -> STATEFUL: a queried instance answers differently from one that has been "
              "playing. DAgger must rebuild the labeller per decision, and the diagnostics "
              "already collected inherit the same error.")


if __name__ == "__main__":
    main()
