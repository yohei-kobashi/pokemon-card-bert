#!/usr/bin/env python3
"""Price the (deck, kind) cells the loss analysis flagged: is the gap a CAUSE or a symptom?

tools/diag_lm_losses.py says, for example, that on mega_lucario_tr the winning seat evolves
29.9pp more often than the losing seat in the same mirror game -- same decklist, same shuffle,
same policy. That is a controlled contrast, but it is still an observation: it cannot separate
"evolving more WINS" from "winning lets you evolve more". Only a counterfactual can, and this
is the counterfactual.

THE COMPARISON IS TWO-WAY, NOT K-WAY. attach_value.py branches every candidate because it asks
"how much is this decision worth at all". Here the question is narrower and already named by
the analysis: at a decision where the flagged KIND was offered, is the LM on the right side of
it? Two branches instead of five or six is what makes the expensive rollout mode affordable.

WHICH SIDE, AND WHY BOTH. There are two different decisions to price and they are not the same
question:

  --side decline   decisions where the kind was offered and the LM did NOT take it. Branches the
                   LM's move against a move of that kind. Answers "should it take this MORE?"
  --side take      decisions where the LM DID take the kind. Branches it against non-kind
                   alternatives. Answers "should it take this LESS?"

The first run of this tool priced only `decline`, and that was the wrong side for the finding it
was pricing. Every observed gap from diag_lm_losses.py is NEGATIVE -- the winning seat takes the
kind LESS often than the losing seat -- so the hypothesis on the table is over-taking, and the
`decline` side cannot see it. Its verdict ("the LM is right to decline `end`") is true and does
not address the question. `both` runs each cell twice and emits a row per side.

ONE SIGN CONVENTION ACROSS BOTH SIDES: positive dQ = the flagged kind is UNDERVALUED here,
negative = OVERVALUED. On `decline` that is q(kind) - q(LM); on `take` it is q(LM's kind move) -
q(alternative), which is the same axis with the roles swapped.

THE ALTERNATIVE IS ARBITRARY, IN OPPOSITE DIRECTIONS ON THE TWO SIDES. On `decline` the branch
is one move of the flagged kind picked without ranking, which understates the kind. On `take` the
LM's move is the kind's BEST by the LM's own reckoning while the alternatives are sampled, which
overstates it. Both biases are conservative for the hypothesis each side tests.

**dQ_max IS DIAGNOSTIC ONLY. DO NOT DECIDE ON IT.** It compares the LM's move against the
best-MEASURED of k alternatives, and max-of-k over noisy estimates is biased upward by roughly
0.85 sd at k=3 -- so dQ_max is biased DOWNWARD by the same amount whatever the truth is. The
first run made this concrete: all six cells came back at dQ_max -0.13 to -0.21 with z -3.8 to
-6.1, INCLUDING mega_lucario_tr/evolve whose mean is +0.082 at z +2.44 and which the decline
side independently scores +0.079 at z +2.88. A statistic that is significantly negative on every
cell it is applied to, including the cells everything else says are positive, is measuring its
own selection bias. Only dQ (the mean over the sampled alternatives) enters a verdict.

TWO ROLLOUT MODES, and running both on purpose.

  --rollout engine   both sides engine_v2 after the branch. ~0.27 s/playout of pure CPU,
                     parallel across instance1's 40 workers, so a thousand branch points cost
                     minutes. But it measures Q^engine, not Q*: engine_v2 follows up an
                     off-policy move badly, which depresses every alternative and biases the
                     LM's own pick to look better than it is.
  --rollout lm       the LM plays OUR side for the first --lm-plies decisions after the branch,
                     then engine_v2 takes over. The bias lives in the immediate follow-up, so a
                     handful of plies removes most of it at a fraction of a full LM rollout.
                     ~0.1 s per LM decision on a contended GPU, serial -- roughly 60x the cost.

The intended use is NOT to pick one. Run `engine` over every cell, then run `lm` over the top
two or three. If they agree, the cheap number is trusted for the rest; if they disagree, the
rollout-policy bias is real for this task and only `lm` numbers may be used. Buying that
confidence on a subset is far cheaper than paying for it everywhere.

WHAT A RESULT MEANS. Positive `dQ` = taking the flagged kind is worth that much win rate over
what the LM actually played, on the +/-1 scale where 0.2 is about 10 percentage points. A cell
whose dQ is indistinguishable from zero is a SYMPTOM: the winners' behaviour correlates with
winning but does not cause it, and writing a rule from it is
[[setup-execution-audit-and-budew-overattack]] all over again.

    PYTHONPATH=cg-lib python3 tools/price_targets.py --targets evaluations/lm_targets.json \\
        --model qwen:/root/out/i2_r6 --rollout engine --points 200 --out priced.json
"""

import argparse
import collections
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def kind_of(t):
    import re
    m = re.match(r"([a-z_]+)", t or "")
    return m.group(1) if m else "?"


def capped_lm(lm_agent, engine_agent, plies):
    """LM for the first `plies` decisions of a playout, engine_v2 after that.

    Stateful ON PURPOSE and rebuilt per playout by the caller: a counter shared across
    playouts would give the first branch its LM plies and starve the rest, which is a
    systematic advantage to whichever candidate happened to be evaluated first.
    """
    box = {"n": 0}

    def f(obs):
        box["n"] += 1
        return (lm_agent if box["n"] <= plies else engine_agent)(obs)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, help="diag_lm_losses.py --targets output")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollout", choices=("engine", "lm"), default="engine")
    ap.add_argument("--side", choices=("decline", "take", "both"), default="decline",
                    help="price the decisions where the LM declined the kind, took it, or both")
    ap.add_argument("--alt-k", type=int, default=3,
                    help="--side take: non-kind alternatives branched against the LM's move")
    ap.add_argument("--lm-plies", type=int, default=4, help="--rollout lm: our-side LM plies")
    ap.add_argument("--playouts", type=int, default=16)
    ap.add_argument("--points", type=int, default=200, help="branch points per cell")
    ap.add_argument("--games", type=int, default=200, help="cap of games per cell")
    ap.add_argument("--cells", default="", help="deck/kind,deck/kind ... default: all")
    ap.add_argument("--top", type=int, default=0, help="only the N highest-|z| cells")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import rl_branch
    from lm.agent import make_lm_agent
    from mirror_env import DEFAULT_SO, MirrorEngine
    from mirror_match import load_deck, make_agent
    from lm.actions import encode_option

    cells = json.load(open(a.targets))["cells"]
    if a.cells:
        want = {tuple(x.split("/")) for x in a.cells.split(",")}
        cells = [c for c in cells if (c["deck"], c["kind"]) in want]
    if a.top:
        cells = cells[:a.top]
    if not cells:
        sys.exit("no cells selected")

    so = a.mirror_so or DEFAULT_SO
    eng = MirrorEngine(so)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    results = []
    t0 = time.time()

    sides = ("decline", "take") if a.side == "both" else (a.side,)
    for cell, side in [(c, s) for c in cells for s in sides]:
        deck, kind = cell["deck"], cell["kind"]
        ids = load_deck(deck)
        prof = tuning.get(deck, {})
        lm, _sc = make_agent(a.model, deck, ids, prof)
        engine = make_lm_agent(ids, prof, model=None)
        rng = random.Random(a.seed + hash(deck + kind) % 10000)
        opp_roll = engine                      # the opponent is engine_v2 in both modes

        diffs, dmax, n_pts, n_games = [], [], 0, 0
        for g in range(a.games):
            if n_pts >= a.points:
                break
            obs = eng.start(ids, ids, a.seed + g, mirror=1)
            n_games += 1
            if obs is None:
                continue
            try:
                for _ in range(4000):
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1:
                        break
                    sel = obs.get("select")
                    if sel is None:
                        break
                    yi = cur.get("yourIndex", 0)
                    raw = sel.get("option") or []
                    if yi != 0 or len(raw) < 2 or n_pts >= a.points:
                        obs = eng.select((lm if yi == 0 else opp_roll)(obs))
                        continue
                    texts = [encode_option(o, obs) for o in raw]
                    pick = lm(obs)
                    idx = pick[0] if isinstance(pick, (list, tuple)) else pick
                    of_kind = [i for i, t in enumerate(texts) if kind_of(t) == kind]
                    took = isinstance(idx, int) and idx in of_kind
                    if not of_kind:
                        obs = eng.select(pick)
                        continue
                    if side == "decline":
                        if took:                               # nothing to price: it took it
                            obs = eng.select(pick)
                            continue
                        alts = of_kind[:1]                     # LM's move vs a move of the kind
                    else:
                        if not took:                           # it declined; the other side's job
                            obs = eng.select(pick)
                            continue
                        oth = [i for i in range(len(texts)) if i not in of_kind]
                        if not oth:                            # forced -- not a decision at all
                            obs = eng.select(pick)
                            continue
                        alts = oth if len(oth) <= a.alt_k else rng.sample(oth, a.alt_k)
                    sels = [pick] + [[i] for i in alts]
                    if a.rollout == "lm":
                        me_roll = capped_lm(lm, engine, a.lm_plies)
                    else:
                        me_roll = engine
                    q = rl_branch.branch_values(obs, ids, ids, 0, sels, me_roll, opp_roll,
                                                n_playouts=a.playouts, rng=rng)
                    got = [v for v in q[1:] if v is not None]
                    if q[0] is None or not got:
                        obs = eng.select(pick)
                        continue
                    n_pts += 1
                    if side == "decline":
                        # + => taking the kind beats what the LM played
                        diffs.append(got[0] - q[0])
                        dmax.append(got[0] - q[0])
                    else:
                        # + => the kind move the LM played beats the alternative, i.e. it was
                        # right to take it. Negative on BOTH numbers is the over-taking case.
                        diffs.append(q[0] - sum(got) / len(got))
                        dmax.append(q[0] - max(got))
                    obs = eng.select(pick)                     # keep playing the LM's own line
            except Exception as e:
                print("   [%s/%s] game %d aborted: %s" % (deck, kind, g, e), flush=True)

        def _stat(xs):
            if not xs:
                return 0.0, 0.0
            u = sum(xs) / len(xs)
            s = math.sqrt(sum((x - u) ** 2 for x in xs) / max(1, len(xs) - 1))
            return u, s / math.sqrt(len(xs))

        m, se = _stat(diffs)
        mx, sex = _stat(dmax)
        row = {"deck": deck, "kind": kind, "side": side, "obs_gap_pp": cell["gap_pp"],
               "dQ": round(m, 4), "se": round(se, 4),
               "z": round(m / se, 2) if se else 0.0,
               "dQ_max": round(mx, 4), "se_max": round(sex, 4),
               "z_max": round(mx / sex, 2) if sex else 0.0,
               "points": len(diffs), "games": n_games, "rollout": a.rollout,
               "alt_k": a.alt_k if side == "take" else 1, "playouts": a.playouts}
        results.append(row)
        tail = ("" if side == "decline" else
                " | vs best alt %+.4f z %+5.2f" % (mx, row["z_max"]))
        print("  %-20s %-8s %-7s obs %+6.1fpp | dQ %+.4f ± %.4f  z %+5.2f | %d pts %d games%s"
              % (deck, kind, side, cell["gap_pp"], m, se, row["z"], len(diffs), n_games, tail),
              flush=True)
        with open(a.out, "w") as f:
            json.dump({"rollout": a.rollout, "playouts": a.playouts, "side": a.side,
                       "lm_plies": a.lm_plies if a.rollout == "lm" else None,
                       "cells": results}, f, indent=1)

    print("\nwrote %d rows to %s in %.1f min" % (len(results), a.out, (time.time() - t0) / 60))
    real = [r for r in results if r["se"] and abs(r["z"]) >= 2]
    print("CAUSES (|z| >= 2): %s"
          % (", ".join("%s/%s[%s] %+.4f" % (r["deck"], r["kind"], r["side"], r["dQ"])
                       for r in real) or "none"))
    # z_max is deliberately NOT in this rule -- see the max-selection bias note in the docstring.
    over = [r for r in results if r["side"] == "take" and r["z"] <= -2]
    if over:
        print("OVER-TAKEN (the LM's move loses to a typical alternative): %s"
              % ", ".join("%s/%s" % (r["deck"], r["kind"]) for r in over))
    take = [r for r in results if r["side"] == "take"]
    if take:
        neg = sum(1 for r in take if r["dQ"] < 0)
        print("take-side sign count: %d of %d cells negative "
              "(a fleet-wide lean shows up here even when no single cell clears |z| 2)"
              % (neg, len(take)))
    print("Everything else is a SYMPTOM at this sample size -- do not write a rule from it.")


if __name__ == "__main__":
    main()
