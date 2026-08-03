#!/usr/bin/env python3
"""Is the energy-attach TARGET a decision at all? Measure it with counterfactual playouts.

WHY. `attach-decisions-at-chance` measured the reranker picking attach targets at 16-29% top1
against a 14% chance level while every other decision kind runs +25 to +79pt above chance, and
deferring only attach to engine_v2 paid +11.4pt in play. Two very different explanations fit
that:

  (a) the model cannot learn a decision that is genuinely decidable, or
  (b) most attach targets are worth the same, engine_v2's pick between them is close to
      arbitrary, and the model is correctly failing to fit noise.

Imitation data cannot separate them -- engine_v2's opinion is the label AND the thing being
questioned. Counterfactual playouts can: branch the live state once per target, play each out
with engine_v2 on both sides, and read the win rate. `tools/rl_branch.py` already does the hard
part (K children from one root, one determinization shared across them so the hidden cards do
not differ between branches).

WHAT IS REPORTED
    spread          max Q - min Q over the targets, on a +/-1 scale (0.2 ~ 10pp of win rate)
    engine edge     Q(engine's pick) - mean Q(the others). If this is ~0, imitating engine_v2
                    on attach cannot help however well the model fits it.
    best edge       Q(best) - mean Q(others), SPLIT-SAMPLE: the argmax is taken on one half of
                    the playouts and scored on the other. Taking both from the same playouts
                    is a winner's curse and inflates the headroom -- the trap that produced a
                    meaningless "chosen was best 72.4%" in `engine-native-search-api`.
    decisive        share of decisions whose spread exceeds what pure playout noise produces,
                    estimated from the same split (a null spread is computed by scoring the
                    SAME candidate on the two halves).

Deliberately NOT restricted to the states the LM reaches: this asks whether the DECISION holds
signal, not whether a particular pilot walks into it, and the at-chance result was itself
measured on engine-piloted states.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _attach_groups(obs, opts):
    """-> [(canonical index into opts, text)] for the DISTINGUISHABLE attach targets.

    Uses the same collapse the training data now uses, so a pair the model is not asked to
    rank apart is not branched apart either -- branching them would spend playouts proving
    that two identical benched Pokemon are worth the same.
    """
    from lm.actions import encode_option
    from lm.action_token import dedup_options
    raw = [encode_option(o, obs) for o in opts]
    keep, pos, _keys = dedup_options(raw, obs)
    return [(pos[i], t) for i, t in enumerate(keep) if t.startswith("attach:")]


def _split(vals):
    """(mean of first half, mean of second half) for a list of playout results."""
    h = len(vals) // 2
    if h < 1:
        return None, None
    return sum(vals[:h]) / h, sum(vals[h:]) / len(vals[h:])


def _one_deck(job):
    deck, games, playouts, seed, max_points = job
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    import rl_branch

    rng = random.Random(seed)
    ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    prof = tuning.get(deck, {})
    me = make_lm_agent(ids, prof, model=None)
    opp = make_lm_agent(ids, prof, model=None)
    rows = []
    t0 = time.time()
    for g in range(games):
        if len(rows) >= max_points:
            break
        obs, _ = battle_start(ids, ids)
        if obs is None:
            continue
        try:
            for _ in range(4000):
                cur = obs.get("current") or {}
                if cur.get("result", -1) != -1 or obs.get("select") is None:
                    break
                yi = cur.get("yourIndex", 0)
                opts = (obs.get("select") or {}).get("option") or []
                pick = me(obs) if yi == 0 else opp(obs)
                if yi == 0 and len(opts) >= 2 and len(rows) < max_points:
                    grp = _attach_groups(obs, opts)
                    # Only where the engine ITSELF attaches. A decision that merely offers
                    # attach targets while engine_v2 plays a card is not the decision under
                    # test, and on the first probe it left `engine edge` measured on 1 point
                    # in 6 -- the branch budget went to states whose label is not an attach.
                    if len(grp) >= 2 and pick and pick[0] < len(opts) \
                            and any(i == pick[0] for i, _t in grp):
                        sels = [[i] for i, _t in grp]
                        # per-candidate playout lists, not just the mean: the split estimator
                        # needs the individual results
                        per = [[] for _ in sels]
                        for _s in range(playouts):
                            q = rl_branch.branch_values(
                                obs, ids, ids, 0, sels, me, opp, n_playouts=1, rng=rng)
                            for i, v in enumerate(q):
                                if v is not None:
                                    per[i].append(v)
                        if all(len(p) >= 4 for p in per):
                            eng = next((n for n, (i, _t) in enumerate(grp)
                                        if i == pick[0]), None)
                            rows.append({
                                "deck": deck, "q": per, "engine": eng,
                                "k": len(grp),
                                "prizes": len(cur["players"][0].get("prize") or []),
                                "options": len(opts),
                                "texts": [t for _i, t in grp]})
                obs = battle_select(pick)
        finally:
            battle_finish()
    return deck, rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=6, help="per deck")
    ap.add_argument("--playouts", type=int, default=16, help="scenarios per candidate")
    ap.add_argument("--max-points", type=int, default=40, help="branch points per deck")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default="/root/attach_value.json")
    a = ap.parse_args()

    import library
    decks = ([d.strip() for d in a.decks.split(",") if d.strip()]
             or sorted(library.list_decks()))
    jobs = [(d, a.games, a.playouts, 1000 + i, a.max_points) for i, d in enumerate(decks)]
    rows = []
    t0 = time.time()
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for deck, r, dt in pool.imap_unordered(_one_deck, jobs):
            rows += r
            print("  %-24s %3d branch points  %.0fs" % (deck, len(r), dt), flush=True)
    json.dump(rows, open(a.out, "w"))
    report(rows)
    print("\n%.1f min | %s" % ((time.time() - t0) / 60, a.out))


def report(rows):
    if not rows:
        print("no branch points")
        return
    rng = random.Random(7)
    spread, eng_edge, best_edge, null_spread = [], [], [], []
    by_prize = collections.defaultdict(list)
    by_k = collections.defaultdict(list)
    eng_is_best = tot = 0
    for r in rows:
        means = [sum(q) / len(q) for q in r["q"]]
        halves = [_split(q) for q in r["q"]]
        a_h = [h[0] for h in halves]
        b_h = [h[1] for h in halves]
        s = max(means) - min(means)
        spread.append(s)
        by_prize[r["prizes"]].append(s)
        by_k[min(r["k"], 6)].append(s)
        # engine's edge over the alternatives it passed over
        if r["engine"] is not None and len(means) > 1:
            others = [m for n, m in enumerate(means) if n != r["engine"]]
            eng_edge.append(means[r["engine"]] - sum(others) / len(others))
            tot += 1
            eng_is_best += (means[r["engine"]] >= max(means) - 1e-9)
        # split-sample: choose on half A, score on half B
        pick = max(range(len(a_h)), key=lambda i: a_h[i])
        others = [m for n, m in enumerate(b_h) if n != pick]
        if others:
            best_edge.append(b_h[pick] - sum(others) / len(others))
        # A PERMUTATION null. Pool every playout in the decision and redeal it into groups of
        # the same sizes: true differences between targets are destroyed, sample sizes are
        # preserved exactly. An earlier version compared the two HALVES instead, which made
        # the null noisier than the statistic it was judging (it ran on 12 playouts against
        # 24) and would have called real signal noise.
        pool = [v for q in r["q"] for v in q]
        sizes = [len(q) for q in r["q"]]
        draws = []
        for _t in range(8):
            rng.shuffle(pool)
            off, ms = 0, []
            for s in sizes:
                ms.append(sum(pool[off:off + s]) / s)
                off += s
            draws.append(max(ms) - min(ms))
        null_spread.append(sum(draws) / len(draws))

    def line(name, xs):
        if not xs:
            return
        m = sum(xs) / len(xs)
        sd = statistics.pstdev(xs)
        print("  %-24s mean %+.4f  sd %.4f  se %.4f  n %d"
              % (name, m, sd, sd / max(1, len(xs)) ** 0.5, len(xs)))

    print("\n=== attach target value, %d branch points ===" % len(rows))
    line("spread (max-min Q)", spread)
    line("NULL spread (noise)", null_spread)
    line("engine edge", eng_edge)
    line("best edge (split)", best_edge)
    # The null is what max-min looks like when every candidate is worth the same: it re-reads
    # the SAME playouts split in half, so it carries the identical sample size and candidate
    # count. A spread that does not clear it is not evidence of a decision.
    real = sum(1 for s, z in zip(spread, null_spread) if s > z)
    print("  spread beats its own null      : %.1f%% of %d decisions"
          % (100.0 * real / max(1, len(spread)), len(spread)))
    print("  engine's pick was the argmax   : %.1f%% of %d (chance = %.1f%%)"
          % (100.0 * eng_is_best / max(1, tot), tot,
             100.0 * sum(1.0 / r["k"] for r in rows) / max(1, len(rows))))
    print("\n  by prizes remaining:")
    for p in sorted(by_prize):
        line("    %d prizes" % p, by_prize[p])
    print("\n  by distinguishable targets:")
    for k in sorted(by_k):
        line("    k=%d" % k, by_k[k])


if __name__ == "__main__":
    main()
