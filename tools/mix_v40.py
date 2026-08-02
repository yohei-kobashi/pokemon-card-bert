#!/usr/bin/env python3
"""Build the v40 training mix: imitation base + DAgger + playout-VALUED attach records.

Proportions, and why each is what it is:

  base 85%      the imitation pool. Still the bulk -- the valued files only cover attach, which
                is 8.5% of decisions, and nothing else teaches the other 91.5%.
  dagger 10%    `dagger_r1` ONLY. Round 1 of the loop trained on it and gained +3.06pt paired;
                rounds 2 and 3 accumulated more DAgger from progressively weaker pilots and gave
                the gain back (-4.25pt, 16 decks turned WORSE). Accumulating is what regressed,
                so this reverts to the recipe that worked.
  valued 5%     `attach_q1` (which target) + `attach_q2` (whether to attach at all). Repeated to
                reach the share, because there are only ~21k of them: they cost 16 playouts per
                branch and 70-80% of branch points are discarded as value-neutral.

The valued share is a judgement, not a measurement. 5% of a 600k-sample run is ~30k slots over
~21k unique records, so each is seen roughly 1.4 times -- enough to move the attach behaviour
without letting a file that covers one decision kind dominate the update.
"""
import argparse
import gzip
import json
import os
import random
import sys


def count(path):
    n = 0
    with gzip.open(path, "rt") as f:
        for _ in f:
            n += 1
    return n


def reservoir(path, want, rng):
    """A UNIFORM sample without holding the file in memory.

    Reading the head instead would read whatever the file happens to be sorted by -- for the
    base pool that is matchup order, which once silently reduced 62 decks to 19.
    """
    res, n = [], 0
    with gzip.open(path, "rt") as f:
        for line in f:
            n += 1
            if len(res) < want:
                res.append(line)
            else:
                j = rng.randrange(n)
                if j < want:
                    res[j] = line
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--dagger", required=True)
    ap.add_argument("--valued", required=True, help="comma-separated")
    ap.add_argument("--dagger-frac", type=float, default=0.10)
    ap.add_argument("--valued-frac", type=float, default=0.05)
    ap.add_argument("--total", type=int, default=0,
                    help="FIXED round size. Without it the total is nd/dagger-frac, which ties "
                         "how much the round trains on to how many decks were bad enough to "
                         "collect from -- so the loop starves itself exactly as it succeeds. "
                         "Measured: round 1 targeted 3 decks (7,209 DAgger rows -> 144,180-row "
                         "mix); round 2 targeted 1 (3,361 -> 67,220), which also cut the valued "
                         "attach records from 7,209 of 21,600 to 3,361 and pushed the run from "
                         "3.6 epochs to 7.8 over half the data. With --total the DAgger is used "
                         "whole and its share simply falls where it falls.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    vpaths = [p for p in a.valued.split(",") if p]
    for p in [a.base, a.dagger] + vpaths:
        if not os.path.exists(p):
            raise SystemExit("missing: %s" % p)

    nd = count(a.dagger)
    total = a.total or int(round(nd / a.dagger_frac))
    # With a fixed total the DAgger is CAPPED at its share and subsampled uniformly when it
    # overruns. Collecting wide and then cutting is deliberate: 8 decks x 72 games subsampled to
    # 7,500 rows covers eight matchups, where 3 decks x 72 covers three, at the same cost -- the
    # screen and the training dominate the round, not the collection.
    want_d = min(nd, int(round(total * a.dagger_frac))) if a.total else nd
    want_v = int(round(total * a.valued_frac))
    want_b = total - want_d - want_v
    if want_b <= 0:
        raise SystemExit("fractions leave no room for the base pool")

    valued = []
    for p in vpaths:
        with gzip.open(p, "rt") as f:
            v = f.readlines()
        print("  valued %-42s %d" % (os.path.basename(p), len(v)), flush=True)
        valued += v
    if not valued:
        raise SystemExit("no valued records")
    reps = want_v / len(valued)

    rows = []
    with gzip.open(a.dagger, "rt") as f:
        dag = f.readlines()
    if want_d < len(dag):
        dag = rng.sample(dag, want_d)
    rows += dag
    print("  dagger %-42s %d of %d" % (os.path.basename(a.dagger), len(dag), nd), flush=True)
    # whole copies plus a sampled remainder, so the repetition is exact rather than lumpy
    full = int(reps)
    rows += valued * full
    rest = want_v - full * len(valued)
    if rest > 0:
        rows += rng.sample(valued, min(rest, len(valued)))
    print("  valued repeated x%.2f -> %d rows" % (reps, want_v), flush=True)
    rows += reservoir(a.base, want_b, rng)
    print("  base   %-42s %d sampled" % (os.path.basename(a.base), want_b), flush=True)

    rng.shuffle(rows)
    with gzip.open(a.out, "wt") as f:
        f.writelines(rows)
    print("\n%s: %d rows | dagger %.1f%% | valued %.1f%% | base %.1f%%"
          % (a.out, len(rows), 100.0 * want_d / len(rows), 100.0 * want_v / len(rows),
             100.0 * want_b / len(rows)), flush=True)


if __name__ == "__main__":
    main()
