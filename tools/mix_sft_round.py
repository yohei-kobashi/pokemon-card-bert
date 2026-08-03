#!/usr/bin/env python3
"""Build one decoder SFT round: a FRESH base sample + this round's DAgger + the valued records.

The base is SAMPLED, not taken whole. instance2's rounds so far re-used one frozen 193,919-row
file, so every round saw the same imitation examples and the rounds differed only by their DAgger
-- the base was a constant, not a sample. With a 5.7M-row pool behind it, drawing a different
uniform slice per round (``--seed``, set to the round number) means the model keeps meeting new
imitation data at no extra cost, which is the same fix ``dagger_loop3`` applied on instance1.

The sample is a reservoir so the pool never has to fit in memory, and uniform because the pool is
written in matchup order: reading the head would silently narrow 63 decks to a handful.

DAgger is THIS ROUND'S ONLY. Accumulating every round's collection is what ran instance1's loop
backwards (+3.06pt, then -4.25pt with 16 decks turning WORSE). The valued attach records are not
DAgger -- they are playout-measured labels that do not depend on which pilot collected them --
so they are carried in every round, each used exactly once.

Not to be confused with ``tools/mix_sft.py``, which merges imitation and [COMPARE] contrastive
data for the old multi-task format.
"""
import argparse
import gzip
import os
import random


def reservoir(path, want, rng):
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
    return res, n


def read_all(paths):
    rows = []
    for p in [x for x in paths if x]:
        with gzip.open(p, "rt") as f:
            v = f.readlines()
        print("  %-46s %d" % (os.path.basename(p), len(v)), flush=True)
        rows += v
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the big converted imitation pool")
    ap.add_argument("--base-n", type=int, default=200000, help="rows to sample from it")
    ap.add_argument("--dagger", default="", help="comma-separated, this round's only")
    ap.add_argument("--dagger-max-frac", type=float, default=0.0,
                    help="cap the DAgger share, subsampling uniformly when it overruns. 0 = use "
                         "the whole collection. Round 1 ran uncapped and landed at 18.1% (49,532 "
                         "rows of 273,314), which is where the risk lives: instance1 lost 2.75pt "
                         "across 63 decks in a round whose DAgger came from ONE deck. That round "
                         "also ran 7.8 epochs, and this trainer runs 1, so the two regimes are "
                         "not the same -- but if round 2's screen regresses on the decks that "
                         "were NOT targeted, this is the first knob to turn.")
    ap.add_argument("--valued", default="", help="comma-separated, used once each")
    ap.add_argument("--seed", type=int, required=True,
                    help="the ROUND number -- a fixed seed makes the base a constant")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    for p in [a.base] + [x for x in (a.dagger + "," + a.valued).split(",") if x]:
        if not os.path.exists(p):
            raise SystemExit("missing: %s" % p)

    print("valued:", flush=True)
    valued = read_all(a.valued.split(","))
    print("dagger:", flush=True)
    dagger = read_all(a.dagger.split(","))
    if a.dagger_max_frac > 0 and dagger:
        # solve n / (base_n + n + valued) <= f  for n
        cap = int(a.dagger_max_frac * (a.base_n + len(valued))
                  / max(1e-9, 1.0 - a.dagger_max_frac))
        if cap < len(dagger):
            print("  capped %d -> %d rows (%.1f%% of the round)"
                  % (len(dagger), cap, 100 * a.dagger_max_frac), flush=True)
            dagger = rng.sample(dagger, cap)
    print("base:", flush=True)
    base, npool = reservoir(a.base, a.base_n, rng)
    print("  %-46s %d of %d sampled (seed %d)"
          % (os.path.basename(a.base), len(base), npool, a.seed), flush=True)

    rows = base + dagger + valued
    rng.shuffle(rows)
    with gzip.open(a.out, "wt") as f:
        f.writelines(rows)
    t = len(rows)
    print("\n%s: %d rows | base %.1f%% | dagger %.1f%% | valued %.1f%%"
          % (a.out, t, 100.0 * len(base) / t, 100.0 * len(dagger) / t,
             100.0 * len(valued) / t), flush=True)


if __name__ == "__main__":
    main()
