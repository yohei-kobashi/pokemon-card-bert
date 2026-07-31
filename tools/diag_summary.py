#!/usr/bin/env python3
"""Aggregate the per-deck diagnostics: what do the collapsed decks have in COMMON?

36 of 63 decks came back WORSE, median mirror win rate 35.5%. That is not a per-deck bug list,
so the useful question is which failure repeats across decks -- one shared cause would mean one
fix, where 36 separate causes would mean 36.

Reported:
  * seat asymmetry, since mega_lucario was 26% first / 4% second and the aggregate hid it
  * the action-kind BALANCE (LM share minus engine share) averaged over decks -- which kinds the
    LM systematically over- and under-plays
  * the paired swap table pooled over decks: on the same state, engine picked X and the LM
    picked Y instead. Marginals say a kind is under-used; only this says what replaced it.
  * how consistent each swap is ACROSS decks, so a fleet-wide habit is distinguishable from one
    deck contributing every instance
"""
import collections
import glob
import json
import sys


def main(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("no files match %r" % pattern)
    seat_w = {0: [0, 0], 1: [0, 0]}
    agree = [0, 0]
    bal = collections.defaultdict(list)          # kind -> per-deck (lm_share - ref_share)
    swaps = collections.Counter()
    swap_decks = collections.defaultdict(set)
    per_deck_seat = []
    for f in files:
        d = json.load(open(f))
        name = d.get("deck") or f.split("/")[-1][:-5]
        s0, s1 = d["per_seat"]["0"], d["per_seat"]["1"]
        for si, st in ((0, s0), (1, s1)):
            seat_w[si][0] += st["w"]
            seat_w[si][1] += st["l"]
            agree[0] += st["agree"]
            agree[1] += st["total"]
        tot = s0["total"] + s1["total"]
        if tot:
            kinds = set(s0["lm"]) | set(s1["lm"]) | set(s0["ref"]) | set(s1["ref"])
            for k in kinds:
                lm = s0["lm"].get(k, 0) + s1["lm"].get(k, 0)
                rf = s0["ref"].get(k, 0) + s1["ref"].get(k, 0)
                bal[k].append(100.0 * (lm - rf) / tot)
        for st in (s0, s1):
            for kv, c in st["swap"].items():
                swaps[kv] += c
                swap_decks[kv].add(name)
        w0, l0 = s0["w"], s0["l"]
        w1, l1 = s1["w"], s1["l"]
        per_deck_seat.append((name, w0, l0, w1, l1))

    print("decks %d | decisions %d | agreement with engine_v2 %.1f%%"
          % (len(files), agree[1], 100.0 * agree[0] / max(1, agree[1])))
    for si in (0, 1):
        w, l = seat_w[si]
        print("  LM in seat %d (%s): %d-%d = %.1f%%"
              % (si, "first" if si == 0 else "second", w, l, 100.0 * w / max(1, w + l)))

    print("\naction-kind balance, mean over decks of (LM share - engine share), pp:")
    for k, v in sorted(bal.items(), key=lambda kv: -abs(sum(kv[1]) / len(kv[1]))):
        m = sum(v) / len(v)
        if abs(m) < 0.3:
            continue
        n_neg = sum(1 for x in v if x < 0)
        print("  %-12s %+6.2f pp   (under-played in %d/%d decks)" % (k, m, n_neg, len(v)))

    print("\npooled swaps -- engine picked X, the LM picked Y on the SAME state:")
    tot_sw = sum(swaps.values())
    for (x, y), c in swaps.most_common(14):
        print("  %-11s -> %-11s %5d (%4.1f%%)  seen in %2d/%d decks"
              % (x, y, c, 100.0 * c / tot_sw, len(swap_decks[(x, y)]), len(files)))

    print("\nper-deck seat split (LM wins-losses):")
    for name, w0, l0, w1, l1 in sorted(per_deck_seat,
                                       key=lambda r: (r[3] / max(1, r[3] + r[4]))):
        print("  %-22s first %2d-%-2d   second %2d-%-2d" % (name, w0, l0, w1, l1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/diag_worse/*.json")
