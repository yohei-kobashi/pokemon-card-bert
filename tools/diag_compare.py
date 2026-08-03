#!/usr/bin/env python3
"""Compare two models' diagnostic dumps: do they fail the SAME WAY?

Deck-level win rates already said they do not fail on the same DECKS (r = +0.35 over 35 common
decks, with the alakazam family completely inverted). That leaves the sharper question: on the
states they do lose, is the mistake the same kind of mistake?

Two views, because they answer different things and can disagree:

  ALL   -- each model's own collapsed decks, i.e. the profile of "how this model fails".
           Comparable in kind but not paired: different decks, so a deck-specific habit
           (alakazam leans on abilities) can masquerade as a model-wide one.
  PAIRED-- only the decks both sets cover. Same decks, same opponent, so a difference here is
           the model. This is the one that carries weight; ALL is context.

Reported per view: the action-kind balance (LM share minus engine share) and the swap table
(engine picked X, LM picked Y on the same state). Balance says which kinds go missing; only the
swap table says what replaced them.
"""
import collections
import glob
import json
import sys


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        out[d.get("deck") or f.split("/")[-1][:-5]] = d
    if not out:
        raise SystemExit("no files match %r" % pattern)
    return out


def profile(decks):
    """-> (balance {kind: (mean_pp, n_decks_negative, n_decks)}, swaps {(x,y): (pct, n_decks)})"""
    bal = collections.defaultdict(list)
    swaps = collections.Counter()
    swap_decks = collections.defaultdict(set)
    total_swaps = 0
    dec = 0
    agree = [0, 0]
    for name, d in decks.items():
        s0, s1 = d["per_seat"]["0"], d["per_seat"]["1"]
        tot = s0["total"] + s1["total"]
        dec += tot
        for st in (s0, s1):
            agree[0] += st["agree"]
            agree[1] += st["total"]
        if tot:
            kinds = set(s0["lm"]) | set(s1["lm"]) | set(s0["ref"]) | set(s1["ref"])
            for k in kinds:
                lm = s0["lm"].get(k, 0) + s1["lm"].get(k, 0)
                rf = s0["ref"].get(k, 0) + s1["ref"].get(k, 0)
                bal[k].append(100.0 * (lm - rf) / tot)
        for st in (s0, s1):
            for kv, c in st["swap"].items():
                x, _, y = kv.partition("->")
                swaps[(x, y)] += c
                swap_decks[(x, y)].add(name)
                total_swaps += c
    balance = {k: (sum(v) / len(v), sum(1 for x in v if x < 0), len(v)) for k, v in bal.items()}
    swap = {k: (100.0 * c / max(1, total_swaps), len(swap_decks[k])) for k, c in swaps.items()}
    return balance, swap, dec, agree, len(decks)


def show(tag_a, a, tag_b, b):
    ba, sa, da, aga, na = a
    bb, sb, db, agb, nb = b
    print("  %-10s decks %2d  decisions %6d  agreement %.1f%%"
          % (tag_a, na, da, 100.0 * aga[0] / max(1, aga[1])))
    print("  %-10s decks %2d  decisions %6d  agreement %.1f%%"
          % (tag_b, nb, db, 100.0 * agb[0] / max(1, agb[1])))

    print("\n  action-kind balance (LM share - engine share, pp; +over-plays / -under-plays)")
    print("    %-10s %11s %11s   %s" % ("kind", tag_a, tag_b, "verdict"))
    kinds = sorted(set(ba) | set(bb), key=lambda k: -max(abs(ba.get(k, (0,))[0]),
                                                         abs(bb.get(k, (0,))[0])))
    for k in kinds:
        va = ba.get(k, (float("nan"), 0, 0))[0]
        vb = bb.get(k, (float("nan"), 0, 0))[0]
        if va != va or vb != vb:
            note = "only one model"
        elif abs(va) < 0.05 and abs(vb) < 0.05:
            # sub-selection contexts (card / num / energy / yes-no): the menu offers exactly one
            # kind, so both shares are the same by construction and the balance carries no signal
            note = "forced - no signal"
        elif va * vb > 0:
            note = "SAME direction"
        else:
            note = "OPPOSITE"
        fa = "%+7.2f" % va if va == va else "    n/a"
        fb = "%+7.2f" % vb if vb == vb else "    n/a"
        # per-deck consistency: how many decks share the sign
        ca = ba.get(k)
        cb = bb.get(k)
        ta = "%d/%d" % (ca[1] if ca[0] < 0 else ca[2] - ca[1], ca[2]) if ca else "-"
        tb = "%d/%d" % (cb[1] if cb[0] < 0 else cb[2] - cb[1], cb[2]) if cb else "-"
        print("    %-10s %7s %-4s %7s %-4s  %s" % (k, fa, ta, fb, tb, note))

    print("\n  swap table (engine picked X -> LM picked Y), %% of all swaps")
    print("    %-22s %10s %10s" % ("swap", tag_a, tag_b))
    top = sorted(set(list(sa) + list(sb)),
                 key=lambda k: -(sa.get(k, (0, 0))[0] + sb.get(k, (0, 0))[0]))[:14]
    for k in top:
        pa, na_ = sa.get(k, (0.0, 0))
        pb, nb_ = sb.get(k, (0.0, 0))
        print("    %-22s %6.1f%% %-4s %6.1f%% %-4s" % ("%s -> %s" % k, pa, "%d" % na_,
                                                        pb, "%d" % nb_))


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    ta = sys.argv[3] if len(sys.argv) > 3 else "A"
    tb = sys.argv[4] if len(sys.argv) > 4 else "B"
    A, B = load(pa), load(pb)
    print("=" * 78)
    print("VIEW 1: ALL -- each model on its own collapsed decks (not paired)")
    print("=" * 78)
    show(ta, profile(A), tb, profile(B))

    common = sorted(set(A) & set(B))
    print("\n" + "=" * 78)
    print("VIEW 2: PAIRED -- the %d decks both cover: %s" % (len(common), ", ".join(common)))
    print("=" * 78)
    if not common:
        raise SystemExit("no common decks -- the paired view is the one that counts, so this is "
                         "a coverage problem, not a result")
    show(ta, profile({k: A[k] for k in common}), tb, profile({k: B[k] for k in common}))


if __name__ == "__main__":
    main()
