"""Per-deck row counts in the winner-only vs both-sides builds of the SAME games.

Tests the mechanism directly: --sides winner can only emit rows for games the pilot WON, so a
deck that loses often contributes fewer decisions -- and --cap-matchup cannot fix it, because
the cap is an upper bound, not a fill. If the ratio (winner rows / both-sides rows) tracks a
deck's win rate, then winner-only was systematically under-training exactly the decks that
need the most help, and v36's both-sides build removed that bias.
"""
import collections
import gzip
import json
import multiprocessing as mp
import sys


def count(path):
    c = collections.Counter()
    with gzip.open(path, "rt") as f:
        for line in f:
            i = line.find('"deck":"')
            if i < 0:
                c[json.loads(line).get("deck")] += 1
            else:
                j = line.index('"', i + 8)
                c[line[i + 8:j]] += 1
    return c


if __name__ == "__main__":
    both = sys.argv[1].split(",")
    win = sys.argv[2].split(",")
    with mp.Pool(len(both) + len(win)) as pool:
        res = pool.map(count, both + win)
    cb = collections.Counter()
    cw = collections.Counter()
    for c in res[:len(both)]:
        cb.update(c)
    for c in res[len(both):]:
        cw.update(c)

    decks = sorted(set(cb) | set(cw), key=lambda d: -(cb.get(d, 0)))
    tb, tw = sum(cb.values()), sum(cw.values())
    print("both-sides total %d   winner-only total %d   (%.1f%%)" % (tb, tw, 100.0 * tw / tb))
    print()
    print("%-24s %10s %10s %8s   %s" % ("deck", "both", "winner", "w/b %", "share both -> winner"))
    rows = []
    for d in decks:
        b, w = cb.get(d, 0), cw.get(d, 0)
        if b < 1000:
            continue
        rows.append((d, b, w, 100.0 * w / b, 100.0 * b / tb, 100.0 * w / max(1, tw)))
    rows.sort(key=lambda r: r[3])
    for d, b, w, r, sb, sw in rows:
        print("%-24s %10d %10d %8.1f   %5.2f%% -> %5.2f%%" % (d, b, w, r, sb, sw))
    if rows:
        rr = [r[3] for r in rows]
        print()
        print("winner/both ratio: min %.1f%% (%s)  max %.1f%% (%s)  spread %.2fx"
              % (rr[0], rows[0][0], rr[-1], rows[-1][0], rr[-1] / max(1e-9, rr[0])))
