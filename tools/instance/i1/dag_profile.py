#!/usr/bin/env python3
"""What is the DAgger collection actually concentrating on?

Each record carries the ENGINE's answer (`chosen`) and a flag for whether the LM disagreed
(`lm_was_wrong`); the LM's own pick is NOT stored, so this can report what the LM FAILS TO
PLAY, not what it plays instead. The `lm_was_wrong=False` rows are a uniform 25% sample of
the decisions the LM already gets right, so their kind distribution is an unbiased picture
of the agreement population and makes a fair control.
"""
import collections
import gzip
import json
import sys


def kind(s):
    return s.split(":", 1)[0]


def body(s):
    return s.split(":", 1)[1] if ":" in s else ""


def main(path):
    lab = {True: collections.Counter(), False: collections.Counter()}
    off = {True: collections.Counter(), False: collections.Counter()}   # decisions offering k
    n = {True: 0, False: 0}
    ncand = {True: 0, False: 0}
    per_deck = collections.defaultdict(collections.Counter)
    cards = collections.defaultdict(collections.Counter)
    menu_size = {True: collections.Counter(), False: collections.Counter()}
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            w = bool(d["lm_was_wrong"])
            c = d["candidates"]
            k = kind(c[d["chosen"]])
            n[w] += 1
            ncand[w] += len(c)
            lab[w][k] += 1
            menu_size[w][min(len(c), 12)] += 1
            for u in set(kind(x) for x in c):
                off[w][u] += 1
            if w:
                per_deck[d["deck"]][k] += 1
                cards[k][body(c[d["chosen"]]).split("#")[0]] += 1

    print("records: wrong %d | right(25%% sample) %d | mean menu %.2f / %.2f"
          % (n[True], n[False], ncand[True] / max(1, n[True]), ncand[False] / max(1, n[False])))

    print("\n=== the move the LM FAILED to play (label kind), wrong vs already-right ===")
    print("%-12s %8s %8s %9s | %8s %8s %9s | %s"
          % ("kind", "wrong", "%", "offered%", "right", "%", "offered%", "lift"))
    for k, _ in lab[True].most_common():
        pw = 100.0 * lab[True][k] / max(1, n[True])
        pr = 100.0 * lab[False][k] / max(1, n[False])
        ow = 100.0 * off[True][k] / max(1, n[True])
        orr = 100.0 * off[False][k] / max(1, n[False])
        print("%-12s %8d %7.1f%% %8.1f%% | %8d %7.1f%% %8.1f%% | %+.2fx"
              % (k, lab[True][k], pw, ow, lab[False][k], pr, orr,
                 (pw / pr) if pr else float("nan")))

    print("\n=== menu size where the LM errs vs where it is right ===")
    for s in sorted(set(menu_size[True]) | set(menu_size[False])):
        print("  %2s%s  wrong %5.1f%%   right %5.1f%%"
              % (s, "+" if s == 12 else " ",
                 100.0 * menu_size[True][s] / max(1, n[True]),
                 100.0 * menu_size[False][s] / max(1, n[False])))

    print("\n=== top cards inside each kind (errors only) ===")
    for k in [k for k, _ in lab[True].most_common(6)]:
        top = ", ".join("%s x%d" % (c or "-", v) for c, v in cards[k].most_common(6))
        print("  %-10s %s" % (k, top))

    print("\n=== per-deck error mix (top 8 decks by records) ===")
    ks = [k for k, _ in lab[True].most_common(5)]
    print("  %-24s %6s  %s" % ("deck", "n", "  ".join("%-7s" % k for k in ks)))
    for deck, c in sorted(per_deck.items(), key=lambda kv: -sum(kv[1].values()))[:8]:
        t = sum(c.values())
        print("  %-24s %6d  %s"
              % (deck, t, "  ".join("%6.1f%%" % (100.0 * c[k] / t) for k in ks)))


if __name__ == "__main__":
    main(sys.argv[1])
