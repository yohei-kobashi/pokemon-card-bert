#!/usr/bin/env python3
"""mega_venusaur is 28% of the whole error pool. Is that 24 games of varied mistakes, or one
mistake repeated? Distinct states answer it: DAgger is only worth its cost if the states are
new, and a state the LM revisits 50 times inside one game contributes one lesson, not 50."""
import collections
import gzip
import hashlib
import json
import sys


def main(path, deck):
    n = 0
    seen = collections.Counter()
    per_game = []
    menus = collections.Counter()
    lab_kind = collections.Counter()
    ex = None
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d["deck"] != deck or not d["lm_was_wrong"]:
                continue
            n += 1
            h = hashlib.md5(d["state"].encode()).hexdigest()[:12]
            seen[h] += 1
            k = d["candidates"][d["chosen"]].split(":", 1)[0]
            lab_kind[k] += 1
            menus[" ".join(sorted(set(c.split(":", 1)[0] for c in d["candidates"])))] += 1
            if ex is None and k == "end":
                ex = d
    print("%s: %d error records | %d DISTINCT states (%.1f%% unique)"
          % (deck, n, len(seen), 100.0 * len(seen) / max(1, n)))
    print("repeat histogram: " + ", ".join(
        "%dx:%d states" % (r, c) for r, c in sorted(collections.Counter(seen.values()).items())[:12]))
    print("most repeated state seen %d times" % max(seen.values()))
    print("\nlabel kinds: %s" % lab_kind.most_common())
    print("\nmenu kind-sets (top 8):")
    for m, c in menus.most_common(8):
        print("  %6d  %s" % (c, m))
    if ex:
        print("\n--- one 'end' error state (truncated) ---")
        s = ex["state"]
        print(s[-1200:])
        print("candidates: %s" % ex["candidates"])
        print("engine chose: %s" % ex["candidates"][ex["chosen"]])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
