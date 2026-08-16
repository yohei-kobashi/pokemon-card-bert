"""Is the REMAINING deck rendered in the TRAINING data actually right?

build_rerank replays logged observations rather than a live battle, so its obs could be
missing a zone (discard, attached energy cards, the evolution stack) that my_known_ids
subtracts. That would silently render a wrong DECK[] in training while inference renders a
right one -- a train/deploy divergence with no error message.

The state string carries its own check: ``dk<N>`` (library) and ``pz<M>`` (prizes) come
straight from the obs, and the cards we cannot see are exactly those two zones. So
sum(DECK[] counts) must equal N + M. The known exception is a sub-selection, where the card
being resolved sits in no zone and the total runs 1 high.
"""
import collections
import gzip
import json
import re
import sys

path = sys.argv[1]
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
RE_DECK = re.compile(r"^DECK\[([^\]]*)\]")
RE_ENTRY = re.compile(r"c(\d+)(?:x(\d+))?")
RE_ME = re.compile(r" ME .*? pz(\d+) dk(\d+) ")

diff = collections.Counter()
n = skipped = 0
with gzip.open(path, "rt") as f:
    for line in f:
        r = json.loads(line)
        st = r["state"]
        m = RE_DECK.match(st)
        mm = RE_ME.search(st)
        if not m or not mm:
            skipped += 1
            continue
        total = sum(int(c or 1) for _i, c in RE_ENTRY.findall(m.group(1)))
        want = int(mm.group(2)) + int(mm.group(1))
        diff[total - want] += 1
        n += 1
        if n >= cap:
            break

ok = diff.get(0, 0)
print(f"rows {n} (skipped {skipped})   exact {ok} = {100.0 * ok / max(1, n):.1f}%")
print("surplus histogram:", dict(sorted(diff.items())))
