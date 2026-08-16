"""How much of DECK[...] is ALREADY somewhere else in the state, by turn?

If a card token in our decklist also appears on the board / in hand / in the discard, the
model can learn the same policy without ever reading DECK[]. Redundancy is the mechanism by
which a segment becomes ignorable, so measuring it per turn says WHERE the segment can still
earn gradient -- and whether rendering the REMAINING deck (60 minus the known zones) instead
of the static 60 would make it non-redundant everywhere.
"""
import collections
import gzip
import json
import re
import sys

path = sys.argv[1]
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
RE_DECK = re.compile(r"^DECK\[([^\]]*)\]")
RE_TURN = re.compile(r"\bT(\d+)\.\d+")
RE_TOK = re.compile(r"\bc(\d+)")
BUCKETS = ((1, 2), (3, 5), (6, 10), (11, 999))


def bucket(t):
    for lo, hi in BUCKETS:
        if lo <= t <= hi:
            return f"T{lo}-{hi if hi < 999 else '+'}"
    return "T?"


agg = collections.defaultdict(lambda: [0, 0, 0, 0])   # rows, distinct, seen elsewhere, uniq
n = 0
with gzip.open(path, "rt") as f:
    for line in f:
        r = json.loads(line)
        st = r["state"]
        m = RE_DECK.match(st)
        if not m:
            continue
        deck = set(RE_TOK.findall(m.group(1)))
        rest = set(RE_TOK.findall(st[m.end():])) | set(
            t for c in r["candidates"] for t in RE_TOK.findall(c))
        b = bucket(int(RE_TURN.search(st).group(1)) if RE_TURN.search(st) else 0)
        a = agg[b]
        a[0] += 1
        a[1] += len(deck)
        a[2] += len(deck & rest)
        a[3] += len(deck - rest)
        n += 1
        if n >= cap:
            break

print(f"{'bucket':10s} {'rows':>8s} {'deck tok':>9s} {'also elsewhere':>15s} {'ONLY in DECK[]':>15s}")
tot = [0, 0, 0, 0]
for b in [f"T{lo}-{hi if hi < 999 else '+'}" for lo, hi in BUCKETS] + ["T?"]:
    if b not in agg:
        continue
    rows, dis, seen, uniq = agg[b]
    for i, v in enumerate((rows, dis, seen, uniq)):
        tot[i] += v
    print(f"{b:10s} {rows:8d} {dis / rows:9.1f} {seen / rows:9.1f} ({100 * seen / dis:4.1f}%) "
          f"{uniq / rows:9.1f} ({100 * uniq / dis:4.1f}%)")
rows, dis, seen, uniq = tot
print(f"{'ALL':10s} {rows:8d} {dis / rows:9.1f} {seen / rows:9.1f} ({100 * seen / dis:4.1f}%) "
      f"{uniq / rows:9.1f} ({100 * uniq / dis:4.1f}%)")
