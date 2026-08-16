"""Two questions the choice between a 1-token and a 2-token answer turns on.

1. Does the long tail of the atomic scheme actually get in the way? A rare token hurts twice --
   when it is the answer (bounded by its share of labels) and, less visibly, when it is merely
   ON THE MENU, because a near-random output row can out-score a well-trained correct one by
   chance. The second is not bounded by the label share, so the exposure is measured directly.

2. How big is the factored vocabulary? Splitting into <card> + <kind@slot> puts the first token
   on the EXISTING card vocabulary -- already trained by millions of prompt occurrences, and with
   tied embeddings its output row is its input row -- and leaves only kind-and-slot for the
   second. If that second vocabulary is small and well covered, the factored scheme has no tail
   at all.
"""
import gzip, json, re, sys, collections
sys.path.insert(0, ".")
from lm.action_token import action_token

RE = re.compile(r"(?:^| )(\d+)=(\S+)")
ID = re.compile(r"(c\d+)")
counts = json.load(open("data/action_vocab_v39.json"))["counts"]

first = collections.Counter()
second = collections.Counter()
pairs = set()
n = 0
exposure = collections.Counter()
lab_rare = collections.Counter()
amb2 = 0


def factor(o):
    """-> (first token, second token) for the <card> + <kind@slot> scheme"""
    kind, _, rest = o.partition(":")
    if not rest:
        return "c_none", "K|%s" % kind
    if kind == "facedown":
        return "c_none", "K|facedown"
    if kind == "num":
        return "c_none", "K|num|" + rest
    if kind == "attack":
        return ("a" + rest) if rest.isdigit() else rest, "K|attack"
    body, _, _sub = rest.partition("#")
    card, _, tgt = body.partition("@")
    m = re.match(r"^(DECK|HAND|DISCARD|LOST|PRIZE|STADIUM)\d*$", tgt or "")
    if m:
        tgt = m.group(1)
    return (card or "c_none"), ("K|%s@%s" % (kind, tgt) if tgt else "K|%s" % kind)


with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("target")
        if not t:
            continue
        opts = [o for _, o in RE.findall(d["prompt"].rsplit(":: ", 1)[-1])]
        k = int(t)
        if k >= len(opts):
            continue
        n += 1
        toks = [action_token(o) for o in opts]
        nr20 = sum(1 for x in toks if counts.get(x, 0) < 20)
        exposure[min(nr20, 5)] += 1
        c = counts.get(toks[k], 0)
        lab_rare["<20" if c < 20 else ("<100" if c < 100 else ">=100")] += 1
        fs = [factor(o) for o in opts]
        for a, b in fs:
            first[a] += 1
            second[b] += 1
            pairs.add((a, b))
        # does the factored pair still determine the act?
        if sum(1 for i, p in enumerate(fs) if p == fs[k] and opts[i] != opts[k]):
            from lm.action_token import equivalent
            if not all(equivalent(opts[i], opts[k]) for i, p in enumerate(fs) if p == fs[k]):
                amb2 += 1
        if n >= 400000:
            break

print("decisions %d\n" % n)
print("1) EXPOSURE to rare (<20 seen) tokens on the menu")
for k2 in sorted(exposure):
    print("   %s rare options: %6d decisions (%.1f%%)"
          % ("%d" % k2 if k2 < 5 else "5+", exposure[k2], 100.0 * exposure[k2] / n))
print("   the CORRECT token's training count: " +
      "  ".join("%s %.2f%%" % (k2, 100.0 * v / n) for k2, v in sorted(lab_rare.items())))

print("\n2) FACTORED <card> + <kind@slot>")
print("   first-token vocabulary  %d  (of which already-existing card/attack tokens: %d)"
      % (len(first), sum(1 for x in first if x != "c_none")))
print("   second-token vocabulary %d" % len(second))
c2 = sorted(second.values())
print("   second token seen <20:  %d of %d   median %d   min %d"
      % (sum(1 for x in c2 if x < 20), len(c2), c2[len(c2) // 2], c2[0]))
c1 = sorted(first.values())
print("   first  token seen <20:  %d of %d   median %d" % (sum(1 for x in c1 if x < 20), len(c1), c1[len(c1) // 2]))
print("   distinct (first, second) pairs actually used: %d" % len(pairs))
print("   residual ambiguity of the factored pair: %d (%.4f%%)" % (amb2, 100.0 * amb2 / n))
