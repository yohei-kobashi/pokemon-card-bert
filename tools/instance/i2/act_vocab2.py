"""Residual ambiguity of an action-token scheme, counting only collisions that CHANGE the act.

Two options that share a token but differ solely in a position number inside a hidden or
unordered pile (DECK3 vs DECK17) are the same act, so resolving them to the first match is
correct, not a guess. Only collisions that differ in kind, card, or a BOARD slot are real.
"""
import gzip, json, re, collections
RE = re.compile(r"(?:^| )(\d+)=(\S+)")
ID = re.compile(r"(c\d+|a\d+)")
POSZONE = re.compile(r"^(DECK|HAND|DISCARD|LOST|PRIZE|STADIUM)\d*$")


def split(o):
    kind = o.split(":", 1)[0]
    m = ID.search(o)
    cid = m.group(1) if m else kind
    tgt = o.split("@", 1)[1] if "@" in o else ""
    z = POSZONE.match(tgt)
    return kind, cid, (z.group(1) if z else tgt or "-")


def equivalent(a, b):
    """same act, differing only in which copy inside a positional pile"""
    return split(a) == split(b)


for mode in ("card_zone", "kind_card_zone"):
    vocab = collections.Counter()
    real = benign = n = 0
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
            kind, cid, z = split(opts[k])
            tokk = "%s@%s" % (cid, z) if mode == "card_zone" else "%s:%s@%s" % (kind, cid, z)
            vocab[tokk] += 1
            coll = []
            for i, o in enumerate(opts):
                if i == k:
                    continue
                ki, ci, zi = split(o)
                tk_ = "%s@%s" % (ci, zi) if mode == "card_zone" else "%s:%s@%s" % (ki, ci, zi)
                if tk_ == tokk:
                    coll.append(o)
            if coll:
                if all(equivalent(o, opts[k]) for o in coll):
                    benign += 1
                else:
                    real += 1
            if n >= 400000:
                break
    c = sorted(vocab.values(), reverse=True)
    print("\n=== %s : %d tokens ===" % (mode, len(c)))
    print("  no collision      %.2f%%" % (100.0 * (n - benign - real) / n))
    print("  benign collision  %.2f%%   (same act, different copy -> first match is correct)"
          % (100.0 * benign / n))
    print("  REAL  collision   %.2f%%   (%d decisions the token cannot resolve)"
          % (100.0 * real / n, real))
