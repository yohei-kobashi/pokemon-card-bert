"""How big is an action-token vocabulary, and is every token seen often enough to learn?"""
import gzip, json, re, collections
RE = re.compile(r"(?:^| )(\d+)=(\S+)")
ID = re.compile(r"(c\d+|a\d+)")
# Positional zones carry an index that names WHICH COPY inside an unordered or hidden pile.
# Keeping it would multiply the vocabulary by ~60 and encode nothing: picking c305 at DECK1 or
# at DECK6 is the same act. Board slots are the opposite -- ACTIVE0 and BENCH2 are different
# Pokemon -- so their index is kept.
POSZONE = re.compile(r"^(DECK|HAND|DISCARD|LOST|PRIZE|STADIUM|)\d*$")


def zone(o):
    t = o.split("@", 1)[1] if "@" in o else ""
    m = POSZONE.match(t)
    return (m.group(1) or "-") if m else t


def key(o, mode):
    kind = o.split(":", 1)[0]
    m = ID.search(o)
    cid = m.group(1) if m else kind          # end / retreat / yes / no name themselves
    if mode == "card_zone":
        return "%s@%s" % (cid, zone(o))
    return "%s:%s@%s" % (kind, cid, zone(o))


for mode in ("card_zone", "kind_card_zone"):
    vocab = collections.Counter()
    amb = n = 0
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
            ks = [key(o, mode) for o in opts]
            vocab[ks[k]] += 1
            if sum(1 for i, x in enumerate(ks) if x == ks[k] and opts[i] != opts[k]):
                amb += 1
            if n >= 400000:
                break
    c = sorted(vocab.values(), reverse=True)
    tot = sum(c)
    print("\n=== mode %s ===" % mode)
    print("  distinct tokens        %d" % len(c))
    print("  residual ambiguity     %d / %d (%.2f%%)" % (amb, n, 100.0 * amb / n))
    for thr in (5, 20, 100):
        rare = sum(1 for x in c if x < thr)
        mass = sum(x for x in c if x < thr)
        print("  seen < %-4d            %5d tokens (%.0f%% of vocab), %.2f%% of the label mass"
              % (thr, rare, 100.0 * rare / len(c), 100.0 * mass / tot))
    print("  median count %d | p10 %d | max %d" % (c[len(c) // 2], c[int(len(c) * 0.9)], c[0]))
