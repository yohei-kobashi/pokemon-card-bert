import gzip, json, re, collections
RE = re.compile(r"(?:^| )(\d+)=(\S+)")
ID = re.compile(r"(c\d+|a\d+)")


def parts(o):
    kind = o.split(":", 1)[0]
    m = ID.search(o)
    tgt = o.split("@", 1)[1] if "@" in o else ""
    return kind, (m.group(1) if m else None), tgt


# A target is POSITIONAL (which copy, inside an unordered or hidden pile) or SUBSTANTIVE (which
# Pokemon on the board the action applies to). Only the latter changes the outcome, so only the
# latter makes a card-id label genuinely ambiguous.
POS = re.compile(r"^(DECK|HAND|DISCARD|LOST|PRIZE)\d*$")
n = benign = real = noid = 0
ex = collections.Counter()
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
        ck, cc, ctg = parts(opts[k])
        if cc is None:
            noid += 1
            continue
        coll = [o for o in opts if parts(o)[1] == cc and o != opts[k]]
        if not coll:
            continue
        ok = all(parts(o)[0] == ck and POS.match(parts(o)[2] or "X") and POS.match(ctg or "X")
                 for o in coll)
        if ok:
            benign += 1
        else:
            real += 1
            ex[(ck, parts(coll[0])[0], (ctg or "-")[:7], (parts(coll[0])[2] or "-")[:7])] += 1
        if n >= 400000:
            break
tot = n - noid
print("decisions with an id      %d" % tot)
print("  no collision            %d (%.1f%%)" % (tot - benign - real, 100 * (tot - benign - real) / tot))
print("  BENIGN collision        %d (%.1f%%)  same kind, both targets positional (which copy)" % (benign, 100 * benign / tot))
print("  REAL   collision        %d (%.1f%%)  different kind, or a board target" % (real, 100 * real / tot))
print()
print("what the real collisions look like (chosen kind/target vs colliding kind/target):")
for k, v in ex.most_common(10):
    print("   %-9s vs %-9s | %-7s vs %-7s | %6d" % (k + (v,)))
