import json, os, math
def load(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            out.update(json.load(open(os.path.join(d, f))))
    return out
g60 = load("/root/out/base_grid")
g300 = load("/root/out/base_grid300")
old = json.load(open("/root/out/wr_baseline.json"))
keys = sorted(g300)
print("%-34s %8s %8s %8s   %s" % ("cell", "old60", "new60", "n300", "300 SE"))
for k in keys:
    print("%-34s %8s %8s %8.1f   %6.1f" % (
        k,
        ("%.1f" % old[k]) if k in old else "-",
        ("%.1f" % g60[k]) if k in g60 else "-",
        g300[k], 100 * math.sqrt(g300[k] / 100 * (1 - g300[k] / 100) / 300)))
print()
decks = sorted(set(k.split(" vs ")[0] for k in keys))
print("%-24s %8s %8s %8s  %s" % ("PER DECK", "old60", "new60", "n300", "300 SE"))
for d in decks:
    ks = [k for k in keys if k.split(" vs ")[0] == d]
    def agg(src, n):
        vs = [src[k] for k in ks if k in src]
        return sum(vs) / len(vs) if vs else None
    a, b, c = agg(old, 60), agg(g60, 60), agg(g300, 300)
    p = c / 100.0
    se = 100 * math.sqrt(p * (1 - p) / (300 * len(ks)))
    print("%-24s %8s %8s %8.1f  %6.1f" % (
        d, ("%.1f" % a) if a else "-", ("%.1f" % b) if b else "-", c, se))
