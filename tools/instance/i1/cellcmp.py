import json, os

def norm(k):
    # "alakazam_nz_fez (RR) vs alakazam (heuristic)" -> "alakazam_nz_fez vs alakazam"
    if " vs " in k:
        left, right = k.split(" vs ", 1)
        return left.split(" (")[0].strip() + " vs " + right.split(" (")[0].strip()
    return k.replace("__", " vs ")

def cells(path):
    out = {}
    if path.endswith(".json"):
        d = json.load(open(path))
        for k, v in d.items():
            out[norm(k)] = (v, 60)
        return out
    for f in sorted(os.listdir(path)):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(path, f)))
        for k, v in d.get("results", {}).items():
            out[norm(k)] = (100.0 * v["win"] / v["games"], v["games"])
    return out

srcs = [("engine", "/root/out/wr_baseline.json"), ("v34", "/root/out/wr_v34"),
        ("v35", "/root/out/wr_v35"), ("lfm2", "/root/out/wr_lfm2"),
        ("v36", "/root/out/wr_v36")]
srcs = [(n, p) for n, p in srcs if os.path.exists(p)]
data = {n: cells(p) for n, p in srcs}
keys = list(data["engine"])

print("cell".ljust(32) + "".join(n.rjust(12) for n, _ in srcs) + "     best-engine")
for k in keys:
    row = ""
    for n, _ in srcs:
        c = data[n].get(k)
        row += ("%5.1f(%3d)" % c).rjust(12) if c else "-".rjust(12)
    lm = [data[n][k][0] for n, _ in srcs if n != "engine" and k in data[n]]
    d = "%+9.1f" % (max(lm) - data["engine"][k][0]) if lm else ""
    print(k.ljust(32) + row + d)
print()

def deckof(k):
    return k.split(" vs ")[0]

order, seen = [], set()
for k in keys:
    if deckof(k) not in seen:
        seen.add(deckof(k)); order.append(deckof(k))
print("PER DECK".ljust(32) + "".join(n.rjust(12) for n, _ in srcs))
for dk in order:
    row = ""
    for n, _ in srcs:
        cs = [data[n][k] for k in keys if deckof(k) == dk and k in data[n]]
        if not cs:
            row += "-".rjust(12); continue
        g = sum(c[1] for c in cs); w = sum(c[0] * c[1] / 100.0 for c in cs)
        row += ("%5.1f(%3d)" % (100.0 * w / g, g)).rjust(12)
    print(dk.ljust(32) + row)
print()
for n, _ in srcs:
    cs = list(data[n].values())
    if not cs:
        continue
    g = sum(c[1] for c in cs); w = sum(c[0] * c[1] / 100.0 for c in cs)
    print("%-7s OVERALL %.1f%%  (%d games)" % (n, 100.0 * w / g, g))
