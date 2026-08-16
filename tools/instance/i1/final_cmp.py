"""LM vs engine_v2, per cell and per deck, all at 300 games/cell with standard errors.

The engine baseline (base_grid300) and both LM runs use the same sample size, so a delta can
finally be read against its own noise instead of against a 60-game guess. Deltas are printed
with the SE of the DIFFERENCE and flagged only past 2 SE -- at 60 games/cell the same protocol
moved the engine's own mega_lucario score by 8.9pt between two identical runs.
"""
import json, math, os, sys

BASE = "/root/out/base_grid300"
RUNS = [("v35", "/root/out/wr_v35_300"), ("v36", "/root/out/wr_v36_300")]
GAMES = 300


def norm(k):
    if " vs " in k:
        left, right = k.split(" vs ", 1)
        return left.split(" (")[0].strip() + " vs " + right.split(" (")[0].strip()
    return k.replace("__", " vs ")


def load_run(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            for k, v in json.load(open(os.path.join(d, f))).get("results", {}).items():
                out[norm(k)] = (100.0 * v["win"] / v["games"], v["games"])
    return out


def load_base():
    out = {}
    for f in sorted(os.listdir(BASE)):
        if f.endswith(".json"):
            for k, v in json.load(open(os.path.join(BASE, f))).items():
                out[norm(k)] = (v, GAMES)
    return out


def se(p, n):
    p /= 100.0
    return 100.0 * math.sqrt(max(p * (1 - p), 1e-9) / n)


base = load_base()
runs = {n: load_run(d) for n, d in RUNS}
cells = [k for k in base if any(k in r for r in runs.values())]
if not cells:
    raise SystemExit("no LM cells yet (%s)" % ", ".join(
        "%s:%d" % (n, len(runs[n])) for n, _ in RUNS))
decks, seen = [], set()
for k in cells:
    d = k.split(" vs ")[0]
    if d not in seen:
        seen.add(d); decks.append(d)

hdr = "cell".ljust(30) + "engine".rjust(12)
for n, _ in RUNS:
    hdr += n.rjust(12) + ("d" + n).rjust(14)
print(hdr)
for k in cells:
    b, bn = base[k]
    line = k.ljust(30) + ("%5.1f" % b).rjust(12)
    for n, _ in RUNS:
        c = runs[n].get(k)
        if not c:
            line += "-".rjust(12) + "".rjust(14); continue
        d = c[0] - b
        s = math.hypot(se(c[0], c[1]), se(b, bn))
        mark = "**" if abs(d) > 2 * s else ("*" if abs(d) > s else "  ")
        line += ("%5.1f" % c[0]).rjust(12) + ("%+6.1f+-%.1f%s" % (d, s, mark)).rjust(14)
    print(line)

print()
print("PER DECK".ljust(30) + "engine".rjust(12) +
      "".join(n.rjust(12) + ("d" + n).rjust(14) for n, _ in RUNS))
for dk in decks:
    ks = [k for k in cells if k.split(" vs ")[0] == dk]
    bw = sum(base[k][0] for k in ks) / len(ks)
    bn = sum(base[k][1] for k in ks)
    line = dk.ljust(30) + ("%5.1f" % bw).rjust(12)
    for n, _ in RUNS:
        cs = [runs[n][k] for k in ks if k in runs[n]]
        if len(cs) != len(ks):
            line += "-".rjust(12) + "".rjust(14); continue
        g = sum(c[1] for c in cs)
        w = sum(c[0] * c[1] / 100.0 for c in cs)
        p = 100.0 * w / g
        d = p - bw
        s = math.hypot(se(p, g), se(bw, bn))
        mark = "**" if abs(d) > 2 * s else ("*" if abs(d) > s else "  ")
        line += ("%5.1f" % p).rjust(12) + ("%+6.1f+-%.1f%s" % (d, s, mark)).rjust(14)
    print(line)

print()
for n, _ in RUNS:
    cs = list(runs[n].values())
    if len(cs) != len(cells):
        print("%-5s incomplete (%d/%d cells)" % (n, len(cs), len(cells))); continue
    g = sum(c[1] for c in cs); w = sum(c[0] * c[1] / 100.0 for c in cs)
    print("%-5s OVERALL %.1f%% (%d games)" % (n, 100.0 * w / g, g))
bg = sum(base[k][1] for k in cells)
print("%-5s OVERALL %.1f%% (%d games)" % ("engine",
      sum(base[k][0] for k in cells) / len(cells), bg))
print("\n* = |delta| > 1 SE,  ** = > 2 SE")
