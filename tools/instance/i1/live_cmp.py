"""crustle_stall vs the live field: engine_v2 against each LM, live-frequency weighted.

The historical 3-opponent protocol covers 17.8% of the top-500 ladder and omits the two most
common decks. This weights each matchup by how often it actually occurs, so an edge against a
4.4%-of-field opponent cannot carry the verdict.
"""
import json, math, os, sys

W = {"alakazam_nz": 0.212, "marnie_grimmsnarl": 0.172, "archaludon": 0.094,
     "alakazam": 0.086, "cynthia_garchomp": 0.048, "crustle": 0.048, "dragapult": 0.044}
DECK = "crustle_stall"
GAMES = 300
RUNS = [("v35", "/root/out/wr_v35_live"), ("v36", "/root/out/wr_v36_live")]


def norm(k):
    if " vs " in k:
        l, r = k.split(" vs ", 1)
        return l.split(" (")[0].strip() + " vs " + r.split(" (")[0].strip()
    return k


def se(p, n):
    p /= 100.0
    return 100.0 * math.sqrt(max(p * (1 - p), 1e-9) / n)


base = {}
for f in sorted(os.listdir("/root/out/base_live")):
    if f.endswith(".json"):
        base.update({norm(k): v
                     for k, v in json.load(open("/root/out/base_live/" + f)).items()})

runs = {}
for name, d in RUNS:
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                for k, v in json.load(open(os.path.join(d, f))).get("results", {}).items():
                    out[norm(k)] = 100.0 * v["win"] / v["games"]
    runs[name] = out

have = [n for n, _ in RUNS if len(runs[n]) == len(W)]
hdr = "%-20s %7s %8s" % ("opponent", "live w", "engine")
for n in have:
    hdr += "%9s%15s" % (n, "delta +- SE")
print("%s vs the LIVE FIELD (%d games/cell)" % (DECK, GAMES))
print(hdr)

acc = {n: 0.0 for n in have}
accb = 0.0
for o, w in sorted(W.items(), key=lambda x: -x[1]):
    k = "%s vs %s" % (DECK, o)
    b = base.get(k)
    line = "%-20s %7.3f %8.1f" % (o, w, b)
    accb += w * b
    for n in have:
        l = runs[n][k]
        acc[n] += w * l
        s = math.hypot(se(b, GAMES), se(l, GAMES))
        d = l - b
        mark = "**" if abs(d) > 2 * s else ("*" if abs(d) > s else "")
        line += "%9.1f%15s" % (l, "%+.1f+-%.1f%s" % (d, s, mark))
    print(line)

tot = sum(W.values())
n_all = GAMES * len(W)
print()
print("LIVE-WEIGHTED (top-7 = %.1f%% of field)" % (100 * tot))
print("  engine_v2 %.1f%%" % (accb / tot))
for n in have:
    print("  %-9s %.1f%%   delta %+.1fpt" % (n, acc[n] / tot, (acc[n] - accb) / tot))
ub = sum(base["%s vs %s" % (DECK, o)] for o in W) / len(W)
print()
print("UNWEIGHTED mean of the 7 (%d games each side)" % n_all)
print("  engine_v2 %.1f%%" % ub)
for n in have:
    ul = sum(runs[n]["%s vs %s" % (DECK, o)] for o in W) / len(W)
    s = math.hypot(se(ub, n_all), se(ul, n_all))
    print("  %-9s %.1f%%   delta %+.1f +- %.1f  (%.1f SE)" % (n, ul, ul - ub, s,
                                                              abs(ul - ub) / s))
missing = [n for n, _ in RUNS if n not in have]
if missing:
    print("\npending: " + ", ".join("%s (%d/%d cells)" % (n, len(runs[n]), len(W))
                                    for n in missing))
