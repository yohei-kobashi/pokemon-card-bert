"""Per-cell trajectory across Stage-A gates: is the plateau flat, or a redistribution?

The naive test (correlate a cell's LEVEL at r3 with its r3->r6 MOVE) is biased: both
terms contain the same r3 measurement noise, so pure noise manufactures a negative
correlation. Here the level and the move are taken from DISJOINT gates so their noise
is independent.
"""
import json, os, sys, math

WORK = sys.argv[1] if len(sys.argv) > 1 else "/root/out/rlA"
GATES = [3, 6, 9, 12]
BASE = os.path.join(WORK, "baseline")


def load(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        try:
            j = json.load(open(os.path.join(d, fn)))
        except Exception:
            continue
        # two writers, two shapes: eval_rerank.py -> {"results": {...}, "overall_win_rate": pct}
        # rl_baseline_cell.py -> {"<pilot> vs <opp>": pct}
        wr = j.get("overall_win_rate")
        n = 0
        if wr is None:
            vals = [v for v in j.values() if isinstance(v, (int, float))]
            if not vals:
                continue
            wr = sum(vals) / len(vals)
        else:
            n = sum(int(v.get("games", 0)) for v in j.get("results", {}).values())
        out[fn[:-5]] = (float(wr) / 100.0, n)
    return out


base = load(BASE)
cols = {}
for r in GATES:
    cols[r] = load(os.path.join(WORK, "A_r%d_gate" % r))

cells = sorted(set(base) & set.intersection(*[set(c) for c in cols.values()]))
print("cells present in baseline and all %d gates: %d" % (len(GATES), len(cells)))
if not cells:
    print("baseline keys:", sorted(base)[:3])
    print("gate keys:", sorted(cols[GATES[0]])[:3])
    sys.exit(0)

n_lm = cols[GATES[0]][cells[0]][1]
n_bs = base[cells[0]][1]
se_gate = math.sqrt(0.25 / max(n_lm, 1)) * 100
print("games/cell: LM %d, engine baseline %d -> SE(one gate cell) = %.2fpt, "
      "SE(difference of two gates) = %.2fpt" % (n_lm, n_bs, se_gate, se_gate * math.sqrt(2)))
print()

D = {r: {c: (cols[r][c][0] - base[c][0]) * 100 for c in cells} for r in GATES}

hdr = "%-34s %7s | %8s %8s %8s %8s | %8s" % (
    "cell (pilot__opp)", "engine", "r3", "r6", "r9", "r12", "r12-r3")
print(hdr)
print("-" * len(hdr))
rows = sorted(cells, key=lambda c: D[3][c])
for c in rows:
    print("%-34s %6.1f%% | %+8.1f %+8.1f %+8.1f %+8.1f | %+8.1f" % (
        c, base[c][0] * 100, D[3][c], D[6][c], D[9][c], D[12][c], D[12][c] - D[3][c]))
print("-" * len(hdr))
for r in GATES:
    v = [D[r][c] for c in cells]
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    print("r%-3d  mean %+6.2f  sd across cells %5.2f" % (r, m, sd))
print()


def corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return 0.0, 0.0
    r = sxy / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, t


print("REDISTRIBUTION TEST -- level and move from disjoint gates (independent noise)")
print("  a strong-fall/weak-rise trade-off predicts r clearly negative")
for lvl, a, b in [(3, 6, 12), (3, 6, 9), (6, 9, 12)]:
    x = [D[lvl][c] for c in cells]
    y = [D[b][c] - D[a][c] for c in cells]
    r, t = corr(x, y)
    print("  level r%-2d  vs  move r%d->r%-2d :  r = %+.3f   t = %+.2f  (df %d)"
          % (lvl, a, b, r, t, len(cells) - 2))
print()
print("BIASED version for comparison (shared noise inflates the negative):")
for lvl, b in [(3, 6), (3, 12), (6, 12)]:
    x = [D[lvl][c] for c in cells]
    y = [D[b][c] - D[lvl][c] for c in cells]
    r, t = corr(x, y)
    print("  level r%-2d  vs  move r%d->r%-2d :  r = %+.3f   t = %+.2f"
          % (lvl, lvl, b, r, t))
print()

# how much of a cell's spread over time is bigger than measurement noise?
sd_noise = se_gate
print("PER-CELL MOVEMENT vs NOISE (|r12-r3|, noise SE %.2fpt)" % (se_gate * math.sqrt(2)))
big = [(abs(D[12][c] - D[3][c]), c) for c in cells]
big.sort(reverse=True)
thr = 2 * se_gate * math.sqrt(2)
n_big = sum(1 for v, _ in big if v > thr)
print("  cells moving more than 2 SE (%.1fpt): %d of %d  (noise alone predicts ~%.1f)"
      % (thr, n_big, len(cells), 0.046 * len(cells)))
for v, c in big[:5]:
    print("    %-34s %+6.1f" % (c, D[12][c] - D[3][c]))
