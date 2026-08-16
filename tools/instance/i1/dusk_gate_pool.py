"""Pool gate_protagonist shards and report each arm as a PAIRED difference from engine_v2.

Pooling win rates would be enough for a point estimate and useless for deciding anything: the
arms played the same shuffles from the same seats, so the games are matched and the difference
has a far smaller standard error than the two rates do separately. That only survives pooling
if the per-game vectors are lined up, which is why the shards write them out.

A shard whose process died is simply absent, and its opponents drop out of every arm together
-- never out of one.

    python dusk_gate_pool.py <outdir>
"""
import glob, json, math, os, sys

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "shard*.json")))
if not files:
    raise SystemExit("no shard json in %s" % d)

cells = {}          # (arm, opp) -> [0/1, ...]
arms, opps = [], []
for fn in files:
    j = json.load(open(fn))
    for key, v in j["cells"].items():
        arm, opp = key.split("|", 1)
        cells[(arm, opp)] = v["raw"]
        if arm not in arms:
            arms.append(arm)
        if opp not in opps:
            opps.append(opp)
print("pooled %d shards | %d arms | %d opponents" % (len(files), len(arms), len(opps)))

BASE = "engine"
# Only opponents every arm actually played. One arm missing an opponent would otherwise be
# scored against a baseline that still includes it.
opps = [o for o in opps if all((a, o) in cells for a in arms)]

print("\n%-22s %s" % ("opponent", "  ".join("%8s" % a for a in arms)))
for o in opps:
    row = []
    for a in arms:
        v = cells[(a, o)]
        row.append("%7.1f%%" % (100.0 * sum(v) / max(1, len(v))))
    print("%-22s %s" % (o, "  ".join("%8s" % x for x in row)))

print("\n%-8s %8s %9s %8s %8s" % ("arm", "win%", "delta", "se", "t"))
for a in arms:
    mine = [x for o in opps for x in cells[(a, o)]]
    base = [x for o in opps for x in cells[(BASE, o)]]
    diffs = [x - y for x, y in zip(mine, base)]
    wr = 100.0 * sum(mine) / max(1, len(mine))
    m = sum(diffs) / max(1, len(diffs))
    if len(diffs) > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
        se = 100.0 * sd / math.sqrt(len(diffs))
    else:
        se = float("nan")
    t = (100.0 * m / se) if se else float("nan")
    print("%-8s %7.1f%% %+8.2f %8.2f %8.2f%s"
          % (a, wr, 100.0 * m, se, t, "   (baseline)" if a == BASE else ""))
print("\n%d games per arm" % len([x for o in opps for x in cells[(arms[0], o)]]))

# ---- the verdict the overnight branch reads ----------------------------------------------
# s1 against r8 DIRECTLY, not each against engine_v2 and then subtracted: both arms played the
# same seeds from the same seats, so the difference that matters is paired at the game level and
# comparing two engine-relative deltas throws that pairing away.
if "s1" in arms and "r8" in arms:
    s1 = [x for o in opps for x in cells[("s1", o)]]
    r8 = [x for o in opps for x in cells[("r8", o)]]
    diffs = [x - y for x, y in zip(s1, r8)]
    m = sum(diffs) / max(1, len(diffs))
    sd = (sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5 if len(diffs) > 1 else 0.0
    se = 100.0 * sd / math.sqrt(len(diffs)) if diffs else float("nan")
    delta = 100.0 * m
    t = delta / se if se else 0.0
    # Degraded on EITHER reading: a statistically clear loss, or a point estimate worse than
    # 2pt that merely has not reached significance yet. `rl-gate-is-noisier-than-assumed` --
    # the same checkpoint re-scored 2.6pt apart -- is why a small insignificant dip alone is
    # not called degradation, and why a large one is not waved through for lacking a t.
    degraded = bool((t <= -2.0) or (delta <= -2.0))
    print("\ns1 - r8 (paired): %+.2f pt  se %.2f  t %.2f  ->  %s"
          % (delta, se, t, "DEGRADED" if degraded else "NO DEGRADATION"))
    json.dump({"delta_s1_minus_r8": delta, "se": se, "t": t, "degraded": degraded,
               "games": len(diffs), "opponents": opps},
              open(os.path.join(d, "verdict.json"), "w"), indent=1)
