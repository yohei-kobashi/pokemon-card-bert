#!/usr/bin/env bash
# Read the 2x2 factorial when it lands, adopt the best positive arm, and put it into the loop.
#
# The criterion is fixed in advance (user directive 08-14: adopt on any positive delta), so this
# needs no judgement at the moment it fires -- which is the point. Nobody has to be awake at
# 13:15 UTC for the result to reach the next round.
#
#   arms   ss (baseline) | en = +energy_line,energy_focus | pd = +phantom_dive | both = +all
#   pick   the largest delta above zero; ties go to the smaller rule set, because a rule that
#          buys nothing is a rule that can still cost something later
#   apply  append its rules to WRAP_RULES, restart field_chain at the NEXT round boundary
#
# It waits for restart_at_boundary to have fired first. Two watchers restarting the loop within
# minutes of each other would have the second one kill a round the first had just started.
set -u
LOG=/root/hole_adopt.log
say() { echo "[adopt $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
grep -aq HOLE_ADOPT_DONE "$LOG" 2>/dev/null && { say "already done"; exit 0; }

say "waiting for both hole shards"
for _ in $(seq 1 240); do
    pgrep -f "[h]ole_launch.sh" >/dev/null || break
    sleep 30
done
[ -s /root/hole_1.json ] && [ -s /root/hole_2.json ] || { say "STOP: a shard produced no json"; exit 1; }

WIN=$(python3 - <<'PY'
import glob, json, math, sys
cells = {}
for f in sorted(glob.glob("/root/hole_*.json")):
    cells.update(json.load(open(f))["cells"])
opps = sorted({k.split("|", 1)[1] for k in cells})
def vec(a):
    out = []
    for o in opps:
        c = cells.get("%s|%s" % (a, o))
        if c is None:
            return None
        out += c["raw"]
    return out
base = vec("ss")
rows = []
for a in ("en", "pd", "both"):
    v = vec(a)
    if not v or not base or len(v) != len(base):
        continue
    d = [x - y for x, y in zip(v, base)]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    se = 100 * sd / math.sqrt(len(d))
    rows.append((a, 100 * m, se, len(d)))
    print("%-5s delta %+6.2f +- %.2f  n=%d" % (a, 100 * m, se, len(d)), file=sys.stderr)
# largest positive delta; ties (within 0.25pt) go to the smaller rule set
rows.sort(key=lambda r: (-r[1], {"en": 0, "pd": 1, "both": 2}[r[0]]))
best = [r for r in rows if r[1] > 0]
if best and len(best) > 1 and best[0][0] == "both" and best[1][1] > best[0][1] - 0.25:
    best = best[1:]
print(best[0][0] if best else "none")
PY
2>>"$LOG")
say "winner: $WIN"

case "$WIN" in
    en)   ADD="energy_line,energy_focus" ;;
    pd)   ADD="phantom_dive" ;;
    both) ADD="energy_line,energy_focus,phantom_dive" ;;
    *)    say "no arm was positive -- WRAP_RULES unchanged"; say "HOLE_ADOPT_DONE"; exit 0 ;;
esac

# restart_at_boundary owns the next boundary; wait it out rather than race it
for _ in $(seq 1 240); do
    grep -aq RESTART_DONE /root/restart.log 2>/dev/null && break
    sleep 30
done

python3 - "$ADD" <<'PY' >> "$LOG" 2>&1
import os, sys
p = "/root/field_chain.sh"
s = open(p).read()
old = "WRAP_RULES=${WRAP_RULES:-lethal_now,$PROH,search_bottom,setup_search}"
new = "WRAP_RULES=${WRAP_RULES:-lethal_now,$PROH,search_bottom,setup_search,%s}" % sys.argv[1]
assert s.count(old) == 1, "WRAP_RULES anchor"
open(p + ".n", "w").write(s.replace(old, new))
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("WRAP_RULES now ends with %s" % sys.argv[1])
PY

LAST=$(grep -aoE "field round [0-9]+" /root/field_chain.log | tail -1 | awk '{print $3}')
say "patched; waiting for round $LAST to return before restarting"
for _ in $(seq 1 480); do
    grep -aq "round $LAST winner" /root/field_chain.log && break
    sleep 30
done
for P in $(pgrep -f "[f]ield_keep.sh"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "[f]ield_chain.sh"); do kill "$P" 2>/dev/null; done
sleep 8
for P in $(pgrep -f "[f]ield_chain.sh"); do kill -9 "$P" 2>/dev/null; done
NEXT=$(( $(grep -aoE "field round [0-9]+" /root/field_chain.log | tail -1 | awk '{print $3}') + 1 ))
CUR=$(python3 - <<'PY'
import json, os
r = json.load(open("/root/ptcg/repo/models/adapters.json"))
t = (r["decks"]["dragapult_dusknoir"]["target"] or "").partition(":")[2]
print(t if t.startswith("/") else os.path.join("/root/out", t))
PY
)
cd /root
MIN_GAIN=0.0 CUR="$CUR" FROM="$NEXT" setsid --fork nohup bash /root/field_chain.sh \
    >> /root/field_chain.log 2>&1 < /dev/null
sleep 60
setsid --fork nohup bash /root/field_keep.sh >> /root/field_keep.log 2>&1 < /dev/null
say "restarted at round $NEXT with $ADD in WRAP_RULES"
say "HOLE_ADOPT_DONE"
