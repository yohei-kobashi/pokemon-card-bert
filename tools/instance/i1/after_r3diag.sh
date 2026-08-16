#!/usr/bin/env bash
# Wait for the round-3 diagnosis, read its verdict, and start round 4 from what it says.
#
# The user is away, so this has to make the call without them. It makes exactly two decisions,
# both from the same 5-arm 600-game gate, and it writes down why:
#
#   CHAMPION  the arm with the highest win rate, but only if it beats mrl_r2 by more than
#             CHAMP_MIN. Otherwise mrl_r2 stands -- it is the only checkpoint confirmed over
#             600 games, and picking the max of five noisy arms biases the winner up by roughly
#             1.2 SE (~3pt at SE 2.6), so a small lead is not evidence.
#   TEMP      0.25 if the sharp-label arm (r3sharp) beat the soft one (r3q), else 0.5.
#
# It also records the r3-vs-r3s spread: same recipe, same pairs, same parent, only the row order
# differs, so that difference IS the optimisation-path noise of one round. It decides nothing by
# itself but it is the number that says whether the v1 chain's swings were ever signal.
set -u
LOG=/root/r3_diag.log
say() { echo "[after_r3diag $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for R3DIAG_DONE"
while ! grep -q "R3DIAG_DONE" "$LOG" 2>/dev/null; do
    if ! pgrep -f "r3_diag.sh" >/dev/null; then
        say "r3_diag.sh is gone without R3DIAG_DONE -- refusing to start round 4 on a dead run"
        exit 1
    fi
    sleep 120
done
say "diagnosis finished"

read -r CHAMP TEMP <<<"$(python3 - <<'PY'
import json
g = json.load(open("/root/loop_dusk/r3diag/gate.json"))
arms = g.get("arms", {})
d = {k: (v.get("delta_vs_baseline"), v.get("win_rate")) for k, v in arms.items()}
CHAMP_MIN = 2.0
path = {"cur": "/root/out/mrl_r2", "r3": "/root/out/mrl_r3", "r3s": "/root/out/mrl_r3s",
        "r3q": "/root/out/mrl_r3q", "r3sharp": "/root/out/mrl_r3sharp"}
best, bd = "cur", 0.0
for k, (dd, _w) in d.items():
    if k == "cur" or dd is None:
        continue
    if dd > bd:
        best, bd = k, dd
champ = path[best] if bd > CHAMP_MIN else path["cur"]
q = (d.get("r3q") or (None,))[0]
sh = (d.get("r3sharp") or (None,))[0]
temp = "0.25" if (sh is not None and q is not None and sh > q) else "0.5"
print(champ, temp)
import sys
for k in ("cur", "r3", "r3s", "r3q", "r3sharp"):
    v = d.get(k)
    if v:
        print("  %-8s win %.1f%%  delta %+.2f" % (k, v[1], v[0]), file=sys.stderr)
r3, r3s = (d.get("r3") or (None,))[0], (d.get("r3s") or (None,))[0]
if r3 is not None and r3s is not None:
    print("  r3 vs r3s (same recipe, row order only) = %+.2fpt -- this is the round's "
          "optimisation noise" % (r3 - r3s), file=sys.stderr)
print("  champion -> %s (needed > +%.1fpt over mrl_r2)" % (champ, CHAMP_MIN), file=sys.stderr)
print("  temp -> %s" % temp, file=sys.stderr)
PY
)"
[ -n "${CHAMP:-}" ] || { say "could not read the gate -- not starting round 4"; exit 1; }
say "champion=$CHAMP temp=$TEMP"

cd /root
CUR="$CHAMP" TEMP="$TEMP" FROM=4 ROUNDS=8 setsid nohup bash /root/mirror_chain2.sh \
    >> /root/mirror_chain2.log 2>&1 < /dev/null &
sleep 5
say "round 4 launched (mirror_chain2, champion-vs-two-challengers)"
