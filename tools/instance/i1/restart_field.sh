#!/usr/bin/env bash
# Restart field_chain at a ROUND BOUNDARY so the new data allocation starts cleanly.
#
# The loop is running the old field_chain.sh: bash reads a script by byte offset and the patch
# was written to a new inode, so the process in memory still has the even split. It has to be
# restarted -- but not mid-round, which would throw away the round's collection and its gate.
#
# So: wait for round $WAIT_ROUND to declare a winner, stop the loop, and start it again at the
# next round with whatever champion the registry now holds. CUR is read from the registry rather
# than assumed, because a round that ADOPTS moves it and restarting on a stale path would
# silently continue from the wrong weights.
set -u
LOG=/root/restart_field.log
WAIT_ROUND=${WAIT_ROUND:-5}
say() { echo "[restart $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for round $WAIT_ROUND to finish"
while ! grep -aqE "round $WAIT_ROUND winner:" /root/field_chain.log 2>/dev/null; do
    pgrep -f "/root/field_chain.sh" > /dev/null || { say "field_chain is gone"; break; }
    sleep 60
done
say "round $WAIT_ROUND done: $(grep -aE "round $WAIT_ROUND winner:" /root/field_chain.log | tail -1)"

for P in $(pgrep -f "/root/field_chain.sh"); do kill "$P" 2>/dev/null && say "stopped field_chain pid $P"; done
sleep 3
for P in $(pgrep -f "lm_mirror_log|dpo_branch|gate_protagonist|dusk_plan_train"); do
    kill "$P" 2>/dev/null && say "stopped in-flight worker $P"
done
sleep 5

CUR=$(python3 - <<'PY'
import json, os
r = json.load(open("/root/ptcg/repo/models/adapters.json"))
t = (r["decks"]["dragapult_dusknoir"]["target"] or "").partition(":")[2]
print(t if t.startswith("/") else os.path.join("/root/out", t))
PY
)
if [ ! -f "$CUR/model.safetensors" ]; then
    say "REFUSING to restart: registry champion $CUR has no weights"
    exit 1
fi
NEXT=$((WAIT_ROUND + 1))
say "restarting at round $NEXT with champion $CUR (allocation now ON, power 1.0)"
cd /root
CUR="$CUR" FROM="$NEXT" setsid --fork nohup bash /root/field_chain.sh >> /root/field_chain.log 2>&1 < /dev/null
sleep 5
pgrep -f "/root/field_chain.sh" > /dev/null && say "field_chain running again" || say "FAILED to restart"
