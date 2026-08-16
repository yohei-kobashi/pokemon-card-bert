#!/usr/bin/env bash
# Bring MIN_GAIN=0.0 into force at the next round boundary.
#
# field_chain.sh was replaced with a new inode, so the round in flight is still reading the old
# file and still applying the +1.0pt bar. Restarting between rounds is what swaps it -- and
# between rounds is also the only safe moment, because a restart mid-round throws away that
# round's collection and training.
#
# The kill list is DELIBERATELY narrow. The merge watcher earlier killed every
# `gate_protagonist --deck dragapult_dusknoir`, which also took out the two hole-gate shards
# running beside it. Between rounds field_chain has no children, so killing the script alone is
# enough and nothing else on the box is touched.
set -u
LOG=/root/restart2.log
say() { echo "[restart $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
WANT=${WANT:-25}

# Idempotent: a supervisor (or a reboot) may start this again after it has already fired,
# and a second firing would kill a round that is mid-collection for no reason.
grep -aq RESTART_DONE "$LOG" 2>/dev/null && { say "already done -- nothing to do"; exit 0; }
say "waiting for round $WANT's verdict (then MIN_GAIN=0.0 + PHI_MIN=0.10 take effect)"
for _ in $(seq 1 720); do
    grep -aq "round $WANT winner" /root/field_chain.log && break
    sleep 30
done
grep -aq "round $WANT winner" /root/field_chain.log || { say "STOP: round $WANT never returned"; exit 1; }
say "$(grep -a "round $WANT winner" /root/field_chain.log | tail -1)"

for P in $(pgrep -f "[f]ield_keep.sh"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "[f]ield_chain.sh"); do kill "$P" 2>/dev/null; done
sleep 8
for P in $(pgrep -f "[f]ield_chain.sh"); do kill -9 "$P" 2>/dev/null; done

LAST=$(grep -aoE "field round [0-9]+" /root/field_chain.log | tail -1 | awk '{print $3}')
NEXT=$(( ${LAST:-$WANT} + 1 ))
CUR=$(python3 - <<'PY'
import json, os
r = json.load(open("/root/ptcg/repo/models/adapters.json"))
t = (r["decks"]["dragapult_dusknoir"]["target"] or "").partition(":")[2]
print(t if t.startswith("/") else os.path.join("/root/out", t))
PY
)
say "restarting at round $NEXT with $CUR, MIN_GAIN=0.0"
cd /root
MIN_GAIN=0.0 CUR="$CUR" FROM="$NEXT" setsid --fork nohup bash /root/field_chain.sh \
    >> /root/field_chain.log 2>&1 < /dev/null
sleep 60
setsid --fork nohup bash /root/field_keep.sh >> /root/field_keep.log 2>&1 < /dev/null
sleep 5
say "field_chain $(pgrep -cf '[f]ield_chain.sh') up, keep $(pgrep -cf '[f]ield_keep.sh') up"
say "RESTART_DONE"
