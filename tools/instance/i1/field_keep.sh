#!/usr/bin/env bash
# Restart field_chain if it dies overnight, at the NEXT round with the registry's champion.
#
# field_chain exits on several guards (disk, GPU held too long, a failed train). Every one of
# them is recoverable by starting the next round, and with nobody watching until morning an
# exit costs the whole night -- roughly a third of the time left before STOP_AFTER.
# CUR is read from models/adapters.json, never assumed: a round that ADOPTED moves the
# champion, and restarting on a stale path would silently continue from the wrong weights.
set -u
LOG=/root/field_keep.log
say() { echo "[keep $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
say "supervising field_chain"
while :; do
    if pgrep -f "/root/field_chain.sh" > /dev/null; then sleep 120; continue; fi
    if [ "$(date -u +%s)" -ge "$(date -u -d 2026-08-16T23:00:00Z +%s)" ]; then
        say "past STOP_AFTER -- not restarting"; exit 0
    fi
    LAST=$(grep -aoE "field round [0-9]+" /root/field_chain.log 2>/dev/null | tail -1 | awk '{print $3}')
    NEXT=$(( ${LAST:-1} + 1 ))
    CUR=$(python3 - <<'PY'
import json, os
r = json.load(open("/root/ptcg/repo/models/adapters.json"))
t = (r["decks"]["dragapult_dusknoir"]["target"] or "").partition(":")[2]
print(t if t.startswith("/") else os.path.join("/root/out", t))
PY
)
    if [ ! -f "$CUR/model.safetensors" ]; then say "champion $CUR has no weights -- NOT restarting"; sleep 600; continue; fi
    say "field_chain is down; restarting at round $NEXT with $CUR"
    cd /root
    CUR="$CUR" FROM="$NEXT" setsid --fork nohup bash /root/field_chain.sh >> /root/field_chain.log 2>&1 < /dev/null
    sleep 90
done
