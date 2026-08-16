#!/usr/bin/env bash
# Round-boundary restart #2: pick up the human-DPO/doctrine wiring at the START of the next
# round (kills field_chain seconds into its collect; field_keep revives from the patched file).
set -u
LOG=/root/hrows_swap2.log
say() { echo "[hswap2 $(date -u +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }
CUR=$(grep -ac "field round" /root/field_chain.log)
say "waiting: markers $CUR"
for i in $(seq 1 720); do
    N=$(grep -ac "field round" /root/field_chain.log)
    if [ "$N" -gt "$CUR" ]; then
        sleep 5
        for P in $(ps -eo pid=,args= | awk "/bash \/root\/field_chain\.sh/ {print \$1}"); do
            say "next round started -- killing field_chain pid $P"
            kill "$P" 2>/dev/null
        done
        exit 0
    fi
    sleep 20
done
say "TIMEOUT 4h"
