#!/usr/bin/env bash
# Wait for the NEXT round to begin (its collect just started, <1 min of lost work), then kill
# the running field_chain so field_keep revives it from the PATCHED file (human-rows append).
# Run from a file: an inline pattern naming field_chain would match the ssh command itself.
set -u
LOG=/root/hrows_swap.log
say() { echo "[hswap $(date -u +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }
CUR=$(grep -ac "field round" /root/field_chain.log)
say "waiting: rounds seen so far $CUR"
for i in $(seq 1 720); do
    N=$(grep -ac "field round" /root/field_chain.log)
    if [ "$N" -gt "$CUR" ]; then
        sleep 5
        for P in $(ps -eo pid=,args= | awk "/bash \/root\/field_chain\.sh/ {print \$1}"); do
            say "new round started -- killing field_chain pid $P (field_keep revives with patched file)"
            kill "$P" 2>/dev/null
        done
        exit 0
    fi
    sleep 20
done
say "TIMEOUT: no new round in 4h"
