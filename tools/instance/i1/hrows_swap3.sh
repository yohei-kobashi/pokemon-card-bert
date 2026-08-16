#!/usr/bin/env bash
# Boundary restart #3: small-round regime (user directive -- human 4-deck data is the MAIN
# signal; i2/self-play data cut down so more rounds fit before 23:00Z).
set -u
LOG=/root/hrows_swap3.log
say() { echo "[hswap3 $(date -u +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }
CUR=$(grep -ac "field round" /root/field_chain.log)
say "waiting: markers $CUR"
for i in $(seq 1 360); do
    N=$(grep -ac "field round" /root/field_chain.log)
    if [ "$N" -gt "$CUR" ]; then
        sleep 5
        for P in $(ps -eo pid=,args= | awk "/bash \/root\/field_chain\.sh/ {print \$1}"); do
            say "next round started -- killing field_chain pid $P"
            kill "$P" 2>/dev/null
        done
        exit 0
    fi
    sleep 15
done
say "TIMEOUT"
