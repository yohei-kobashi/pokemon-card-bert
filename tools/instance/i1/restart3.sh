#!/usr/bin/env bash
# Bring the promotion trio (front_dive,promote_dive,promote_line, +4.75 on the shipping
# pilot, user-adopted 2026-08-15) into force at the NEXT round boundary.
#
# field_chain.sh was already replaced with a new inode carrying the trio in WRAP_RULES and
# DUSK_FRONT_DIVE=1; the round in flight still reads the old file. Between rounds the chain
# has no children, so killing the script alone is enough -- and UNLIKE restart2, nothing
# else is touched: field_keep survives and revives the chain within ~120s with the champion
# and round number it reads itself.
set -u
LOG=/root/restart3.log
say() { echo "[restart3 $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
grep -aq RESTART3_DONE "$LOG" 2>/dev/null && exit 0
BASE=$(grep -ac 'winner' /root/field_chain.log 2>/dev/null || echo 0)
say "waiting for the NEXT 'winner' line (currently $BASE) -- then the trio takes effect"
for _ in $(seq 1 720); do
    NOW=$(grep -ac 'winner' /root/field_chain.log 2>/dev/null || echo 0)
    [ "$NOW" -gt "$BASE" ] && break
    grep -aq STOP_AFTER_REACHED /root/field_chain.log 2>/dev/null && break
    sleep 30
done
NOW=$(grep -ac 'winner' /root/field_chain.log 2>/dev/null || echo 0)
if [ "$NOW" -le "$BASE" ]; then say 'STOP: no round boundary appeared'; exit 1; fi
say "boundary: $(grep -a 'winner' /root/field_chain.log | tail -1)"
for P in $(pgrep -f '/root/field_chain.sh'); do kill "$P" 2>/dev/null; done
sleep 8
for P in $(pgrep -f '/root/field_chain.sh'); do kill -9 "$P" 2>/dev/null; done
say 'field_chain stopped at the boundary; field_keep will revive it with the trio'
say RESTART3_DONE
