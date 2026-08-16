#!/usr/bin/env bash
# Restart the two SUPERVISORS so they pick up the new STOP date (2026-08-16T03:00:00Z).
#
# Run from a FILE, never as `ssh host '...'`: the kill patterns have to name the scripts, and a
# command line that names them matches ITSELF. Doing this inline killed the ssh session halfway
# through a moment ago, leaving keepd down and field_keep still on the old date. A script's
# own /proc cmdline is just `bash /root/restart_sup.sh`, so nothing self-matches.
#
# field_chain is deliberately NOT restarted. Its STOP_AFTER is already bound in the running
# process, so it will exit at a ROUND BOUNDARY at 12:00Z today with no work in flight, and
# field_keep -- by then on the new date -- brings it back within 120 s at the next round.
set -u
LOG=/root/restart_sup.log
say() { echo "[sup $(date -u +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }

for NAME in keepd field_keep; do
    for P in $(ps -eo pid=,args= | awk -v n="$NAME" '$0 ~ ("/root/" n "\\.sh") {print $1}'); do
        say "killing $NAME pid $P"
        kill "$P" 2>/dev/null
    done
done
sleep 4

cd /root
setsid --fork nohup bash /root/keepd.sh > /dev/null 2>&1 < /dev/null
setsid --fork nohup bash /root/field_keep.sh >> /root/field_keep.log 2>&1 < /dev/null
sleep 8

say "--- running now ---"
ps -eo pid=,etime=,args= | awk '/\/root\/(field_chain|field_keep|keepd|lmab|ckptd|genpull|statusd)\.sh/ {printf "  %-8s %-10s %s\n", $1, $2, $NF}' | tee -a "$LOG"
say "--- stop date bound in each file ---"
for f in field_chain field_keep keepd; do
    printf "  %-14s %s\n" "$f" "$(grep -o '2026-08-1[0-9]T[0-9:]*Z' "/root/$f.sh" | head -1)" | tee -a "$LOG"
done
