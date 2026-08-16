#!/usr/bin/env bash
# Restart sub4_prep so it waits for the NEW build time (23:05Z).
set -u
for P in $(ps -eo pid=,args= | awk "/bash \/root\/sub4_prep\.sh/ {print \$1}"); do
    echo "killing sub4_prep pid $P"
    kill "$P" 2>/dev/null
done
sleep 2
setsid --fork nohup bash /root/sub4_prep.sh >> /root/sub4_run.log 2>&1 < /dev/null
sleep 2
ps -eo pid=,args= | awk "/bash \/root\/sub4_prep\.sh/ {print \$1}"
