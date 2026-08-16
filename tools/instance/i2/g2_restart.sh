#!/usr/bin/env bash
# Restart gend2 so it reads the NEW stop date (23:00Z). Run from a file to avoid self-match.
set -u
for P in $(ps -eo pid=,args= | awk '/bash \/root\/gend2\.sh/ {print $1}'); do
    echo "killing gend2 pid $P"
    kill "$P" 2>/dev/null
done
sleep 3
setsid --fork nohup bash /root/gend2.sh >> /root/gend2.log 2>&1 < /dev/null
sleep 2
ps -eo pid=,args= | awk '/bash \/root\/gend2\.sh/ {print $1}'
