#!/usr/bin/env bash
# Supervisor for branchd2. instance1 rebooted at ~06:50 on 2026-08-12 (36 days of uptime, then
# 47 minutes) and took the brancher with it. Nothing noticed: instance2 kept collecting, kept
# requesting, and each deck sat for its full 2h timeout before falling back. A daemon that only
# ever dies with the machine still needs a supervisor, because the machine does die.
set -u
while :; do
    pgrep -f "[b]ranchd2.sh" >/dev/null || {
        echo "[keep $(date -u +%m-%d_%H:%M:%S)] branchd2 not running -- starting it"
        setsid nohup bash /root/branchd2.sh >> /root/branchd2.log 2>&1 < /dev/null &
    }
    sleep 120
done
