#!/usr/bin/env bash
# After restart4 fires, the OLD chain may have spawned one more collect python in the
# 30s poll race (it happened at restart3: round-35 ran to completion on the old wrap).
# Wait for the marker, then kill any dragapult planfilter python whose spec lacks the
# just-adopted lethal_boss.
until grep -aq RESTART4_DONE /root/restart4.log 2>/dev/null; do sleep 20; done
sleep 30
for P in $(pgrep -f 'setup_search,front_dive,promote_dive,promote_line:hf'); do
    if ! tr '\0' ' ' < /proc/$P/cmdline | grep -q lethal_boss; then
        kill "$P" 2>/dev/null && echo "[orphan4] killed $P" >> /root/restart4.log
    fi
done
echo '[orphan4] done' >> /root/restart4.log
