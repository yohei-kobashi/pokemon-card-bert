#!/usr/bin/env bash
for P in $(pgrep -f '/root/gend2.sh'); do kill "$P" 2>/dev/null; done
sleep 5
for P in $(pgrep -f 'gen_selfplay'); do kill "$P" 2>/dev/null; done
sleep 3
cd /root
setsid nohup bash /root/gend2.sh > /dev/null 2>&1 < /dev/null &
sleep 2
echo gend2_procs=$(pgrep -cf '/root/gend2.sh')
