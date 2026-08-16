#!/usr/bin/env bash
# User-authorized mid-round stop for the guide-list deck revision (2026-08-16).
for P in $(pgrep -f "/root/field_chain.sh"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "effrun.sh"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "eff_audit.py"); do kill "$P" 2>/dev/null; done
sleep 8
for P in $(pgrep -f "/root/field_chain.sh"); do kill -9 "$P" 2>/dev/null; done
# also stop the round in flight (its gate/collect children reference the old list)
for P in $(pgrep -f "gate_protagonist.py --deck dragapult_dusknoir"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "lm_mirror_log.py"); do kill "$P" 2>/dev/null; done
rm -rf /root/eff; mkdir -p /root/eff
setsid nohup /root/effrun.sh > /root/eff.log 2>&1 &
echo "stopped; field_keep will revive the chain on the new list; eff audit relaunched"
