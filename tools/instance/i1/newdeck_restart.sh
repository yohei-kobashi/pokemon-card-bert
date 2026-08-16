#!/usr/bin/env bash
echo "[lmab6 $(date -u +%m-%d_%H:%M:%S)] superseded by the pokehubguide deck revision" >> /root/lmab6.log
echo LMAB6_DONE >> /root/lmab6.log
for P in $(pgrep -f "lmab6.sh"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "gate_protagonist"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "/root/field_chain.sh"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "lm_mirror_log"); do kill "$P" 2>/dev/null; done
sleep 8
for P in $(pgrep -f "/root/field_chain.sh"); do kill -9 "$P" 2>/dev/null; done
chmod +x /root/lmab7.sh
setsid nohup /root/lmab7.sh > /dev/null 2>&1 < /dev/null &
echo "restarted: field_keep revives the chain on the pokehubguide list; lmab7 queued"
