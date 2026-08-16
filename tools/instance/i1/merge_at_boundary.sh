#!/usr/bin/env bash
# Merge the setup rules the moment round 20 returns its verdict, and not one second earlier.
#
# rules_fp is md5(dusk_plan.py + plan_filter.py). field_chain samples it at the top of a round
# and again after training, and discards the round if it moved -- correctly, because a round
# whose pilot changed mid-flight is measuring two things. So the merge has to land in the gap
# between one round's verdict and the next round's first collection, which is seconds wide;
# this waits for the verdict, stops the loop, merges, and starts it again pointed at the round
# after the one it was in.
set -u
LOG=/root/merge.log
say() { echo "[merge $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for round 20's verdict"
for _ in $(seq 1 720); do                      # up to 6 hours
    grep -aq "round 20 winner" /root/field_chain.log && break
    sleep 30
done
grep -aq "round 20 winner" /root/field_chain.log || { say "STOP: round 20 never returned"; exit 1; }
say "round 20: $(grep -a 'round 20 winner' /root/field_chain.log | tail -1)"

# field_keep would restart field_chain while the tree is half-copied, so it goes first.
for P in $(pgrep -f "[f]ield_keep.sh"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "[f]ield_chain.sh"); do kill "$P" 2>/dev/null; done
sleep 5
for P in $(pgrep -f "[f]ield_chain.sh"); do kill -9 "$P" 2>/dev/null; done
# the round's own children (collection shards, a training, a gate) outlive their parent
for P in $(pgrep -f "gate_protagonist.py --deck dragapult_dusknoir"); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f "lm_mirror_log.py"); do kill "$P" 2>/dev/null; done
sleep 3
say "loop stopped: $(pgrep -cf '[f]ield_chain.sh') field_chain, $(pgrep -cf '[f]ield_keep.sh') keep still up"

if ! python3 /root/merge_setup.py >> "$LOG" 2>&1; then
    say "MERGE FAILED -- restoring and restarting the loop unchanged"
    cp -f /root/merge_backup/tools_dusk_plan.py       /root/ptcg/repo/tools/dusk_plan.py 2>/dev/null
    cp -f /root/merge_backup/lm_plan_filter.py        /root/ptcg/repo/lm/plan_filter.py 2>/dev/null
    cp -f /root/merge_backup/tools_gate_protagonist.py /root/ptcg/repo/tools/gate_protagonist.py 2>/dev/null
fi

LAST=$(grep -aoE "field round [0-9]+" /root/field_chain.log | tail -1 | awk '{print $3}')
NEXT=$(( ${LAST:-20} + 1 ))
CUR=$(python3 - <<'PY'
import json, os
r = json.load(open("/root/ptcg/repo/models/adapters.json"))
t = (r["decks"]["dragapult_dusknoir"]["target"] or "").partition(":")[2]
print(t if t.startswith("/") else os.path.join("/root/out", t))
PY
)
say "restarting at round $NEXT with champion $CUR"
cd /root
CUR="$CUR" FROM="$NEXT" setsid --fork nohup bash /root/field_chain.sh >> /root/field_chain.log 2>&1 < /dev/null
sleep 60
setsid --fork nohup bash /root/field_keep.sh >> /root/field_keep.log 2>&1 < /dev/null
sleep 5
say "field_chain $(pgrep -cf '[f]ield_chain.sh') up, field_keep $(pgrep -cf '[f]ield_keep.sh') up"
say "MERGE_DONE"
