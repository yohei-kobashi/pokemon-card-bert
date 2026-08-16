#!/usr/bin/env bash
# One place to look, every 10 minutes, for both machines.
#
# WHY. Two failures this week were invisible until someone happened to look: night4b died at
# 15:10 and the GPU idled until 23:45, and instance2 sat idle another 1h45 after night5 finished.
# Neither was a crash anybody would have seen -- the logs looked fine, there was simply nothing
# running. So the digest reports what is RUNNING, not just what is written, and says IDLE out
# loud when a machine has nothing to do.
#
#   tail -f /root/status.log        follow it
#   grep ALERT /root/status.log     only the things that need a human
set -u
LOG=/root/status.log
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
I2HOST=root@175.155.64.145
I2PORT=19839
EVERY=${EVERY:-600}

say() { echo "$*" >> "$LOG"; }

# how much a gate log has produced, and whether it is still moving
cells() {   # $1 log
    local n age
    n=$(grep -ac " vs " "$1" 2>/dev/null); n=${n:-0}
    age=$(( ($(date -u +%s) - $(stat -c %Y "$1" 2>/dev/null || echo 0)) / 60 ))
    if [ "$age" -gt 25 ]; then
        echo "$n cells, STALE ${age}m"
    else
        echo "$n cells, ${age}m ago"
    fi
}

while :; do
    say "[status $(date -u +%m-%d_%H:%M)] ------------------------------------------------"
    ALERTS=""

    # ---------------- instance1 ----------------
    L=$(awk '{print $1}' /proc/loadavg)
    G=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')
    D=$(df -BG --output=avail /root | tail -1 | tr -d ' ')
    say "i1  load $L  gpu $G  disk $D"
    [ "${D%G}" -lt 8 ] && ALERTS="$ALERTS i1-disk-${D}"

    if pgrep -f "[f]ield_chain.sh" >/dev/null; then
        R=$(grep -aoE "field round [0-9]+" /root/field_chain.log | tail -1)
        W=$(grep -aoE "round [0-9]+ winner: [a-z]+" /root/field_chain.log | tail -1)
        say "    field_chain  $R | last verdict: ${W:-none yet}"
    else
        say "    field_chain  NOT RUNNING"
        pgrep -f "[f]ield_keep.sh" >/dev/null || ALERTS="$ALERTS field_chain+keep-both-down"
    fi
    pgrep -f "[m]erge_at_boundary.sh" >/dev/null \
        && say "    merge        armed: $(tail -1 /root/merge.log 2>/dev/null | cut -c1-70)" \
        || { grep -aq MERGE_DONE /root/merge.log 2>/dev/null && say "    merge        DONE"; }
    for i in 1 2; do
        [ -f /root/hole_$i.log ] && {
            C=$(cells /root/hole_$i.log)
            if pgrep -f "[h]ole_launch.sh" >/dev/null; then
                say "    hole gate s$i $C"
                case "$C" in *STALE*) ALERTS="$ALERTS hole-s$i-stale";; esac
            else
                say "    hole gate s$i $C (launcher exited -- finished)"
            fi
        }
    done

    # ---------------- instance2 ----------------
    I2OUT=$(ssh $I2 -p $I2PORT $I2HOST '
        echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d " ")"
        echo "disk $(df -BG --output=avail /root | tail -1 | tr -d " ")"
        echo "procs $(pgrep -cf "night[0-9]|lm_mirror_log|dpo_teacher|gate_protagonist")"
        for i in 1 2; do
            for J in /root/pf4b_$i.log /root/gate_night6.log; do
                [ -f "$J" ] && echo "$(basename $J) $(grep -ac " vs " "$J") cells $(( ($(date -u +%s) - $(stat -c %Y "$J")) / 60 ))m"
            done
        done
        tail -1 /root/night5.log 2>/dev/null | cut -c1-70' 2>/dev/null)
    if [ -z "$I2OUT" ]; then
        say "i2  UNREACHABLE"
        ALERTS="$ALERTS i2-unreachable"
    else
        say "i2  $(echo "$I2OUT" | tr '\n' ' | ' | cut -c1-190)"
        NP=$(echo "$I2OUT" | awk '/^procs/{print $2}')
        [ "${NP:-0}" -eq 0 ] && ALERTS="$ALERTS i2-IDLE"
    fi

    [ -n "$ALERTS" ] && say "ALERT:$ALERTS"
    sleep "$EVERY"
done
