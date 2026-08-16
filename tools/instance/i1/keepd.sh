#!/usr/bin/env bash
# Bring the unsupervised watchers back after a machine reboot.
#
# instance1 rebooted once already this week (36 days of uptime, then 47 minutes) and took the
# brancher with it; field_keep and branchd2_keep exist because of that. The scripts added today
# -- the status digest, the round-boundary restart, and instance2's night job -- had no such
# cover, and the most expensive of them is night6: if instance2 comes back without it, the GPU
# idles until morning, which has already happened twice this week.
#
# Everything restarted here is idempotent: restart_at_boundary exits if it has already fired,
# night_run re-runs its preflight and night6 resumes from whatever traces/pairs/adapters exist.
# A supervisor that can only re-run safe things needs no state of its own.
set -u
LOG=/root/keepd.log
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
I2HOST=root@175.155.64.145
I2PORT=19839
STOP=$(date -u -d 2026-08-16T23:00:00Z +%s)
say() { echo "[keepd $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
say "supervising statusd / restart_at_boundary (i1) and night6 (i2)"

while :; do
    [ "$(date -u +%s)" -ge "$STOP" ] && { say "past STOP_AFTER -- exiting"; exit 0; }

    pgrep -f "[s]tatusd.sh" >/dev/null || {
        say "statusd down -- restarting"
        setsid --fork nohup bash /root/statusd.sh >/dev/null 2>&1 </dev/null
    }
    # the two decision watchers added after this supervisor was written
    for W in hole_adopt restart_at_boundary lmab lmab2 lmab3 restart3 lmab4 restart4 lmab7 engd restart5; do
        pgrep -f "[${W:0:1}]${W:1}.sh" >/dev/null && continue
        case "$W" in
            hole_adopt)  grep -aq HOLE_ADOPT_DONE /root/hole_adopt.log 2>/dev/null && continue ;;
            restart_at_boundary) grep -aq RESTART_DONE /root/restart.log 2>/dev/null && continue ;;
            lmab) grep -aq LMAB_DONE /root/lmab.log 2>/dev/null && continue ;;
            lmab2) grep -aq LMAB2_DONE /root/lmab2.log 2>/dev/null && continue ;;
            lmab3) grep -aq LMAB3_DONE /root/lmab3.log 2>/dev/null && continue ;;
            restart3) grep -aq RESTART3_DONE /root/restart3.log 2>/dev/null && continue ;;
            lmab4) grep -aq LMAB4_DONE /root/lmab4.log 2>/dev/null && continue ;;
            restart4) grep -aq RESTART4_DONE /root/restart4.log 2>/dev/null && continue ;;
            lmab7) grep -aq LMAB7_DONE /root/lmab7.log 2>/dev/null && continue ;;
            engd) grep -aq ENGD_DONE /root/engd.log 2>/dev/null && continue ;;
            restart5) grep -aq RESTART5_DONE /root/restart5.log 2>/dev/null && continue ;;
        esac
        say "$W down and not yet fired -- restarting"
        setsid --fork nohup bash /root/$W.sh >/dev/null 2>&1 </dev/null
    done
    if false; then
        if ! grep -aq RESTART_DONE /root/restart.log 2>/dev/null; then
            say "restart_at_boundary down and not yet fired -- restarting"
            setsid --fork nohup bash /root/restart_at_boundary.sh >/dev/null 2>&1 </dev/null
        fi
    fi

    # instance2: only the night job, and only while it has not reported DONE
    R=$(ssh $I2 -p $I2PORT $I2HOST '
        if pgrep -f "[g]end2.sh" >/dev/null; then echo RUNNING
        elif [ ! -s /root/out/champion.txt ]; then echo WAITING
        else echo GONE; fi' 2>/dev/null)
    case "${R:-UNREACHABLE}" in
        RUNNING) ;;
        WAITING) say "instance2 has no champion yet -- ckptd has not delivered one" ;;
        GONE)
            say "the trace generator is down -- restarting it"
            ssh $I2 -p $I2PORT $I2HOST 'bash /root/go_gen.sh' 2>/dev/null \
                && say "restarted" || say "restart FAILED"
            ;;
        *) say "instance2 unreachable" ;;
    esac
    # the two daemons that keep the pipe alive: without ckptd the generator runs a stale
    # champion, without genpull the traces never reach the loop that needs them
    for W in ckptd genpull; do
        pgrep -f "[${W:0:1}]${W:1}.sh" >/dev/null || {
            say "$W down -- restarting"
            setsid --fork nohup bash /root/$W.sh >/dev/null 2>&1 </dev/null
        }
    done
    sleep 300
done
