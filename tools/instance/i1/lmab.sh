#!/usr/bin/env bash
# The promotion + energy plan rules, measured on the PILOT THAT SHIPS.
#
# Everything measured so far ranks with engine_v2 (`planfilter:<rules>:engine`), which is a
# proxy: the submission is the LM, and a rule that helps a heuristic ranker need not help a
# trained one. So this repeats the same four arms with the champion checkpoint and the `dusk`
# prompt format -- @dusk is not optional, running these arms at fmt=prompt once scored the same
# checkpoint 17.9% instead of 35.2% and erased the effect being measured.
#
# Unattended by construction: it waits for round 30's verdict, resolves the champion AFTER it
# (so it tests what the round actually left us), waits for room on the GPU, and is idempotent --
# keepd may relaunch it and it will exit immediately once it has written LMAB_DONE.
set -u
LOG=/root/lmab.log
say() { echo "[lmab $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
grep -aq LMAB_DONE "$LOG" 2>/dev/null && exit 0

cd /root/ptcg/repo
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1
B=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
F=front_dive,promote_dive,promote_line
E=energy_line,energy_focus
# the SAME eight the loop gates on, so these numbers sit beside the round table rather than
# beside nothing
OPPS=marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh
G=${G:-150}
DEADLINE=$(date -u -d 2026-08-15T06:00:00Z +%s)   # go anyway rather than wait forever

say "start. waiting for the CPU verification, then for round 30"
until grep -aq PLANRULE_DONE /root/planrule_verify.log 2>/dev/null; do
    [ "$(date -u +%s)" -ge "$DEADLINE" ] && { say "past the wait deadline -- going anyway"; break; }
    sleep 60
done
say "CPU verification: done (or waited out)"

until grep -q "round 30 winner:" /root/field*.log 2>/dev/null; do
    [ "$(date -u +%s)" -ge "$DEADLINE" ] && { say "round 30 never landed -- going anyway"; break; }
    sleep 60
done
say "round 30 verdict: $(grep -h 'round 30 winner:' /root/field*.log 2>/dev/null | tail -1)"

# The round-31 training is the other tenant on this card. Take the GPU only when there is room.
while :; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${FREE:-0}" -ge 4000 ] && break
    [ "$(date -u +%s)" -ge "$DEADLINE" ] && { say "no GPU room by the deadline -- going anyway"; break; }
    say "only ${FREE:-?} MiB free -- waiting for room"
    sleep 120
done

CKPT=$(PYTHONPATH=cg-lib:tools python3 -c \
    'from lm import registry as r; print(r.resolve("dragapult_dusknoir")["target"])' 2>/dev/null)
case "$CKPT" in
    hf:*) ;;
    *) say "could not resolve the champion (got %s) -- aborting so nothing measures the wrong model"
       say "CKPT=$CKPT"; exit 1 ;;
esac
say "champion = $CKPT, $G games/opponent over $OPPS"

PYTHONPATH=cg-lib python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp "$OPPS" \
    --arm "base=planfilter:$B:$CKPT@dusk" \
    --arm "fpp=planfilter:$B,$F:$CKPT@dusk" \
    --arm "chg=planfilter:$B,$E:$CKPT@dusk" \
    --arm "fppchg=planfilter:$B,$F,$E:$CKPT@dusk" \
    --games "$G" --seed 1 --baseline base --out /root/lmab.json > /root/lmab_run.log 2>&1
RC=$?

{
    echo "=============================================================="
    echo " PLAN RULES ON THE SHIPPING PILOT -- $(date -u +%F\ %H:%M) UTC"
    echo " champion $CKPT, $G games x 8 opponents, fmt=dusk, paired seeds"
    echo "   base   = the eight rules the loop runs today"
    echo "   fpp    = + front_dive, promote_dive, promote_line"
    echo "   chg    = + energy_line, energy_focus"
    echo "   fppchg = + all five"
    echo "=============================================================="
    grep "  dragapult_dusknoir " /root/lmab_run.log
    echo
    sed -n '/^arm /,$p' /root/lmab_run.log
    echo
    echo "for comparison, the same four arms ranked by engine_v2 instead of the LM:"
    for S in /root/planrule/s*.json; do
        echo "  $(basename "$S" .json): $(sed -n '/^arm /,$p' "${S%.json}.log" | tr '\n' ' ')"
    done
} > /root/lmab_report.txt 2>&1

say "gate exited rc=$RC; report in /root/lmab_report.txt"
say LMAB_DONE
