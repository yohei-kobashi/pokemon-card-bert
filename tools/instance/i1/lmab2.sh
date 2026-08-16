#!/usr/bin/env bash
# crispin_line / duskull_shelter on the SHIPPING pilot -- the same harness as lmab.sh, because
# the energy rules just demonstrated why the engine proxy cannot be trusted for adoption:
# chg read +2.69 on engine_v2 and -3.83 on the LM. Nothing goes into WRAP_RULES on proxy
# evidence again.
#
# Baseline INCLUDES the promotion trio: that is the adoption candidate these two would join,
# so the question is "do they add on TOP of fpp", not "do they beat today's base".
# Idempotent (LMAB2_DONE) and supervised by keepd like lmab was.
set -u
LOG=/root/lmab2.log
say() { echo "[lmab2 $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
grep -aq LMAB2_DONE "$LOG" 2>/dev/null && exit 0

cd /root/ptcg/repo
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_CRISPIN=1 DUSK_SHELTER=1
B=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
F=front_dive,promote_dive,promote_line
OPPS=marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh
G=${G:-150}
DEADLINE=$(date -u -d 2026-08-15T10:00:00Z +%s)

# wait for GPU room exactly as lmab did -- the round trainer is the other tenant
while :; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${FREE:-0}" -ge 4000 ] && break
    [ "$(date -u +%s)" -ge "$DEADLINE" ] && { say "no GPU room by deadline -- going anyway"; break; }
    say "only ${FREE:-?} MiB free -- waiting"
    sleep 120
done

CKPT=$(PYTHONPATH=cg-lib:tools python3 -c \
    'from lm import registry as r; print(r.resolve("dragapult_dusknoir")["target"])' 2>/dev/null)
case "$CKPT" in
    hf:*) ;;
    *) say "champion did not resolve (got $CKPT) -- aborting"; exit 1 ;;
esac
say "champion = $CKPT, $G games/opponent, baseline includes $F"

PYTHONPATH=cg-lib python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp "$OPPS" \
    --arm "fpp=planfilter:$B,$F:$CKPT@dusk" \
    --arm "cr=planfilter:$B,$F,crispin_line:$CKPT@dusk" \
    --arm "sh=planfilter:$B,$F,duskull_shelter:$CKPT@dusk" \
    --arm "both=planfilter:$B,$F,crispin_line,duskull_shelter:$CKPT@dusk" \
    --games "$G" --seed 1 --baseline fpp --out /root/lmab2.json > /root/lmab2_run.log 2>&1
RC=$?

{
    echo "=============================================================="
    echo " CRISPIN / SHELTER ON THE SHIPPING PILOT -- $(date -u +%F\ %H:%M) UTC"
    echo " champion $CKPT, $G games x 8 opponents, fmt=dusk, paired seeds"
    echo "   fpp  = base8 + front_dive,promote_dive,promote_line (the adoption candidate)"
    echo "   cr   = fpp + crispin_line"
    echo "   sh   = fpp + duskull_shelter"
    echo "   both = fpp + both"
    echo "=============================================================="
    grep "  dragapult_dusknoir " /root/lmab2_run.log
    echo
    sed -n '/^arm /,$p' /root/lmab2_run.log
    echo
    echo "engine_v2 proxy said (for calibration): cr +2.0/+1.2, sh +1.2/0.0, both +2.8/+1.5"
    echo "(vs ogerpon, seeds 1/20000); field both +1.75 +- 0.93. chg's proxy sign flip is the"
    echo "reason this LM run exists."
} > /root/lmab2_report.txt 2>&1

say "gate exited rc=$RC; report in /root/lmab2_report.txt"
say LMAB2_DONE
