#!/usr/bin/env bash
# The ogerpon race package on the SHIPPING pilot. lmab2 closed crispin_line (global: -8.0 on
# the mirror) and duskull_shelter (+0.7 vs ogerpon) as written; the audits say the matchup is
# fuel ({R}{P} banked in 24% of games, 0.35-0.46 dives/game) plus the 210=200+10 arithmetic.
# Four arms so fuel and chip attribute separately:
#   fpp  = base8 + promotion trio (the adoption candidate baseline)
#   fuel = fpp + denial_fuel,denial_crispin  (OGERPON-scoped attach discipline + Crispin)
#   chip = fpp + spread_kill,spread_reach,munki_close (counter bank/finish + Adrena closer)
#   all  = fpp + all five
# Waits for lmab2 to finish so the GPU is not shared three ways.
set -u
LOG=/root/lmab3.log
say() { echo "[lmab3 $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
grep -aq LMAB3_DONE "$LOG" 2>/dev/null && exit 0

until grep -aq LMAB2_DONE /root/lmab2.log 2>/dev/null; do sleep 120; done

cd /root/ptcg/repo
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_CHIP=1
B=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
F=front_dive,promote_dive,promote_line
OPPS=marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh
G=${G:-150}
DEADLINE=$(date -u -d 2026-08-15T16:00:00Z +%s)

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
say "champion = $CKPT, $G games/opponent, baseline fpp"

PYTHONPATH=cg-lib python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp "$OPPS" \
    --arm "fpp=planfilter:$B,$F:$CKPT@dusk" \
    --arm "fuel=planfilter:$B,$F,denial_fuel,denial_crispin:$CKPT@dusk" \
    --arm "chip=planfilter:$B,$F,spread_kill,spread_reach,munki_close:$CKPT@dusk" \
    --arm "all=planfilter:$B,$F,denial_fuel,denial_crispin,spread_kill,spread_reach,munki_close:$CKPT@dusk" \
    --games "$G" --seed 1 --baseline fpp --out /root/lmab3.json > /root/lmab3_run.log 2>&1
RC=$?

{
    echo "=============================================================="
    echo " OGERPON RACE PACKAGE ON THE SHIPPING PILOT -- $(date -u +%F\ %H:%M) UTC"
    echo " champion $CKPT, $G games x 8 opponents, fmt=dusk, paired seeds"
    echo "   fpp  = base8 + front_dive,promote_dive,promote_line"
    echo "   fuel = fpp + denial_fuel,denial_crispin (OGERPON-scoped)"
    echo "   chip = fpp + spread_kill,spread_reach,munki_close"
    echo "   all  = fpp + all five"
    echo "=============================================================="
    grep "  dragapult_dusknoir " /root/lmab3_run.log
    echo
    sed -n "/^arm /,\$p" /root/lmab3_run.log
    echo
    echo "context: LM fpp vs ogerpon was 12.0% (lmab2). engine proxy with the chip package"
    echo "read 8.0% / dives 0.35/game -- fuel, not placement, is the hypothesis under test."
} > /root/lmab3_report.txt 2>&1

say "gate exited rc=$RC; report in /root/lmab3_report.txt"
say LMAB3_DONE
