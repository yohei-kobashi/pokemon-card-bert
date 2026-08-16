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
LOG=/root/lmab5.log
say() { echo "[lmab5 $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
grep -aq LMAB5_DONE "$LOG" 2>/dev/null && exit 0


cd /root/ptcg/repo
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1 DUSK_SPIKE=1 DUSK_WIDE=1
B=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
F=front_dive,promote_dive,promote_line,lethal_boss
OPPS=ogerpon_mono,dragapult,marnie_grimmsnarl
G=${G:-200}
DEADLINE=$(date -u -d 2026-08-16T06:00:00Z +%s)

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
    --arm "base=planfilter:$B,$F:$CKPT@dusk" \
    --arm "spike=planfilter:$B,$F,spike_candy,spike_race:$CKPT@dusk" \
    --arm "wide=planfilter:$B,$F,third_loak:$CKPT@dusk" \
    --arm "all=planfilter:$B,$F,spike_candy,spike_race,third_loak:$CKPT@dusk" \
    --games "$G" --seed 1 --baseline base --out /root/lmab5.json > /root/lmab5_run.log 2>&1
RC=$?

{
    echo "=============================================================="
    echo " SPIKE-RACE / DRAW-WIDTH RULES (pokehubguide) ON THE SHIPPING PILOT -- $(date -u +%F\ %H:%M) UTC"
    echo " champion $CKPT, $G games x 8 opponents, fmt=dusk, paired seeds"
    echo "   fpp  = base8 + front_dive,promote_dive,promote_line"
    echo "   base  = adopted set (incl lethal_boss)"
    echo "   spike = base + spike_candy,spike_race (race them with spikes, OGERPON-scoped)"
    echo "   wide  = base + third_loak (three Drakloak = maximum draw)"
    echo "   all   = base + all three"
    echo "=============================================================="
    grep "  dragapult_dusknoir " /root/lmab5_run.log
    echo
    sed -n "/^arm /,\$p" /root/lmab5_run.log
    echo
    echo "context: pokehubguide says Ogerpon is THE unfavorable matchup: race with spikes."
} > /root/lmab5_report.txt 2>&1

say "gate exited rc=$RC; report in /root/lmab5_report.txt"
say LMAB5_DONE
