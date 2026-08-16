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
LOG=/root/lmab4.log
say() { echo "[lmab4 $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
grep -aq LMAB4_DONE "$LOG" 2>/dev/null && exit 0


cd /root/ptcg/repo
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1 DUSK_CSPLIT=1
B=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
F=front_dive,promote_dive,promote_line
OPPS=marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh
G=${G:-150}
DEADLINE=$(date -u -d 2026-08-15T20:00:00Z +%s)

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
    --arm "boss=planfilter:$B,$F,lethal_boss:$CKPT@dusk" \
    --arm "split=planfilter:$B,$F,crispin_split:$CKPT@dusk" \
    --arm "both=planfilter:$B,$F,lethal_boss,crispin_split:$CKPT@dusk" \
    --games "$G" --seed 1 --baseline fpp --out /root/lmab4.json > /root/lmab4_run.log 2>&1
RC=$?

{
    echo "=============================================================="
    echo " GUIDE RULES (lethal_boss / crispin_split) ON THE SHIPPING PILOT -- $(date -u +%F\ %H:%M) UTC"
    echo " champion $CKPT, $G games x 8 opponents, fmt=dusk, paired seeds"
    echo "   fpp  = base8 + front_dive,promote_dive,promote_line"
    echo "   boss  = fpp + lethal_boss (gust-completed lethal)"
    echo "   split = fpp + crispin_split (early Crispin, split attach)"
    echo "   both  = fpp + both"
    echo "   NOTE: deck now carries Risky Ruins x1 (A-swap, -1 Watchtower) -- all arms alike"
    echo "=============================================================="
    grep "  dragapult_dusknoir " /root/lmab4_run.log
    echo
    sed -n "/^arm /,\$p" /root/lmab4_run.log
    echo
    echo "context: rules distilled from the published deck guide; fire rates 0.02-0.05/game"
    echo "(lethal_boss, rare-but-decisive) and 0.8/game (crispin_split)."
} > /root/lmab4_report.txt 2>&1

say "gate exited rc=$RC; report in /root/lmab4_report.txt"
say LMAB4_DONE
