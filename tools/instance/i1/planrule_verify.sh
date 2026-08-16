#!/usr/bin/env bash
# Confirm the promotion + energy plan rules on instance1, over INDEPENDENT seed blocks.
#
# Why blocks and not one long run: gate_protagonist uses seed = --seed + g/2, so two runs whose
# seed ranges overlap are re-measuring the same shuffles. The ogerpon ablation earlier today had
# one block (7000) that showed nothing while eleven others nearly doubled -- a single block is
# not evidence either way.
#
# CPU only: `planfilter:<rules>:engine` needs no GPU, so this does not contend with the round
# gate. One process per seed block, each single-threaded.
set -u
cd /root/ptcg/repo
unset DUSK_RULES
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1
B=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
F=front_dive,promote_dive,promote_line
E=energy_line,energy_focus
OPPS=ogerpon_mono,marnie_grimmsnarl,dragapult,ethan_hooh,alakazam_nz,crustle_geco,dudunsparce_box,hydrapple
G=${G:-300}
OUT=/root/planrule
mkdir -p $OUT
for S in 1 5000 20000 30000 40000 50000; do
  PYTHONPATH=cg-lib python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp $OPPS \
    --arm base=planfilter:$B:engine@prompt \
    --arm fpp=planfilter:$B,$F:engine@prompt \
    --arm chg=planfilter:$B,$E:engine@prompt \
    --arm fppchg=planfilter:$B,$F,$E:engine@prompt \
    --games $G --seed $S --baseline base --out $OUT/s$S.json > $OUT/s$S.log 2>&1 &
done
wait
echo "==== per-block deltas vs base ===="
for S in 1 5000 20000 30000 40000 50000; do
  echo "--- seed $S ---"
  sed -n '/^arm /,$p' $OUT/s$S.log
done
echo PLANRULE_DONE
