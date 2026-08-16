set -u
cd /root/ptcg/repo
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_LOCK_BUDEW=1
B=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
OPPS=ogerpon_mono,marnie_grimmsnarl,dragapult,ethan_hooh,alakazam_nz,crustle_geco,dudunsparce_box,hydrapple
mkdir -p /root/lockab
for S in 1 5000 20000; do
  PYTHONPATH=cg-lib python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp $OPPS \
    --arm base=planfilter:$B:engine@prompt \
    --arm lb=planfilter:$B,lock_budew:engine@prompt \
    --arm lble=planfilter:$B,lock_budew,lock_early:engine@prompt \
    --games 300 --seed $S --baseline base --out /root/lockab/s$S.json > /root/lockab/s$S.log 2>&1 &
done
wait
for S in 1 5000 20000; do echo "--- seed $S ---"; sed -n "/^arm /,\$p" /root/lockab/s$S.log | grep -v wrote; done
echo LOCKAB_DONE
