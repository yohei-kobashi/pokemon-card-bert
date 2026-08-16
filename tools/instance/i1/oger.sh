set -u
cd /root/ptcg/repo
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_CRISPIN=1 DUSK_SHELTER=1
B=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
F=front_dive,promote_dive,promote_line
mkdir -p /root/oger
for S in 1 20000; do
  PYTHONPATH=cg-lib python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp ogerpon_mono \
    --arm base=planfilter:$B,$F:engine@prompt \
    --arm cr=planfilter:$B,$F,crispin_line:engine@prompt \
    --arm sh=planfilter:$B,$F,duskull_shelter:engine@prompt \
    --arm both=planfilter:$B,$F,crispin_line,duskull_shelter:engine@prompt \
    --games 400 --seed $S --baseline base > /root/oger/s$S.log 2>&1 &
done
wait
# field-wide no-regression check for the winner candidates
OPPS=ogerpon_mono,marnie_grimmsnarl,dragapult,ethan_hooh,alakazam_nz,crustle_geco,dudunsparce_box,hydrapple
PYTHONPATH=cg-lib python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp $OPPS \
  --arm base=planfilter:$B,$F:engine@prompt \
  --arm cr=planfilter:$B,$F,crispin_line:engine@prompt \
  --arm both=planfilter:$B,$F,crispin_line,duskull_shelter:engine@prompt \
  --games 300 --seed 1 --baseline base > /root/oger/field.log 2>&1
for f in /root/oger/s1.log /root/oger/s20000.log /root/oger/field.log; do
  echo "--- $f ---"; grep -E "vs |^arm |^base |^cr |^sh |^both " $f | grep -v wrote
done
echo OGER_DONE
