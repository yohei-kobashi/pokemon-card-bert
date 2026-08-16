cd /root/ptcg/repo
export PYTHONPATH=cg-lib:tools HF_HUB_OFFLINE=1 DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1
SPEC="planfilter:lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace:hf:/root/out/fld_r10a"
for FLAG in 0 1; do
  export DUSK_BLAST_REACH=$FLAG
  for O in ogerpon_mono marnie_grimmsnarl; do
    echo "######## BLAST_REACH=$FLAG  $O ########"
    python3 -u tools/dusk_ogerpon_audit.py --games 30 --opp $O --spec "$SPEC"         --mirror-so /root/ptcg/repo/data/kaggle_engine_ext/libcg_mirror.so 2>&1       | grep -aE "wins|dusclops|dusknoir|attacks used" 
  done
done
echo BLAST_AB_DONE
