#!/usr/bin/env bash
# Efficiency audit of the SHIPPING pilot (full wrap + champion) across the field.
set -u
cd /root/ptcg/repo
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1
W=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search,front_dive,promote_dive,promote_line,lethal_boss
CKPT=$(PYTHONPATH=cg-lib:tools python3 -c "from lm import registry as r; print(r.resolve(\"dragapult_dusknoir\")[\"target\"])")
SPEC="planfilter:$W:$CKPT"
mkdir -p /root/eff
for OPP in alakazam_nz mega_abomasnow_sample archaludon ogerpon_mono ethan_hooh dragapult; do
  PYTHONPATH=.:cg-lib:tools python3 tools/eff_audit.py --spec "$SPEC" --opp $OPP \
      --games 100 --seed 61000 --json /root/eff/$OPP.json > /dev/null 2>/root/eff/$OPP.err
done
echo EFF_DONE
