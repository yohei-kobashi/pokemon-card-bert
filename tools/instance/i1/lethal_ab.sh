#!/usr/bin/env bash
# Does putting lethal_now back in the wrapper actually stop us missing wins?
# Direct count, not a win rate: the rule fires ~0.26 times per game, which no win-rate gate can
# resolve, but "offered vs taken" resolves at n=1.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1
PROH=clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace
M=/root/out/mrl2_r5b
for arm in "proh:planfilter:$PROH:hf:$M" "prohlethal:planfilter:lethal_now,$PROH:hf:$M"; do
    name=${arm%%:*}; spec=${arm#*:}
    echo "===== $name ====="
    python3 -u tools/lethal_check.py --spec "$spec" --games 40 --fmt dusk \
        --out /root/lethal_$name.json 2>&1 | grep -avE "Loading weights|^\[qwen\]" | tail -8
done
echo LETHAL_AB_DONE
