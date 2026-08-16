#!/usr/bin/env bash
# Round 6 collection. NEW: slowking joins the opponent set.
#
# Why it has to. slowking holds #1 and #2 on the ladder and 8.0% of the top-50, and our own
# engine_v2 measurements put it at 68% against dragapult across three independent runs
# (68.1 / 67.8 / 67.5). It is in neither STAGE_C_TARGETS nor the opponent list, so every round
# so far has trained and gated against a field that excludes the deck at the top of it.
#
# Human meta says the opposite -- Dragapult is FAVOURED into Slowking, conditional on how well
# Slowking plays Smoochum -- and the reference agents run Delightful Kiss as 30% of their
# attacks. So our 68% is measuring OUR dragapult being weak, not the matchup being lost. Either
# way the fix is the same: play the games.
set -u
say() { echo "[c6 $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODEL=${MODEL:-qwen:/root/out/dpo_r5}
OPPS=$(python3 -c "
import sys; sys.path.insert(0,'tools'); import rl_config
d=[x for x in rl_config.STAGE_C_TARGETS if x!='dragapult_dusknoir']
d.append('slowking')          # the ladder's #1 and #2, absent from every round so far
print(','.join(d))")
say "protagonist dragapult_dusknoir vs $OPPS"
j=0
for SH in 0 1 2; do
  DK=$(python3 -c "
import sys
d='$OPPS'.split(',')
print(','.join(d[$SH::3]))")
  [ -n "$DK" ] || continue
  PYTHONPATH=cg-lib nohup python3 tools/lm_mirror_log.py --model "$MODEL" \
    --protagonist dragapult_dusknoir --decks "$DK" --games 150 --seed $((60000 + SH * 1000)) \
    --out /root/lmlog_r6.s$SH.jsonl.gz --trace-out /root/traces_r6.s$SH.jsonl.gz \
    --mirror-so "$SO" > /root/collect_r6.s$SH.log 2>&1 &
  j=$((j+1))
done
say "launched $j shards"
wait
python3 -c "
import gzip, json, glob, collections
n=0; c=collections.Counter()
for f in sorted(glob.glob('/root/traces_r6.s*.jsonl.gz')):
    for line in gzip.open(f,'rt'):
        d=json.loads(line)
        if d.get('header'): continue
        n+=1; c[(d['deck0'],d['deck1'])]+=1
print('[c6] %d games over %d matchup orderings' % (n, len(c)))
sk=sum(v for k,v in c.items() if 'slowking' in k)
print('[c6] dusknoir vs slowking games: %d' % sk)
"
say COLLECT_R6_DONE
