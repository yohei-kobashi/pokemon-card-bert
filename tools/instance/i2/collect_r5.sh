#!/usr/bin/env bash
# Round 4 collection: dragapult_dusknoir against the ten Stage-C opponents, cross-deck, seats
# alternating. Rounds 1-3 were same-deck mirror self-play, which produces matchups that do not
# occur on the ladder and leaves the prompt's opponent-ID segment carrying no information --
# the opponent was always our own list. This is also the first round whose labels are not
# sign-inverted for the second seat.
set -u
say() { echo "[c5 $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
# The first attempt died in all three shards on "Temporary failure in name resolution":
# transformers/unsloth phone home while loading, and a transient DNS blip therefore costs the
# whole 2.5-hour collection. Every weight this needs is already in ~/.cache/huggingface (26 GB),
# so refuse the network outright instead of depending on it.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODEL=qwen:/root/out/dpo_r4                 # adopted on the user's criterion: 52.0% on the 11
OPPS=$(python3 -c "
import sys; sys.path.insert(0,'tools'); import rl_config
print(','.join(d for d in rl_config.STAGE_C_TARGETS if d != 'dragapult_dusknoir'))")
say "protagonist dragapult_dusknoir vs $OPPS"
j=0
for SH in 0 1 2; do
  DK=$(python3 -c "
import sys
d='$OPPS'.split(',')
print(','.join(d[$SH::3]))")
  [ -n "$DK" ] || continue
  PYTHONPATH=cg-lib nohup python3 tools/lm_mirror_log.py --model "$MODEL" \
    --protagonist dragapult_dusknoir --decks "$DK" --games 150 --seed $((50000 + SH * 1000)) \
    --out /root/lmlog_r5.s$SH.jsonl.gz --trace-out /root/traces_r5.s$SH.jsonl.gz \
    --mirror-so "$SO" > /root/collect_r5.s$SH.log 2>&1 &
  j=$((j+1))
done
say "launched $j shards"
wait
grep -ahE "^  [a-z_]+ +[0-9]+ games" /root/collect_r5.s*.log | tail -12
python3 -c "
import gzip, json, glob, collections
n=0; c=collections.Counter()
for f in sorted(glob.glob('/root/traces_r5.s*.jsonl.gz')):
    for line in gzip.open(f,'rt'):
        d=json.loads(line)
        if d.get('header'): continue
        n+=1; c[(d['deck0'],d['deck1'])]+=1
print('[c5] %d games over %d matchup orderings' % (n, len(c)))
"
say COLLECT_R5_DONE
