#!/usr/bin/env bash
# The slowking cell: LM vs engine_v2, mirror, on the live decklist. Reported SEPARATELY from
# the 11-deck paired gate so the adoption criterion stays comparable across rounds -- but
# measured every round from now on, because slowking is the ladder's #1 and #2 and round 6 is
# the first round that trains against it.
set -u
say() { echo "[skcell $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
for M in "$@"; do
  TAG=$(basename "$M")
  PYTHONPATH=cg-lib python3 tools/mirror_match.py --deck slowking --a engine --b "qwen:$M" \
    --max-games 229 --mirror --seed 1 --mirror-so "$SO" \
    --out /root/loop_dpo/skcell_$TAG.json > /root/loop_dpo/skcell_$TAG.log 2>&1
  python3 -c "
import json
d=json.load(open('/root/loop_dpo/skcell_$TAG.json'))['decks']['slowking']
s0,s1=d['seat0'],d['seat1']
print('  %-10s slowking %5.1f%% (%d-%d)  seat0 %.1f%%  seat1 %.1f%%  verdict %s'
      % ('$TAG', 100*d['p'], d['w'], d['l'],
         100*s0[0]/max(1,sum(s0)), 100*s1[0]/max(1,sum(s1)), d['verdict']))"
done
say SK_CELL_DONE
