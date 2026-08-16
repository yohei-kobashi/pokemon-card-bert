#!/usr/bin/env bash
# d41_r8 was CHOSEN because it scored highest of six checkpoints on dusknoir, so its 42.7%
# carries max-of-6 selection bias -- the same winner's curse that inflated the 16-playout dQ
# gaps. Re-measure both at 400 games on the same seeds before calling the SFT round a
# regression. The other five checkpoints scored 33.3-36.0% on this deck.
set -u
say() { echo "[dab $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
mkdir -p /root/dusk
for M in d41_r8 dusk_r1 d41_r6; do
  [ -d /root/out/$M ] || continue
  PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py --deck dragapult_dusknoir \
    --a engine --b "hf:/root/out/$M" --max-games 400 --mirror --seed 1 --mirror-so "$SO" \
    --out /root/dusk/ab_$M.json > /root/dusk/ab_$M.log 2>&1 &
done
say "launched 3 arms x 400 games on dragapult_dusknoir"
wait
python3 -c "
import json, os
for m in ('d41_r8','dusk_r1','d41_r6'):
    p='/root/dusk/ab_%s.json'%m
    if not os.path.exists(p): continue
    d=json.load(open(p))['decks']['dragapult_dusknoir']
    s0,s1=d['seat0'],d['seat1']
    print('%-9s %5.1f%%  (%d-%d)  seat0 %.1f%%  seat1 %.1f%%  verdict %s'
          % (m, 100*d['p'], d['w'], d['l'],
             100*s0[0]/max(1,sum(s0)), 100*s1[0]/max(1,sum(s1)), d['verdict']))
"
say DUSK_AB_DONE
