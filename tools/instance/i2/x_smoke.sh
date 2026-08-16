#!/usr/bin/env bash
# Verify cross-deck collection + branching END TO END on the machine and model that will
# actually run it: the 4B on instance2's GPU. The first attempt did this with the reranker on
# instance1 AND forced CUDA_VISIBLE_DEVICES= -- a 184M cross-encoder on a contended CPU, which
# produced nothing in six hours and left instance2 idle waiting on a check that never finished.
set -u
say() { echo "[xs $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
say "collect: dusknoir vs 2 opponents, 2 games each, seats alternating"
PYTHONPATH=cg-lib python3 tools/lm_mirror_log.py --model qwen:/root/out/dpo_r1 \
  --protagonist dragapult_dusknoir --decks marnie_grimmsnarl,alakazam_nz \
  --games 2 --seed 777 --out /root/xs.jsonl.gz --trace-out /root/xs.trace.jsonl.gz \
  --mirror-so "$SO" 2>&1 | tail -6
python3 -c "
import gzip, json
rows=[json.loads(l) for l in gzip.open('/root/xs.trace.jsonl.gz','rt')]
hdr=[r for r in rows if r.get('header')]; gs=[r for r in rows if not r.get('header')]
print('[xs] header %d | games %d' % (len(hdr), len(gs)))
for g in gs:
    print('     deck0=%-20s deck1=%-20s seed=%d decisions=%d result=%s'
          % (g.get('deck0'), g.get('deck1'), g['seed'], len(g['meta']), g['result']))
seats=set(tuple(sorted({m[0] for m in g['meta']})) for g in gs)
print('[xs] seats seen per game:', seats)
"
say "branch: replay those games and label from the ACTING seat"
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib python3 tools/dpo_branch.py \
  --traces /root/xs.trace.jsonl.gz --budget 60 --per-game 8 --margin-min 0.01 \
  --playouts 8 --workers 6 --out /root/xs_pairs.jsonl.gz --mirror-so "$SO" 2>&1 | tail -10
python3 -c "
import gzip, json, collections
rs=[json.loads(l) for l in gzip.open('/root/xs_pairs.jsonl.gz','rt')]
print('[xs] pairs %d' % len(rs))
c=collections.Counter((r['deck'], r['opp'], r['seat']) for r in rs)
for k,v in sorted(c.items()): print('     pilot=%-20s opp=%-20s seat=%d  n=%d' % (k[0],k[1],k[2],v))
print('[xs] both seats present:', len({r['seat'] for r in rs})==2)
print('[xs] pilot is always the acting deck:', all(r['deck']!=r['opp'] for r in rs))
"
say XSMOKE_DONE
