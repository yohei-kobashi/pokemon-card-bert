#!/usr/bin/env bash
# Stage 1-3 of the single-deck overhaul: re-render the pool without DECK[], re-SFT from
# d41_r8, and check the strength came back.
#
# Warm start, not from scratch: the card knowledge lives in the domain-token embeddings, and
# the last format change measured base top1 at 50.6% on the NEW format from a warm start
# against ~20% from the stock base. The segment removal is smaller than that change.
#
# The retreat is explicit. If stage 3 does not return to d41_r8's 45.2%, the segment was a
# scaffold the model leans on even when it carries no information (`deck-segment-reliance`
# measured exactly that), and this whole branch is reverted rather than patched.
set -u
say() { echo "[fmt $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so

say "stage 1: re-render the pool without DECK[] (3M rows is plenty for one round)"
python3 /root/mk_dusk_sft.py "$REPO/data/rerank/v41_dusk.jsonl.gz" \
        "$REPO/data/rerank/v41_dusk_nodeck.jsonl.gz" 3000000
ls -la "$REPO/data/rerank/v41_dusk_nodeck.jsonl.gz"

say "stage 2: re-SFT from d41_r8, 400k rows, lr 1e-5, L2-SP anchored"
OUT=/root/out/dusk_nodeck
rm -rf "$OUT"; mkdir -p "$OUT"
cp -r /root/out/d41_r8/. "$OUT"/ && rm -f "$OUT/rr_progress.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 tools/train_rerank.py --data "$REPO/data/rerank/v41_dusk_nodeck.jsonl.gz" --out "$OUT" \
  --resume --deadline-h 5 --max-samples 400000 --lr 1e-5 --pair-batch 32 --accum 2 \
  --max-len 512 --eval-n 2000 --grad-ckpt --margin-weight 0.5 --l2sp 1e-3 \
  > /root/rl/fmt_train.log 2>&1 || { say "TRAIN FAILED"; tail -6 /root/rl/fmt_train.log; exit 1; }
grep -aE "\[l2sp\]|eval|FINAL" /root/rl/fmt_train.log | tail -5

say "stage 3: did the strength come back? 400 games, dusknoir mirror, vs engine_v2"
python3 tools/mirror_match.py --deck dragapult_dusknoir --a engine --b "hf:$OUT" \
  --max-games 400 --mirror --seed 1 --mirror-so "$SO" \
  --out /root/rl/fmt_gate.json > /root/rl/fmt_gate.log 2>&1
python3 -c "
import json
d=json.load(open('/root/rl/fmt_gate.json'))['decks']['dragapult_dusknoir']
p=100*d['p']
print('  no-DECK model: %.1f%% (%d-%d)   baseline d41_r8: 45.2%%   delta %+.1fpt' % (p,d['w'],d['l'],p-45.2))
print('  VERDICT:', 'format change is safe -- proceed' if p >= 42.0 else
      'strength did NOT come back -- revert the format change')"
say FMT_DONE
