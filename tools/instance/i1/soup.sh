#!/usr/bin/env bash
# Weight interpolation between the pre-finetune checkpoint and the dusknoir-only round.
# theta(a) = a*dusk_r1 + (1-a)*d41_r8. Costs no training: if round 1 learned something about
# this deck but paid for it by drifting off the eleven-deck representation, the interpolation
# keeps the first and undoes the second, and the curve says which. If every alpha sits between
# the endpoints there was nothing to recover and the round simply did not learn.
set -u
say() { echo "[soup $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
mkdir -p /root/soup
for A in 0.25 0.50 0.75; do
  OUT=/root/out/soup_$A
  rm -rf "$OUT"; mkdir -p "$OUT"
  cp /root/out/d41_r8/config.json /root/out/d41_r8/tokenizer.json \
     /root/out/d41_r8/tokenizer_config.json "$OUT"/ 2>/dev/null
  python3 - "$A" "$OUT" <<'PY'
import sys, torch
from safetensors.torch import load_file, save_file
a, out = float(sys.argv[1]), sys.argv[2]
base = load_file("/root/out/d41_r8/model.safetensors")
ft = load_file("/root/out/dusk_r1/model.safetensors")
assert set(base) == set(ft), "checkpoints are not the same architecture"
merged = {}
for k in base:
    b, f = base[k], ft[k]
    if b.dtype.is_floating_point:
        merged[k] = ((1 - a) * b.float() + a * f.float()).to(b.dtype)
    else:
        merged[k] = b.clone()
save_file(merged, out + "/model.safetensors")
print("[soup] alpha %.2f -> %s (%d tensors)" % (a, out, len(merged)))
PY
done
say "gating three alphas at 400 games each"
for A in 0.25 0.50 0.75; do
  PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py --deck dragapult_dusknoir \
    --a engine --b "hf:/root/out/soup_$A" --max-games 400 --mirror --seed 1 --mirror-so "$SO" \
    --out /root/soup/a_$A.json > /root/soup/a_$A.log 2>&1 &
done
wait
python3 -c "
import json, os
print('%-14s %8s %10s' % ('model','wr','record'))
print('%-14s %7.1f%%  %s' % ('d41_r8 (a=0)', 45.2, '181-219'))
for a in ('0.25','0.50','0.75'):
    p='/root/soup/a_%s.json'%a
    if not os.path.exists(p): continue
    d=json.load(open(p))['decks']['dragapult_dusknoir']
    print('%-14s %7.1f%%  %d-%d' % ('soup a=%s'%a, 100*d['p'], d['w'], d['l']))
print('%-14s %7.1f%%  %s' % ('dusk_r1 (a=1)', 41.5, '166-234'))
"
say SOUP_DONE
