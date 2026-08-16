#!/bin/bash
# instance2 all-in: scheme B + playout-valued attach, warm-started from the scheme-A checkpoint.
#
# WHY THE VALUED DATA BELONGS HERE MORE THAN ANYWHERE. Profiled on this model's own DAgger,
# `attach` is 26.0% of everything it gets wrong at a lift of 4.38x -- the largest error category
# it has, and worse than the reranker's 18.0%/2.82x. And the teacher cannot fix it: on decisions
# where the playouts find a decisive answer, ATTACH is right 60% of the time while engine_v2
# attaches on only 34.8%, and its pick is the measured best on 18.9%. The model inherited an
# under-attaching teacher, so more imitation cannot help.
#
# The menu-dedup prompt change is NOT applied here: `to_scheme_b` rewrites the menu through
# `groups()`, which already collapses the same duplicates. Verified identical on 4,000/4,000
# prompts, so for scheme B it is a no-op.
#
# NO MARGIN TERM. The decoder's loss is cross-entropy over one or two answer tokens; a pairwise
# value margin needs logits for competing candidates, which is a different objective and is not
# implemented. The valued records enter as ordinary labels -- their value is that the LABEL is
# right, not that the loss reads Q.
set -u
REPO=/root/ptcg/repo
LOG=/root/chain_v40_i2.log
SRC=/root/out/qwen3_4b_cf1
VOCAB=$REPO/data/cardfirst_b_v39.json
BASEMIX=$REPO/data/sft/cf_b_r2.jsonl.gz
MIX=$REPO/data/sft/cf_b_v40.jsonl.gz
OUT=/root/out/qwen3_4b_cfb_v40
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[i2 $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "############ waiting for attach_q4 ############"
while pgrep -f "[a]ttach_label.py --games" > /dev/null; do sleep 60; done
sleep 15

say "=== 1/4 convert every valued file to the decoder's SFT schema ==="
V=""
for f in attach_q2 attach_q3 attach_q4; do
  [ -s "data/rerank/$f.jsonl.gz" ] && V="$V,data/rerank/$f.jsonl.gz"
done
V="${V#,}"
[ -n "$V" ] || { say "STOP: no valued attach files found"; exit 1; }
python3 tools/valued_to_sft.py --inp "$V" --out data/sft/valued_q234.jsonl.gz \
  || { say "STOP: conversion failed"; exit 1; }

say "=== 2/4 mix ==="
python3 - "$BASEMIX" "$MIX" data/sft/valued_q1.jsonl.gz data/sft/valued_q234.jsonl.gz <<'PY'
import gzip, json, random, sys
base, out = sys.argv[1], sys.argv[2]
rng = random.Random(0)
rows = []
with gzip.open(base, "rt") as f:
    rows += f.readlines()
nb = len(rows)
val = []
for p in sys.argv[3:]:
    with gzip.open(p, "rt") as f:
        v = f.readlines()
    print("  valued %-34s %d" % (p.rsplit("/", 1)[-1], len(v)))
    val += v
# Every valued record is used EXACTLY ONCE. Repeating them to hit a round number would upweight
# a file that covers one decision kind, and subsampling would throw away playouts that cost 16
# rollouts each; using them as they are makes the share a fact rather than a knob.
rows += val
rng.shuffle(rows)
with gzip.open(out, "wt") as f:
    f.writelines(rows)
print("  %s: %d rows | base %d (%.1f%%) | valued %d (%.1f%%)"
      % (out, len(rows), nb, 100.0 * nb / len(rows), len(val), 100.0 * len(val) / len(rows)))
PY
[ -s "$MIX" ] || { say "STOP: mix missing"; exit 1; }

# A mix that silently lost the valued rows trains as a plain imitation run and completes looking
# like a fair test of the package.
python3 - "$MIX" <<'PY' || { say "STOP: the mix carries no valued rows"; exit 1; }
import gzip, json, sys
n = v = 0
for line in gzip.open(sys.argv[1], "rt"):
    n += 1
    if n % 5 == 0 and json.loads(line).get("valued"):
        v += 1
print("[check] %d rows; %d of every 5th are valued (%.2f%%)" % (n, v, 100.0 * v / max(1, n // 5)))
raise SystemExit(0 if v else 1)
PY

say "=== 3/4 warm-start preflight (5 steps) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf /root/out/i2_preflight
python3 tools/instance/sft_teacher.py --model unsloth/Qwen3-4B-Base --data "$MIX" \
  --domain-tokens --card-first "$VOCAB" --init-from "$SRC" \
  --out /root/out/i2_preflight --limit 400 --eval-n 0 --steps 5 \
  --bsz 8 --accum 4 --maxlen 896 --group-by-length --save-steps 100000 2>&1 \
  | grep -E "^\[warm\]|^\[cardfirst\]|^\[data\]|REFUSING|Error" | tee /root/i2_preflight.txt
grep -qE "^\[warm\] embedding rows restored by name: [0-9]{4,}" /root/i2_preflight.txt \
  || { say "STOP: too few embedding rows restored"; exit 1; }
grep -q "^\[warm\] LoRA tensors restored: 0 " /root/i2_preflight.txt \
  && { say "STOP: the LoRA did not load"; exit 1; }
rm -rf /root/out/i2_preflight
say "warm-start preflight OK"

say "=== 4/4 train scheme B from $SRC ==="
python3 tools/instance/sft_teacher.py --model unsloth/Qwen3-4B-Base --data "$MIX" \
  --domain-tokens --card-first "$VOCAB" --init-from "$SRC" \
  --out "$OUT" --limit 400000 --eval-n 4000 --epochs 1 \
  --bsz 8 --accum 4 --maxlen 896 --group-by-length --save-steps 1000 \
  || { say "STOP: training failed"; exit 1; }
[ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no domain_embeddings.pt -- the added rows are lost"; exit 1; }
say "CHAIN DONE -> $OUT"
