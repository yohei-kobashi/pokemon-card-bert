#!/bin/bash
# Unattended: the scheme-A run finishes -> mirror screen against engine_v2 -> collect DAgger on
# whatever it loses -> continue training in SCHEME B on a 5% mix.
#
# WHY A MIRROR AND NOT AN ACCURACY NUMBER. `[eval] FIRST` is agreement with engine_v2, and this
# project has three demonstrations that agreement does not predict play: the 9B reached 91% top1
# and still lost to a 149M reranker that agrees far less. The screen is same-deck, so the null is
# exactly 0.500 by symmetry -- no baseline to measure, and none of the 2.6pt re-scoring noise
# that made earlier gates unreadable.
#
# TARGETS CASCADE, so a screen that proves nothing WORSE still produces work:
#   1. verdict WORSE   2. below 45%   3. below 50%   4. the weakest N
# Which tier fired is printed, because "we trained on tier 1" and "we trained on tier 3" mean
# very different things about the model.
#
# COLLECTION RUNS IN SCHEME A. The pilot doing the collecting is the scheme-A model and only
# understands scheme-A prompts; sft_teacher rewrites the menu to scheme B when it loads the data,
# so the conversion happens once, in the place that trains.
set -u
REPO=/root/ptcg/repo
SRC=${SRC:-/root/out/qwen3_4b_cf1}
VOCAB_B=${VOCAB_B:-/root/ptcg/repo/data/cardfirst_b_v39.json}
BASE=${BASE:-/root/ptcg/repo/data/sft/v39_dag005.jsonl.gz}
SCREEN_GAMES=${SCREEN_GAMES:-40}
SHARDS=${SHARDS:-4}
COLLECT_GAMES=${COLLECT_GAMES:-24}
MAX_TARGETS=${MAX_TARGETS:-20}
RATIO=${RATIO:-0.05}
LIMIT=${LIMIT:-400000}
LOG=/root/chain_cf.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[chain $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "############ waiting for the scheme-A run ############"
while ps -eo args | grep -q "[s]ft_teacher.py --model unsloth/Qwen3-4B"; do sleep 120; done
sleep 30

# The added embedding rows are written only at the END of training and live nowhere else. Without
# them the checkpoint cannot be scored at all, so stop rather than screen a half-built model.
if [ ! -f "$SRC/domain_embeddings.pt" ] || [ ! -f "$SRC/cardfirst_vocab.json" ]; then
  say "STOP: $SRC is missing domain_embeddings.pt or cardfirst_vocab.json -- the run did not"
  say "      reach its final save, so the added rows are lost and nothing can be resumed."
  exit 1
fi
say "scheme-A checkpoint ready: $SRC"

say "=== 1/4 preflight: two games on one deck ==="
PYTHONPATH=cg-lib timeout 1800 python3 tools/mirror_match.py --deck crustle_stall \
    --a engine --b "qwen:$SRC" --max-games 2 --out /root/preflight_cf.json 2>&1 | tail -12
[ -s /root/preflight_cf.json ] || { say "STOP: preflight produced nothing -- the model could not be loaded or scored."; exit 1; }
say "preflight OK"

# SHARDED, because one process would take all night. Measured on this model: 0.135 s per
# decision and ~81 decisions per game, so 63 decks x 40 games is ~7.7 hours serially -- the
# follow-up training would not start until the next afternoon. Training has finished by the
# time this runs, so the card is free: each shard holds its own 4B (~8 GB of 48).
say "=== 2/4 mirror screen vs engine_v2 (${SCREEN_GAMES} games/deck, ${SHARDS} shards) ==="
python3 - "$SHARDS" <<'PY' > /root/cf_shards.txt
import sys
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
import library
d = sorted(library.list_decks())
n = int(sys.argv[1])
for i in range(n):
    print(" ".join("--deck " + x for x in d[i::n]))
PY
i=0
while read -r DECKS; do
  [ -n "$DECKS" ] || continue
  PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "qwen:$SRC" \
      --max-games "$SCREEN_GAMES" --out /root/mirror_cf.$i.json > /root/screen_$i.log 2>&1 &
  i=$((i+1))
done < /root/cf_shards.txt
say "launched $i screen shards"
wait
say "shards finished"
python3 - $i <<'PY' || { say "screen FAILED -- no shard produced a result"; exit 1; }
import json, sys
out = {"decks": {}}
got = 0
for i in range(int(sys.argv[1])):
    try:
        d = json.load(open("/root/mirror_cf.%d.json" % i))
    except Exception as e:
        print("shard %d unreadable: %s" % (i, e))
        continue
    out["decks"].update(d.get("decks") or {})
    got += 1
print("merged %d shards -> %d decks" % (got, len(out["decks"])))
if not out["decks"]:
    raise SystemExit(1)
json.dump(out, open("/root/mirror_cf.json", "w"))
PY

python3 - /root/mirror_cf.json <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))["decks"]
p = [v["p"] for v in d.values()]
print("[screen] decks %d | median %.1f%% | mean %.1f%% | WORSE %d | below 50%% %d"
      % (len(p), 100*statistics.median(p), 100*sum(p)/len(p),
         sum(1 for v in d.values() if v["verdict"] == "WORSE"), sum(1 for x in p if x < .5)))
for k in sorted(d, key=lambda k: d[k]["p"])[:10]:
    print("   %-24s %.1f%%  %s" % (k, 100*d[k]["p"], d[k]["verdict"]))
PY

say "=== 3/4 target selection ==="
TARGETS=$(python3 -c "
import json, sys
d = json.load(open('/root/mirror_cf.json'))['decks']
for name, ks in [('WORSE',   [k for k,v in d.items() if v['verdict']=='WORSE']),
                 ('below45', [k for k,v in d.items() if v['p']<0.45]),
                 ('below50', [k for k,v in d.items() if v['p']<0.50]),
                 ('weakest', sorted(d, key=lambda k: d[k]['p']))]:
    if ks:
        print('TIER=%s (%d decks)' % (name, len(ks)), file=sys.stderr)
        print(','.join(sorted(ks, key=lambda k: d[k]['p'])[:$MAX_TARGETS])); break
")
say "targets: $TARGETS"
[ -n "$TARGETS" ] || { say "no targets -- stopping"; exit 1; }

say "=== 4/4 DAgger collection (${COLLECT_GAMES} games/deck), scheme-A prompts ==="
DAG=/root/ptcg/repo/data/rerank/dagger_cf1.jsonl.gz
PYTHONPATH=cg-lib python3 tools/collect_dagger.py --decks "$TARGETS" \
    --model "qwen:$SRC" --games "$COLLECT_GAMES" --out "$DAG" || { say "collect FAILED"; exit 1; }

MIX=/root/ptcg/repo/data/sft/cf_b_r2.jsonl.gz
python3 tools/dagger_to_sft.py --dagger "$DAG" --base "$BASE" --ratio "$RATIO" --out "$MIX" \
    || { say "convert FAILED"; exit 1; }

# The warm-start path has never run: --init-from was written for this chain and the file it
# reads (domain_embeddings.pt) does not exist until the scheme-A run finishes. Five steps prove
# the rows and the LoRA actually load before a 15-hour run is committed to them. Failing here
# costs two minutes; failing silently costs a night and produces a model that looks trained.
say "=== warm-start preflight (5 steps) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf /root/out/cfb_preflight
python3 tools/instance/sft_teacher.py \
  --model unsloth/Qwen3-4B-Base --data "$MIX" \
  --domain-tokens --card-first "$VOCAB_B" --init-from "$SRC" \
  --out /root/out/cfb_preflight --limit 400 --eval-n 0 --steps 5 \
  --bsz 8 --accum 4 --maxlen 896 --group-by-length --save-steps 100000 2>&1 \
  | grep -E "^\[warm\]|^\[cardfirst\]|^\[data\]|^\[done\]|REFUSING|Error" | tee /root/cfb_preflight.txt
if ! grep -qE "^\[warm\] embedding rows restored by name: [0-9]{4,}" /root/cfb_preflight.txt; then
  say "STOP: the warm start restored too few embedding rows. Resuming would train on card"
  say "      vectors that mean nothing; a fresh run would be better than this one."
  exit 1
fi
grep -q "^\[warm\] LoRA tensors restored: 0 " /root/cfb_preflight.txt && {
  say "STOP: the LoRA did not load -- this would be a fresh run wearing a resume's name."; exit 1; }
rm -rf /root/out/cfb_preflight
say "warm-start preflight OK"

say "=== continue training in SCHEME B from $SRC ==="
python3 tools/instance/sft_teacher.py \
  --model unsloth/Qwen3-4B-Base \
  --data "$MIX" \
  --domain-tokens --card-first "$VOCAB_B" \
  --init-from "$SRC" \
  --out /root/out/qwen3_4b_cfb \
  --limit "$LIMIT" --eval-n 4000 --epochs 1 \
  --bsz 8 --accum 4 --maxlen 896 --group-by-length \
  --save-steps 1000
say "CHAIN DONE rc=$?"
