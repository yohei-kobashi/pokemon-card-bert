#!/bin/bash
# Repeat: mirror-screen the current model -> pick the decks it loses -> collect DAgger there ->
# retrain -> screen again. Runs unattended until DEADLINE_H hours have passed.
#
# EACH ROUND TRAINS FROM THE BASE CHECKPOINT on the UNION of every DAgger file collected so far,
# never from the previous round's model. Continuing from the previous model would compound any
# round that came out worse; training from base on accumulated data is also DAgger's own
# formulation. Collection, by contrast, always uses the NEWEST model -- the point is to fix the
# errors the current policy actually makes.
#
# Env:
#   KIND            rerank | qwen
#   MODEL           the model to screen in round 1
#   DEADLINE_H      stop starting new rounds after this many hours (default 24)
#   SCREEN_GAMES    mirror games per deck (default 100)
#   COLLECT_GAMES   DAgger games per target deck (default 24)
#   MAX_TARGETS     cap on target decks (default 24)
#   SKIP_FIRST_SCREEN=1  reuse an existing $STATE/mirror_r1.json instead of re-screening
set -u
REPO=/root/ptcg/repo
KIND=${KIND:-rerank}
MODEL=${MODEL:?set MODEL}
DEADLINE_H=${DEADLINE_H:-24}
SCREEN_GAMES=${SCREEN_GAMES:-100}
COLLECT_GAMES=${COLLECT_GAMES:-24}
MAX_TARGETS=${MAX_TARGETS:-24}
SFT_LIMIT=${SFT_LIMIT:-100000}
STATE=/root/loop_$KIND
LOG=$STATE/loop.log
mkdir -p "$STATE"
cd "$REPO"
exec >> "$LOG" 2>&1
START=$(date +%s)
say() { echo "[loop $(date -u +%m-%d_%H:%M:%S) r$ROUND] $*"; }

spec() { case "$KIND" in qwen) echo "qwen:$1";; *) echo "hf:$1";; esac; }

ROUND=1
while :; do
  NOW=$(date +%s); HRS=$(( (NOW - START) / 3600 ))
  if [ "$HRS" -ge "$DEADLINE_H" ]; then
    say "deadline reached (${HRS}h) -- stopping before round $ROUND"; break
  fi
  say "=== round $ROUND | model $MODEL | ${HRS}h elapsed ==="

  MJSON=$STATE/mirror_r$ROUND.json
  if [ "$ROUND" = 1 ] && [ "${SKIP_FIRST_SCREEN:-0}" = 1 ] && [ -s "$MJSON" ]; then
    say "reusing existing screen $MJSON"
  else
    DECKS=$(PYTHONPATH=cg-lib python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'cg-lib')
import library; print(' '.join('--deck '+d for d in sorted(library.list_decks())))")
    PYTHONPATH=cg-lib python3 tools/mirror_match.py $DECKS --a engine \
        --b "$(spec "$MODEL")" --max-games "$SCREEN_GAMES" --out "$MJSON" \
        || { say "screen FAILED -- stopping"; break; }
  fi

  # trend line: the whole point of looping is that this moves
  python3 - "$MJSON" "$ROUND" "$STATE/history.tsv" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))["decks"]
p = [v["p"] for v in d.values()]
row = "%s\t%d\t%.4f\t%.4f\t%d\t%d" % (
    sys.argv[2], len(p), statistics.median(p), sum(p)/len(p),
    sum(1 for v in d.values() if v["verdict"] == "WORSE"),
    sum(1 for x in p if x < 0.5))
open(sys.argv[3], "a").write(row + "\n")
print("[trend] round\tdecks\tmedian\tmean\tWORSE\tbelow50")
print("[trend] " + open(sys.argv[3]).read().strip().replace("\n", "\n[trend] "))
PY

  TARGETS=$(python3 -c "
import json
d=json.load(open('$MJSON'))['decks']
for name, ks in [('WORSE',[k for k,v in d.items() if v['verdict']=='WORSE']),
                 ('below45',[k for k,v in d.items() if v['p']<0.45]),
                 ('below50',[k for k,v in d.items() if v['p']<0.50]),
                 ('weakest',sorted(d,key=lambda k:d[k]['p']))]:
    if ks:
        import sys; print('TIER '+name, file=sys.stderr)
        print(','.join(sorted(ks,key=lambda k:d[k]['p'])[:$MAX_TARGETS])); break
")
  say "targets: $TARGETS"
  [ -n "$TARGETS" ] || { say "no targets -- stopping"; break; }

  DAG=$STATE/dagger_r$ROUND.jsonl.gz
  PYTHONPATH=cg-lib python3 tools/collect_dagger.py --decks "$TARGETS" \
      --model "$(spec "$MODEL")" --games "$COLLECT_GAMES" --out "$DAG" \
      || { say "collect FAILED -- stopping"; break; }

  NEXT=$((ROUND + 1))
  if [ "$KIND" = qwen ]; then
    MIX=/root/ptcg/repo/data/sft/loop_r$NEXT.jsonl.gz
    python3 tools/dagger_to_sft.py --dagger "$DAG" \
        --base /root/ptcg/repo/data/sft/v39_0731.jsonl.gz --ratio 0.3 --out "$MIX" \
        || { say "convert FAILED -- stopping"; break; }
    OUT=/root/out/teacher9b_loop$NEXT
    python3 tools/instance/sft_teacher.py --domain-tokens --data "$MIX" --out "$OUT" \
        --limit "$SFT_LIMIT" --epochs 1 --bsz 8 --accum 4 --eval-n 4000 --save-steps 400 \
        || { say "train FAILED -- stopping"; break; }
    [ -f "$OUT/domain_embeddings.pt" ] || { say "no domain_embeddings.pt -- stopping"; break; }
  else
    MIX=/root/ptcg/repo/data/rerank/loop_r$NEXT.rerank.jsonl.gz
    # accumulate EVERY round's dagger, not just this one
    python3 - "$STATE" "$MIX" <<'PY'
import glob, gzip, json, random, sys
state, out = sys.argv[1], sys.argv[2]
rng = random.Random(0)
dag = []
for f in sorted(glob.glob(state + "/dagger_r*.jsonl.gz")):
    with gzip.open(f, "rt") as fh:
        dag += fh.readlines()
want = int(len(dag) * 0.7 / 0.3)
res, n = [], 0
with gzip.open("/root/ptcg/repo/data/rerank/v39_0731.rerank.jsonl.gz", "rt") as fh:
    for line in fh:
        n += 1
        if len(res) < want:
            res.append(line)
        else:
            j = rng.randrange(n)
            if j < want:
                res[j] = line
rows = dag + res
rng.shuffle(rows)
with gzip.open(out, "wt") as fh:
    fh.writelines(rows)
print("[mix] dagger %d (all rounds) + base %d = %d" % (len(dag), len(res), len(rows)))
PY
    OUT=/root/out/rerank_loop$NEXT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python3 tools/train_rerank.py --data "$MIX" --out "$OUT" --deadline-h 5 \
        --max-samples 600000 --pair-batch 32 --accum 12 --lr 2e-5 --max-len 768 \
        --eval-n 2000 --grad-ckpt || { say "train FAILED -- stopping"; break; }
    [ -f "$OUT/model.safetensors" ] || { say "no model saved -- stopping"; break; }
  fi

  say "round $ROUND done -> $OUT"
  MODEL=$OUT
  ROUND=$NEXT
  rm -f "$MIX"          # 200-400 MB each; the dagger files are what matter and they are kept
done
say "LOOP ENDED"
