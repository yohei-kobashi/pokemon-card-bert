#!/bin/bash
# screen -> collect DAgger on what it loses -> retrain from base -> screen again, repeating until
# a deadline. Successor to dagger_loop2.sh, with one deliberate change.
#
# EACH ROUND USES ONLY ITS OWN DAgger. dagger_loop2 accumulated every round's collection, and
# that is where the run went backwards -- paired on the same 63 decks:
#
#     base   -> round 1 out   +3.06pt +- 1.15   0 decks became WORSE, 16 left   exact p 0.000
#     round1 -> round 2 out   -4.25pt +- 1.62  16 decks became WORSE,  3 left   exact p 0.004
#     base   -> round 2 out   -1.19pt +- 1.21   indistinguishable from the base  p 0.581
#
# Two rounds netted nothing. Round 1 trained on one collection; round 2 added a second, gathered
# by a pilot that was itself weak. The accumulation is not PROVEN to be the cause -- that round
# also re-deduped its first collection and drew a different base sample -- but it is the one
# thing the loop does that gets monotonically worse, and dropping it costs nothing to test.
#
# The valued-attach files are NOT DAgger and are NOT dropped: they are playout-measured labels
# that do not depend on which pilot collected them, so they stay in every round.
#
# RATIO is 0.05, down from 0.10, per instruction.
set -u
REPO=/root/ptcg/repo
KIND=${KIND:-rerank3}
MODEL=${MODEL:?set MODEL to the checkpoint this loop starts from}
BASE=${BASE:-$REPO/data/rerank/v40_base.jsonl.gz}
VALUED=${VALUED:-$REPO/data/rerank/v40_attach_q1.jsonl.gz,$REPO/data/rerank/v40_attach_q2.jsonl.gz}
RATIO=${RATIO:-0.05}
VALUED_FRAC=${VALUED_FRAC:-0.05}
SCREEN_GAMES=${SCREEN_GAMES:-40}
SHARDS=${SHARDS:-4}
# 72, not 24. At 24 games a round collects ~19,000 records, and holding DAgger at 5% then fixes
# the whole mix at ~380,000 rows -- the base pool would contribute 342,000 of its 2.87M, against
# the 1.26M rerank_loop2 trained on. Sizing the round by the collection was an accident of the
# ratio; 72 games yields ~57,000 records, so 5% still buys a ~1.1M mix.
COLLECT_GAMES=${COLLECT_GAMES:-72}
MAX_TARGETS=${MAX_TARGETS:-20}
DEADLINE_H=${DEADLINE_H:-24}
MARGIN=${MARGIN:-0.5}
STATE=/root/loop_$KIND
LOG=$STATE/loop.log
mkdir -p "$STATE"
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[l3 $(date -u +%m-%d_%H:%M:%S) r$ROUND] $*"; }

T0=$(date +%s)
ROUND=1
while :; do
  HRS=$(( ($(date +%s) - T0) / 3600 ))
  if [ "$HRS" -ge "$DEADLINE_H" ]; then
    echo "[l3] deadline reached (${HRS}h) -- stopping before round $ROUND"; break
  fi
  say "=== round $ROUND | model $MODEL | ${HRS}h elapsed ==="

  # ---- screen every deck ----------------------------------------------------------------
  MIR=$STATE/mirror_r$ROUND.json
  if [ -s "$MIR" ]; then
    say "reusing existing screen $MIR"
  else
    python3 - "$SHARDS" > $STATE/shards.txt <<'PY'
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
      PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "hf:$MODEL" \
          --max-games "$SCREEN_GAMES" --out "$STATE/mirror_r$ROUND.$i.json" \
          > "$STATE/screen_r$ROUND.$i.log" 2>&1 &
      i=$((i+1))
    done < $STATE/shards.txt
    say "launched $i screen shards"
    wait
    python3 - "$i" "$MIR" <<'PY' || { say "screen FAILED"; break; }
import json, sys
n, out = int(sys.argv[1]), sys.argv[2]
d = {}
for k in range(n):
    try:
        d.update(json.load(open(out.replace(".json", ".%d.json" % k)))["decks"])
    except Exception as e:
        print("shard %d unreadable: %s" % (k, e))
if not d:
    raise SystemExit(1)
json.dump({"decks": d}, open(out, "w"))
print("merged -> %d decks" % len(d))
PY
  fi

  python3 - "$MIR" "$STATE/history.tsv" "$ROUND" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))["decks"]
p = [v["p"] for v in d.values()]
row = (int(sys.argv[3]), len(p), statistics.median(p), sum(p)/len(p),
       sum(1 for v in d.values() if v["verdict"] == "WORSE"), sum(1 for x in p if x < .5))
with open(sys.argv[2], "a") as f:
    f.write("%d\t%d\t%.4f\t%.4f\t%d\t%d\n" % row)
print("[screen] round %d | decks %d | median %.1f%% | mean %.1f%% | WORSE %d | below50 %d"
      % (row[0], row[1], 100*row[2], 100*row[3], row[4], row[5]))
for k in sorted(d, key=lambda k: d[k]["p"])[:8]:
    print("   %-24s %.1f%%  %s" % (k, 100*d[k]["p"], d[k]["verdict"]))
PY

  # ---- targets --------------------------------------------------------------------------
  TARGETS=$(python3 -c "
import json, sys
d = json.load(open('$MIR'))['decks']
for name, ks in [('WORSE',   [k for k,v in d.items() if v['verdict']=='WORSE']),
                 ('below45', [k for k,v in d.items() if v['p']<0.45]),
                 ('below50', [k for k,v in d.items() if v['p']<0.50]),
                 ('weakest', sorted(d, key=lambda k: d[k]['p']))]:
    if ks:
        print('TIER=%s (%d decks)' % (name, len(ks)), file=sys.stderr)
        print(','.join(sorted(ks, key=lambda k: d[k]['p'])[:$MAX_TARGETS])); break
")
  say "targets: $TARGETS"
  [ -n "$TARGETS" ] || { say "no targets -- stopping"; break; }

  # ---- collect, THIS ROUND ONLY ---------------------------------------------------------
  DAG=$STATE/dagger_r$ROUND.jsonl.gz
  PYTHONPATH=cg-lib python3 tools/collect_dagger.py --decks "$TARGETS" \
      --model "hf:$MODEL" --games "$COLLECT_GAMES" --out "$DAG" \
      || { say "collect FAILED -- stopping"; break; }

  MIX=$REPO/data/rerank/l3_r$ROUND.jsonl.gz
  # A DIFFERENT base sample every round. The reservoir is seeded, so without this every round
  # trains on the same ~12% slice of the 2.87M pool and the rounds differ only by their DAgger --
  # the base contribution would be a constant, not a sample.
  python3 tools/mix_v40.py --base "$BASE" --dagger "$DAG" --valued "$VALUED" \
      --dagger-frac "$RATIO" --valued-frac "$VALUED_FRAC" --seed "$ROUND" --out "$MIX" \
      || { say "mix FAILED -- stopping"; break; }

  # ---- train from base ------------------------------------------------------------------
  OUT=/root/out/l3_r$ROUND
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 tools/train_rerank.py --data "$MIX" --out "$OUT" \
      --deadline-h 5 --max-samples 600000 --lr 2e-5 --pair-batch 32 --accum 12 --max-len 768 \
      --eval-n 2000 --grad-ckpt --margin-weight "$MARGIN" \
      || { say "train FAILED -- stopping"; break; }
  [ -f "$OUT/model.safetensors" ] || { say "no model saved -- stopping"; break; }

  say "round $ROUND done -> $OUT"
  MODEL=$OUT
  ROUND=$((ROUND+1))
  rm -f "$MIX"        # 200-400 MB each; the dagger file is what matters and it is kept
done
echo "[l3] LOOP ENDED"
