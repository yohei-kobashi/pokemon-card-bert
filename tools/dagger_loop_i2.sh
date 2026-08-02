#!/bin/bash
# instance2's DAgger loop: screen 63 decks -> collect on the ones it loses -> retrain -> repeat.
#
# The decoder twin of tools/dagger_loop3.sh, with the two decisions that loop settled:
#   * EACH ROUND USES ONLY ITS OWN DAgger. Accumulating every round's collection is what ran the
#     reranker loop backwards (+3.06pt, then -4.25pt with 16 decks turning WORSE).
#   * THE BASE IS RESAMPLED EVERY ROUND (mix_sft_round.py --seed $ROUND). instance2 previously
#     re-used one frozen 193,919-row file, so the base was a constant, not a sample.
#
# The base pool itself is new: data/sft/v40_base_sft.jsonl.gz, 5,733,620 rows converted from
# instance1's reranker pool by tools/rerank_to_sft.py. Besides being 30x the old file, it is the
# first instance2 base rendered with menu_dedup -- the old one showed one entry per menu POSITION
# while rl_config.PROMPT_FMT (what mirror_match/collect_dagger actually render at inference) shows
# one per ACT, so every base row was training on a format the model never meets in play.
#
# THREE shards, not four: four 4B scorers ask for 48 GB on a 47.4 GB card and one dies of CUDA
# OOM, which is how the scheme-A baseline lost 16 decks.
set -u
REPO=/root/ptcg/repo
MODEL=${MODEL:?set MODEL to the checkpoint this loop starts from}
BASEQ=${BASEQ:-unsloth/Qwen3-4B-Base}
BASE=${BASE:-$REPO/data/sft/v40_base_sft.jsonl.gz}
BASE_N=${BASE_N:-200000}
VALUED=${VALUED:-$REPO/data/sft/valued_q1.jsonl.gz,$REPO/data/sft/valued_q234.jsonl.gz}
VOCAB=${VOCAB:-$REPO/data/cardfirst_b_v39.json}
SCREEN_GAMES=${SCREEN_GAMES:-40}
COLLECT_GAMES=${COLLECT_GAMES:-48}
MAX_TARGETS=${MAX_TARGETS:-9}
SHARDS=${SHARDS:-3}
DEADLINE_H=${DEADLINE_H:-72}
STATE=/root/loop_i2
LOG=$STATE/loop.log
mkdir -p "$STATE"
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[i2l $(date -u +%m-%d_%H:%M:%S) r$ROUND] $*"; }

T0=$(date +%s)
ROUND=${START_ROUND:-1}
PREV=""
while :; do
  HRS=$(( ($(date +%s) - T0) / 3600 ))
  if [ "$HRS" -ge "$DEADLINE_H" ]; then
    echo "[i2l] deadline reached (${HRS}h) -- stopping before round $ROUND"; break
  fi
  say "=== round $ROUND | model $MODEL | ${HRS}h elapsed ==="

  # ---- screen every deck ----------------------------------------------------------------
  # Round 1 is pre-seeded from the finished v40 screen (/root/mirror_i2v40.json): it screened
  # exactly this checkpoint on all 63 decks, so re-running it would burn 5.7 h to reproduce it.
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
      PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "qwen:$MODEL" \
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

  # Paired against the PREVIOUS round on the decks both screened. The medians in history.tsv
  # swing ~9pt on this sample size; the paired difference resolves 1.2-1.6pt, so it is the only
  # line here that can say whether a round moved anything.
  python3 - "$MIR" "$STATE/history.tsv" "$ROUND" "$PREV" <<'PY'
import json, math, statistics, sys
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
prev = sys.argv[4] if len(sys.argv) > 4 else ""
if prev:
    try:
        q = json.load(open(prev))["decks"]
    except Exception as e:
        print("[paired] previous screen unreadable: %s" % e); raise SystemExit(0)
    both = sorted(set(d) & set(q))
    if len(both) > 2:
        diff = [d[k]["p"] - q[k]["p"] for k in both]
        m = sum(diff) / len(diff)
        se = statistics.stdev(diff) / math.sqrt(len(diff))
        print("[paired vs previous round, %d decks] %+.4f +- %.4f  t %+.2f"
              % (len(both), m, se, m / se if se else 0.0))
        srt = sorted(both, key=lambda k: d[k]["p"] - q[k]["p"])
        print("  drops: " + ", ".join("%s %+.0fpt" % (k, 100*(d[k]["p"]-q[k]["p"])) for k in srt[:5]))
        print("  gains: " + ", ".join("%s %+.0fpt" % (k, 100*(d[k]["p"]-q[k]["p"])) for k in srt[-5:]))
PY

  # ---- targets --------------------------------------------------------------------------
  # The Qwen screen resolves to `undecided` on every deck (SPRT never concludes at 40 games on a
  # pilot this close to the margin), so the WORSE tier is normally empty and the ladder falls
  # through to below45 -- that is expected here, not a failure.
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

  # ---- collect, THIS ROUND ONLY, sharded ------------------------------------------------
  DAG=$STATE/dagger_r$ROUND.jsonl.gz
  if [ -s "$DAG" ]; then
    say "reusing existing collection $DAG"
  else
    python3 - "$TARGETS" "$SHARDS" > $STATE/cshards.txt <<'PY'
import sys
t = [x for x in sys.argv[1].split(",") if x]
n = int(sys.argv[2])
for i in range(n):
    print(",".join(t[i::n]))
PY
    i=0
    while read -r DECKS; do
      [ -n "$DECKS" ] || continue
      PYTHONPATH=cg-lib nohup python3 tools/collect_dagger.py --decks "$DECKS" \
          --model "qwen:$MODEL" --games "$COLLECT_GAMES" --seed "$ROUND" \
          --out "$STATE/dagger_r$ROUND.$i.jsonl.gz" \
          > "$STATE/collect_r$ROUND.$i.log" 2>&1 &
      i=$((i+1))
    done < $STATE/cshards.txt
    say "launched $i collect shards"
    wait
    cat $STATE/dagger_r$ROUND.[0-9].jsonl.gz > "$DAG" || { say "collect merge FAILED"; break; }
    rm -f $STATE/dagger_r$ROUND.[0-9].jsonl.gz
  fi
  NDAG=$(zcat "$DAG" | wc -l)
  [ "$NDAG" -gt 1000 ] || { say "collection is only $NDAG rows -- stopping"; break; }
  say "collected $NDAG rows"

  # ---- convert the collection to the decoder's schema -----------------------------------
  # collect_dagger writes the RERANK schema. Its `menu_index` field is the RAW option index and
  # is stale under menu_dedup; rerank_to_sft derives the target from the rendered menu instead
  # and verifies the identity per record.
  DAGSFT=$STATE/dagger_r$ROUND.sft.jsonl.gz
  python3 tools/rerank_to_sft.py --inp "$DAG" --out "$DAGSFT" \
      || { say "dagger conversion FAILED"; break; }

  MIX=$REPO/data/sft/i2_r$ROUND.jsonl.gz
  python3 tools/mix_sft_round.py --base "$BASE" --base-n "$BASE_N" \
      --dagger "$DAGSFT" --valued "$VALUED" --seed "$ROUND" --out "$MIX" \
      || { say "mix FAILED"; break; }

  # ---- train ----------------------------------------------------------------------------
  OUT=/root/out/i2_r$ROUND
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # A warm start that silently restores 0 LoRA tensors trains from scratch and looks like a
  # normal run for five hours. This preflight cost 10 GPU-hours to learn once.
  rm -rf /root/out/i2_pre_r$ROUND
  python3 tools/instance/sft_teacher.py --model "$BASEQ" --data "$MIX" \
      --domain-tokens --card-first "$VOCAB" --init-from "$MODEL" \
      --out /root/out/i2_pre_r$ROUND --limit 400 --eval-n 0 --steps 5 \
      --bsz 8 --accum 4 --maxlen 896 --group-by-length --save-steps 100000 2>&1 \
      | grep -E "^\[warm\]|^\[cardfirst\]|^\[data\]|REFUSING|Error" | tee $STATE/pre_r$ROUND.txt
  grep -qE "^\[warm\] embedding rows restored by name: [0-9]{4,}" $STATE/pre_r$ROUND.txt \
      || { say "STOP: too few embedding rows restored"; break; }
  grep -q "^\[warm\] LoRA tensors restored: 0 " $STATE/pre_r$ROUND.txt \
      && { say "STOP: the LoRA did not load"; break; }
  rm -rf /root/out/i2_pre_r$ROUND
  say "warm-start preflight OK"

  python3 tools/instance/sft_teacher.py --model "$BASEQ" --data "$MIX" \
      --domain-tokens --card-first "$VOCAB" --init-from "$MODEL" \
      --out "$OUT" --limit 400000 --eval-n 4000 --epochs 1 \
      --bsz 8 --accum 4 --maxlen 896 --group-by-length --save-steps 1000 \
      || { say "train FAILED -- stopping"; break; }
  [ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no domain_embeddings.pt -- the added rows are lost"; break; }

  say "round $ROUND done -> $OUT"
  PREV=$MIR
  MODEL=$OUT
  ROUND=$((ROUND+1))
  rm -f "$MIX"
done
echo "[i2l] LOOP ENDED"
