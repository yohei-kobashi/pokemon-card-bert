#!/bin/bash
# screen -> collect DAgger on what it loses -> CONTINUE TRAINING the same model -> screen again.
#
# THE CHANGE THAT MATTERS. loop3 and loop4 trained every round from
# `Alibaba-NLP/gte-reranker-modernbert-base`, from scratch. That is not a bug on its own -- it is
# textbook DAgger, and the design note said so: "From-base is the safe default and loses nothing
# -- the accumulated DAgger data carries the progress, not the weights."
#
# But loop3 also stopped accumulating the DAgger. From-scratch training AND no data aggregation
# removes the only thing carrying progress while keeping the design that depends on it, so every
# round was an independent draw with no path to compounding. The evidence, four rounds:
#
#     v40    38.6%     l3_r1  39.1%     l3_r2  36.3%     l4_r3  35.7%
#
# l3_r1 (144,180 rows) and l4_r3 (150,000 rows) are the same recipe at the same size and land
# 3.4pt apart, so most of that spread is retraining variance -- larger than the paired screen's
# 1.25pt SE, which measures deck noise and not the re-draw. A loop like that cannot detect a 3pt
# improvement without repeating each recipe several times.
#
# So: progress lives in the WEIGHTS now. Each round continues the previous checkpoint
# (train_rerank --resume) and still collects fresh DAgger against the model that will consume it.
# This is what instance2's decoder does -- the only track that has produced a clear win.
#
# LR DROPS TO 1e-5 for a continued round. At 2e-5 over ~500,000 samples a warm start would be
# washed out and the round would be a retrain in all but name.
#
# The cost, from the note that first proposed continuing: the model becomes a function of round
# ORDER rather than of the data, which makes attribution harder. That is accepted deliberately.
#
# 1. THE TARGET LADDER IS BACK TO WORSE-ONLY. loop4 added a floor of 8 targets because loop3's
#    round 2 trained on ns_zoroark alone and the fleet lost 2.75pt. That evidence was CONFOUNDED:
#    the same round retrained from scratch on a 67,220-row mix, and the 8-target round that
#    replaced it lost 3.37pt -- more, not less. Widening was never shown to help. With the loop
#    now continuing the weights, narrow targeting deserves to be judged on its own, so the ladder
#    reverts to loop3's: WORSE first, and only if that tier is empty fall through to below45,
#    below50, weakest. MIN_TARGETS survives as an opt-in floor, defaulting to 0 (off).
#
#    THE SIDE EFFECT, named rather than discovered later: fewer targets means fewer collected
#    records, and the round is now 1.2M rows instead of 150k, so the same collection is 8x more
#    diluted. 8 decks x 72 games produced 17,929 rows = 1.60% of the mix; one deck would give
#    ~0.2%. COLLECT_GAMES therefore scales INVERSELY with the target count, so the collection is
#    about the same size whether the screen names one deck or eight.
#
# 2. A FIXED ROUND SIZE (--total). loop3 set the mix size to nd/RATIO, so the amount the round
#    trained on was a function of how many decks were bad enough to collect from. As the loop
#    succeeded it starved itself:
#
#        round 1   3 targets   7,209 DAgger rows   ->  144,180-row mix   3.6 epochs
#        round 2   1 target    3,361 DAgger rows   ->   67,220-row mix   7.8 epochs
#
#    Round 2 also carried only 3,361 of the 21,600 playout-valued attach records -- the data that
#    produced the v40 win -- because the valued share is a fraction of that same shrinking total.
#    With --total the round size is constant and the DAgger is capped at its 5% share.
#
# loop3's own change, kept -- EACH ROUND USES ONLY ITS OWN DAgger. dagger_loop2 accumulated every
# round's collection, and that is where the run went backwards -- paired on the same 63 decks:
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
KIND=${KIND:-rerank6}
MODEL=${MODEL:?set MODEL to the checkpoint this loop starts from}
BASE=${BASE:-$REPO/data/rerank/v40_base.jsonl.gz}
VALUED=${VALUED:-$REPO/data/rerank/v40_attach_q1.jsonl.gz,$REPO/data/rerank/v40_attach_q2.jsonl.gz}
RATIO=${RATIO:-0.05}
VALUED_FRAC=${VALUED_FRAC:-0.05}
SCREEN_GAMES=${SCREEN_GAMES:-40}
SHARDS=${SHARDS:-4}
# Collection is NOT the expensive part of a round: 3 decks x 72 games took 5 minutes against a
# 30-minute screen and a 5-hour training. So collect wide and let the mixer cap the share.
COLLECT_GAMES=${COLLECT_GAMES:-72}
MAX_TARGETS=${MAX_TARGETS:-20}
MIN_TARGETS=${MIN_TARGETS:-0}
MAX_GAMES=${MAX_GAMES:-400}
TOTAL=${TOTAL:-150000}
DEADLINE_H=${DEADLINE_H:-24}
MARGIN=${MARGIN:-0.5}
LR=${LR:-1e-5}
STATE=/root/loop_$KIND
LOG=$STATE/loop.log
mkdir -p "$STATE"
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[l6 $(date -u +%m-%d_%H:%M:%S) r$ROUND] $*"; }

T0=$(date +%s)
ROUND=${START_ROUND:-1}
while :; do
  HRS=$(( ($(date +%s) - T0) / 3600 ))
  if [ "$HRS" -ge "$DEADLINE_H" ]; then
    echo "[l6] deadline reached (${HRS}h) -- stopping before round $ROUND"; break
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
  # WORSE only. The ladder falls through to a wider tier ONLY when WORSE is empty -- a round with
  # no failing deck still has to train on something. MIN_TARGETS defaults to 0 and is an opt-in
  # floor, not the behaviour.
  TARGETS=$(python3 -c "
import json, sys
d = json.load(open('$MIR'))['decks']
by_p = sorted(d, key=lambda k: d[k]['p'])
worse = [k for k in by_p if d[k]['verdict'] == 'WORSE']
if worse:
    tier, pick = 'WORSE', list(worse)
else:
    for tier, ks in [('below45', [k for k in by_p if d[k]['p'] < 0.45]),
                     ('below50', [k for k in by_p if d[k]['p'] < 0.50]),
                     ('weakest', by_p)]:
        if ks:
            pick = list(ks); break
while len(pick) < $MIN_TARGETS:
    nxt = next((k for k in by_p if k not in pick), None)
    if nxt is None: break
    pick.append(nxt)
pick = sorted(pick, key=lambda k: d[k]['p'])[:$MAX_TARGETS]
print('TIER=%s WORSE=%d -> targets=%d' % (tier, len(worse), len(pick)), file=sys.stderr)
print(','.join(pick))
")
  say "targets: $TARGETS"
  [ -n "$TARGETS" ] || { say "no targets -- stopping"; break; }

  # ---- collect, THIS ROUND ONLY ---------------------------------------------------------
  # Games per deck scale inversely with the target count so the ROUND's collection is about the
  # same size either way. Measured: 8 decks x 72 games -> 17,929 records (~2,240 per deck), which
  # is 1.60% of a 1.2M-row mix. One deck at 72 games would be 0.2% and effectively absent.
  NT=$(echo "$TARGETS" | tr ',' '\n' | grep -c .)
  GAMES=$(( COLLECT_GAMES * 8 / NT ))
  [ "$GAMES" -gt "$MAX_GAMES" ] && GAMES=$MAX_GAMES
  [ "$GAMES" -lt "$COLLECT_GAMES" ] && GAMES=$COLLECT_GAMES
  say "$NT target(s) -> $GAMES games each"
  DAG=$STATE/dagger_r$ROUND.jsonl.gz
  PYTHONPATH=cg-lib python3 tools/collect_dagger.py --decks "$TARGETS" \
      --model "hf:$MODEL" --games "$GAMES" --out "$DAG" \
      || { say "collect FAILED -- stopping"; break; }

  MIX=$REPO/data/rerank/l6_r$ROUND.jsonl.gz
  # A DIFFERENT base sample every round. The reservoir is seeded, so without this every round
  # trains on the same ~12% slice of the 2.87M pool and the rounds differ only by their DAgger --
  # the base contribution would be a constant, not a sample.
  python3 tools/mix_v40.py --base "$BASE" --dagger "$DAG" --valued "$VALUED" \
      --dagger-frac "$RATIO" --valued-frac "$VALUED_FRAC" --total "$TOTAL" \
      --seed "$ROUND" --out "$MIX" \
      || { say "mix FAILED -- stopping"; break; }

  # ---- CONTINUE from the current model ---------------------------------------------------
  # train_rerank --resume reads the model out of --out, so the checkpoint being continued is
  # copied in first. rr_progress.json must NOT come with it: it records how many samples the
  # PREVIOUS round saw and would fast-forward past this round's entire mix.
  OUT=/root/out/l6_r$ROUND
  rm -rf "$OUT"; mkdir -p "$OUT"
  cp -r "$MODEL"/. "$OUT"/ || { say "could not seed $OUT from $MODEL"; break; }
  rm -f "$OUT/rr_progress.json"
  [ -f "$OUT/config.json" ] && [ -f "$OUT/model.safetensors" ] \
      || { say "STOP: $OUT is not a usable checkpoint to continue from"; break; }
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 tools/train_rerank.py --data "$MIX" --out "$OUT" --resume \
      --deadline-h 5 --max-samples 600000 --lr "$LR" --pair-batch 32 --accum 12 --max-len 768 \
      --eval-n 2000 --grad-ckpt --margin-weight "$MARGIN" \
      || { say "train FAILED -- stopping"; break; }
  [ -f "$OUT/model.safetensors" ] || { say "no model saved -- stopping"; break; }
  # A --resume that silently fell through to the base model trains for five hours and looks
  # normal. instance2 lost 10 GPU-hours to exactly this class of failure.
  # $OUT carries the round number, so this line is unique to this round inside the shared log.
  grep -q "RESUME from $OUT" "$LOG" \
      || { say "STOP: the run did not report RESUME -- it trained from base"; break; }

  say "round $ROUND done -> $OUT"
  MODEL=$OUT
  ROUND=$((ROUND+1))
  rm -f "$MIX"        # 200-400 MB each; the dagger file is what matters and it is kept
done
echo "[l6] LOOP ENDED"
