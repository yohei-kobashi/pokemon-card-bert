#!/usr/bin/env bash
# instance2: Stage 1 of the two-stage curriculum (docs/rl_stages_v2.md).
#
# Structurally the old dagger_loop_i2d with two changes, both of which are the point:
#
#   the GATE is 11 decks x 229 games, not 65 x 40. Same total games, same paired SE (1.14pt).
#       Narrowing the deck set WITHOUT rescaling the games would triple the SE to 2.73pt and
#       the "<= +1pt for two rounds" stop rule could then never fire on evidence.
#   the ROUND DATA is playout-measured Q labels, not DAgger. No collect_dagger step: instance1
#       produces the labels into $INBOX and this loop consumes whatever has landed. Uniform
#       per-game credit is the diagnosed cause of two 12-round plateaus, so this curriculum
#       does not use it.
#
# It waits for three things before the first round, each with a timeout that FAILS LOUD rather
# than proceeding on a wrong assumption: the old loop to be gone, the pricing to finish, and a
# first batch of Q labels to arrive.
#
#   nohup bash tools/stage1_loop.sh > /root/stage1_start.log 2>&1 &
set -u
REPO=${REPO:-/root/ptcg/repo}
STATE=${STATE:-/root/loop_stage1}
MODEL=${MODEL:-/root/out/i2_r7}
BASEQ=${BASEQ:-unsloth/Qwen3-4B-Base}
VOCAB=${VOCAB:-$REPO/data/cardfirst_b_v39.json}
BASE=${BASE:-$REPO/data/sft/v41_base_sft.jsonl.gz}
BASE_N=${BASE_N:-200000}
INBOX=${INBOX:-/root/qlabel_in}
PRICED=${PRICED:-/root/priced_engine.json}
ROUNDS=${ROUNDS:-4}
GATE_GAMES=${GATE_GAMES:-229}
SHARDS=${SHARDS:-3}
MIRROR_SO=${MIRROR_SO:-$REPO/data/kaggle_engine_ext/libcg_mirror.so}
SCREEN_SEED=${SCREEN_SEED:-1}
WAIT_OLD_MIN=${WAIT_OLD_MIN:-300}
WAIT_PRICE_MIN=${WAIT_PRICE_MIN:-240}
WAIT_LABEL_MIN=${WAIT_LABEL_MIN:-300}
MIN_LABELS=${MIN_LABELS:-1}

mkdir -p "$STATE" "$INBOX"
cd "$REPO" || exit 1
LOG=$STATE/loop.log
exec >> "$LOG" 2>&1
ROUND=${START_ROUND:-1}
say() { echo "[s1 $(date -u +%m-%d_%H:%M:%S) r$ROUND] $*"; }

DECKS=$(python3 -c 'import sys;sys.path.insert(0,"tools");import rl_config;print(",".join(rl_config.STAGE_C_TARGETS))')
say "=== Stage 1 | pilots: $DECKS"

wait_for() {   # $1 label  $2 timeout-min  $3 shell test
  local lbl="$1" lim="$2" test="$3" n=0
  while ! eval "$test"; do
    n=$((n + 1))
    if [ "$n" -ge "$lim" ]; then
      say "TIMEOUT waiting for $lbl after ${n} min -- STOPPING rather than guessing"; return 1
    fi
    [ $((n % 15)) -eq 1 ] && say "waiting for $lbl (${n} min)"
    sleep 60
  done
  say "$lbl is ready (${n} min)"
}

wait_for "the old curriculum loop to stop" "$WAIT_OLD_MIN" '! pgrep -f dagger_loop_i2 >/dev/null' || exit 1
# The pricing decides which cells instance1 targets. Missing it is not fatal -- the observed
# gaps are a usable prior and instance1 may already be generating from them -- so this warns
# and continues rather than exiting.
wait_for "the pricing to finish" "$WAIT_PRICE_MIN" "[ -s $PRICED ]" \
  || say "NOTE: no $PRICED. Proceeding on the OBSERVED gaps, which are unpriced."
wait_for "a first Q-label batch" "$WAIT_LABEL_MIN" \
  "[ \$(ls -1 $INBOX/*.jsonl.gz 2>/dev/null | wc -l) -ge $MIN_LABELS ]" || exit 1

screen_model() {   # $1 model  $2 out  $3 tag
  local M="$1" O="$2" T="$3" j=0
  [ -s "$O" ] && { say "reusing screen $O"; return 0; }
  python3 - "$SHARDS" "$DECKS" > "$STATE/shards.txt" <<'PYX'
import sys
n = int(sys.argv[1]); d = sys.argv[2].split(",")
for i in range(n):
    print(" ".join("--deck " + x for x in d[i::n]))
PYX
  while read -r DK; do
    [ -n "$DK" ] || continue
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DK --a engine --b "qwen:$M" \
        --max-games "$GATE_GAMES" --mirror --seed "$SCREEN_SEED" --mirror-so "$MIRROR_SO" \
        --out "$STATE/$T.$j.json" > "$STATE/screen_$T.$j.log" 2>&1 &
    j=$((j + 1))
  done < "$STATE/shards.txt"
  say "launched $j gate shards for $T ($M) at $GATE_GAMES games/deck"
  wait
  python3 - "$j" "$O" "$STATE/$T" <<'PYX'
import json, sys
n, out, stem = int(sys.argv[1]), sys.argv[2], sys.argv[3]
d = {}
for k in range(n):
    try:
        d.update(json.load(open("%s.%d.json" % (stem, k)))["decks"])
    except Exception as e:
        print("shard %d unreadable: %s" % (k, e))
if not d:
    raise SystemExit(1)
json.dump({"decks": d}, open(out, "w"))
print("merged -> %d decks" % len(d))
PYX
}

PREV=""
while [ "$ROUND" -le "$ROUNDS" ]; do
  say "=== round $ROUND | model $MODEL ==="
  GATE=$STATE/gate_r$ROUND.json
  screen_model "$MODEL" "$GATE" "gate_r$ROUND" || { say "gate FAILED"; break; }
  python3 - "$GATE" "$PREV" <<'PY'
import json, math, statistics as st, sys
d = json.load(open(sys.argv[1]))["decks"]
p = [v["p"] for v in d.values()]
print("[gate] %d decks | mean %.1f%% | median %.1f%% | below50 %d"
      % (len(p), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))
for k in sorted(d, key=lambda k: d[k]["p"]):
    print("   %-22s %.1f%%" % (k, 100*d[k]["p"]))
prev = sys.argv[2] if len(sys.argv) > 2 else ""
if prev:
    try:
        q = json.load(open(prev))["decks"]
    except Exception:
        raise SystemExit(0)
    both = sorted(set(d) & set(q))
    if len(both) > 2:
        dd = [d[k]["p"] - q[k]["p"] for k in both]
        m = st.mean(dd); se = st.stdev(dd)/math.sqrt(len(dd))
        print("[paired vs previous round, %d decks] %+.4f +- %.4f  t %+.2f"
              % (len(both), m, se, m/se if se else 0.0))
PY

  # The gate is the only 229-game-per-deck measurement of how well the LM pilots each of the 11,
  # and rl_config's Stage-1 pilot weights are a linear function of exactly that number. Publish it
  # so the weights stop reading the 40-game screen, whose noise they were mistaking for headroom
  # (dudunsparce_box: 37.5% at 40 games -> weight 0.200; 55.3% at 229 -> 0.043).
  cp "$GATE" "$REPO/evaluations/lm_mirror_screen.json" 2>/dev/null \
    && say "published gate_r$ROUND as the pilot-weight screen"

  # ---- consume every Q-label batch that has landed ----------------------------------------
  BATCH=$(ls -1 "$INBOX"/*.jsonl.gz 2>/dev/null | tr '\n' ',' | sed 's/,$//')
  [ -n "$BATCH" ] || { say "no Q-label batches in $INBOX -- stopping"; break; }
  say "consuming: $(echo "$BATCH" | tr ',' '\n' | wc -l) batch(es)"
  QSFT=$STATE/qlabel_r$ROUND.sft.jsonl.gz
  python3 tools/valued_to_sft.py --inp "$BATCH" --out "$QSFT" \
      || { say "valued_to_sft FAILED"; break; }

  MIX=$REPO/data/sft/s1_r$ROUND.jsonl.gz
  python3 tools/mix_sft_round.py --base "$BASE" --base-n "$BASE_N" \
      --valued "$QSFT" --seed "$ROUND" --out "$MIX" || { say "mix FAILED"; break; }

  OUT=/root/out/s1_r$ROUND
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # WARM-START PREFLIGHT. Copied from dagger_loop_i2d verbatim, not simplified: a warm start
  # that silently restores 0 LoRA tensors trains from scratch and looks like a normal seven-hour
  # run. That failure cost 10 GPU-hours once ([[loops-warm-start-switch]]). Five steps on 400
  # rows is the cheapest possible check that --init-from actually landed.
  rm -rf /root/out/s1_pre_r$ROUND
  python3 tools/instance/sft_teacher.py --model "$BASEQ" --data "$MIX" \
      --domain-tokens --card-first "$VOCAB" --init-from "$MODEL" \
      --out /root/out/s1_pre_r$ROUND --limit 400 --eval-n 0 --steps 5 \
      --bsz 32 --accum 1 --maxlen 896 --group-by-length --save-steps 100000 2>&1 \
      | grep -E "^\[warm\]|^\[cardfirst\]|^\[data\]|REFUSING|Error" | tee $STATE/pre_r$ROUND.txt
  grep -qE "^\[warm\] embedding rows restored by name: [0-9]{4,}" $STATE/pre_r$ROUND.txt \
      || { say "STOP: too few embedding rows restored"; break; }
  grep -q "^\[warm\] LoRA tensors restored: 0 " $STATE/pre_r$ROUND.txt \
      && { say "STOP: the LoRA did not load"; break; }
  rm -rf /root/out/s1_pre_r$ROUND
  say "warm-start preflight OK"

  TLOG=$STATE/train_r$ROUND.log
  python3 tools/instance/sft_teacher.py --model "$BASEQ" --data "$MIX" \
      --domain-tokens --card-first "$VOCAB" --init-from "$MODEL" \
      --out "$OUT" --limit 400000 --eval-n 4000 --epochs 1 \
      --bsz 32 --accum 1 --maxlen 896 --group-by-length --save-steps 1000 > "$TLOG" 2>&1
  if [ $? -ne 0 ]; then
    if grep -qiE "out of memory|CUDA error: out of memory" "$TLOG"; then
      say "bsz 32 ran out of memory -- retrying at the proven bsz 8 x accum 4"
      rm -rf "$OUT"
      python3 tools/instance/sft_teacher.py --model "$BASEQ" --data "$MIX" \
          --domain-tokens --card-first "$VOCAB" --init-from "$MODEL" \
          --out "$OUT" --limit 400000 --eval-n 4000 --epochs 1 \
          --bsz 8 --accum 4 --maxlen 896 --group-by-length --save-steps 1000 >> "$TLOG" 2>&1 \
          || { say "train FAILED at bsz 8 too -- stopping. See $TLOG"; break; }
    else
      say "train FAILED for a reason other than memory. Last lines:"
      tr '\r' '\n' < "$TLOG" | grep -av "^$" | tail -8
      break
    fi
  fi
  grep -aE "^\[peak\]|^\[saved\]" "$TLOG" | tail -2
  # The added vocabulary rows live in this file; without it the next round warm-starts from a
  # checkpoint whose domain tokens are gone.
  [ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no domain_embeddings.pt"; break; }

  say "round $ROUND done -> $OUT"
  # Batches are consumed, not accumulated: a round trains on the labels gathered against the
  # policy it is about to replace. Keeping them would mix labels collected against three
  # different policies, which is what made dagger_loop2 go backwards.
  mkdir -p "$INBOX/used" && mv "$INBOX"/*.jsonl.gz "$INBOX/used/" 2>/dev/null
  PREV=$GATE
  MODEL=$OUT
  ROUND=$((ROUND + 1))
  rm -f "$MIX"
done
say "STAGE 1 ENDED"
