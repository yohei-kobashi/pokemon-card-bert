#!/usr/bin/env bash
# Unattended: can DeBERTa-v3-base replace the reranker backbone, on v41 data?
#
# Runs to completion with no input. Order is the cheap-first one:
#
#   0  materialise deberta-v3-base locally (its HF repo ships only pytorch_model.bin, which
#      transformers >=5 refuses to torch.load under torch <2.6)
#   1  DEPLOY CHECK: vocab sweep -> prune -> ONNX -> weight-only INT8 -> argmax agreement.
#      If this fails DeBERTa cannot ship, so its 5-hour training is SKIPPED and the run goes
#      straight to the control. Knowing that costs 30 minutes instead of 5 hours.
#   2  REFERENCE: screen the current v40-trained checkpoint under the v41 prompt. Without it a
#      v41 number cannot be read -- a change could be the format, the backbone, or the data.
#   3  DeBERTa-v3-base trained on v41
#   4  gte-reranker-modernbert-base trained on v41 (the control: same data, same budget, same
#      max-len, so the only difference is the backbone)
#
# Both trainings use --max-len 512 because that is DeBERTa's hard limit. Measured on v41:
# 0.32% of (state, candidate) pairs exceed it, and truncation is LEFT, so those lose the head
# of DECK[] and keep the board and the menu.
set -u
REPO=${REPO:-/root/ptcg/repo}
DATA=${DATA:-$REPO/data/rerank/v41_base.jsonl.gz}
STATE=${STATE:-/root/bench_v41}
REF_MODEL=${REF_MODEL:-/root/out/l6_r8}
DEBERTA_SRC=${DEBERTA_SRC:-microsoft/deberta-v3-base}
DEBERTA_DIR=${DEBERTA_DIR:-/root/deberta_v3_base_hf}
GTE=${GTE:-Alibaba-NLP/gte-reranker-modernbert-base}
SCREEN_GAMES=${SCREEN_GAMES:-40}
SHARDS=${SHARDS:-4}
MIRROR_SO=${MIRROR_SO:-$REPO/data/kaggle_engine_ext/libcg_mirror.so}
DEADLINE_H=${DEADLINE_H:-5}
MAXLEN=${MAXLEN:-512}
MAXSAMP=${MAXSAMP:-600000}

mkdir -p "$STATE"
cd "$REPO"
LOG=$STATE/bench.log
exec >> "$LOG" 2>&1
say() { echo "[bench $(date -u +%m-%d_%H:%M:%S)] $*"; }

screen_model() {   # $1 model dir/name, $2 merged out json, $3 tag
  local SMODEL="$1" SOUT="$2" STAG="$3" j=0
  [ -s "$SOUT" ] && { say "reusing screen $SOUT"; return 0; }
  python3 - "$SHARDS" > "$STATE/shards.txt" <<'PYX'
import sys
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
import library
d = sorted(library.list_decks()); n = int(sys.argv[1])
for i in range(n):
    print(" ".join("--deck " + x for x in d[i::n]))
PYX
  while read -r DECKS; do
    [ -n "$DECKS" ] || continue
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "hf:$SMODEL" \
        --max-games "$SCREEN_GAMES" --mirror --seed 1 --mirror-so "$MIRROR_SO" \
        --out "$STATE/${STAG}.$j.json" > "$STATE/screen_${STAG}.$j.log" 2>&1 &
    j=$((j+1))
  done < "$STATE/shards.txt"
  say "screening $STAG ($SMODEL) on $j shards"
  wait
  python3 - "$j" "$SOUT" "$STATE/${STAG}" <<'PYX'
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
import statistics as st
p = [v["p"] for v in d.values()]
print("[screen] %s | decks %d | mean %.1f%% | median %.1f%%"
      % (out, len(p), 100*st.mean(p), 100*st.median(p)))
PYX
}

train_one() {      # $1 base model, $2 out dir, $3 tag
  local M="$1" O="$2" T="$3"
  if [ -s "$O/rr_progress.json" ] && [ -f "$O/config.json" ]; then
    say "reusing checkpoint $O"; return 0
  fi
  say "training $T from $M -> $O"
  python3 tools/train_rerank.py --data "$DATA" --model "$M" --out "$O" \
      --deadline-h "$DEADLINE_H" --max-samples "$MAXSAMP" --lr 1e-5 \
      --pair-batch 32 --accum 12 --max-len "$MAXLEN" --eval-n 2000 \
      --grad-ckpt --margin-weight 0.5 > "$STATE/train_$T.log" 2>&1
  local rc=$?
  tail -3 "$STATE/train_$T.log"
  [ $rc -eq 0 ] || { say "$T TRAINING FAILED (rc $rc)"; return 1; }
  return 0
}

say "================ v41 backbone bench: data $DATA ($(zcat "$DATA" | wc -l) rows)"

# ---- stage 0: local safetensors copy of deberta ---------------------------------------------
if [ ! -f "$DEBERTA_DIR/config.json" ]; then
  say "stage 0: materialising $DEBERTA_SRC -> $DEBERTA_DIR"
  python3 - "$DEBERTA_SRC" "$DEBERTA_DIR" <<'PYX' || say "stage 0 FAILED"
import sys, torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
src, dst = sys.argv[1], sys.argv[2]
cfg = AutoConfig.from_pretrained(src, num_labels=1)
model = AutoModelForSequenceClassification.from_config(cfg)
sd = torch.load(hf_hub_download(src, "pytorch_model.bin"), map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
real = [k for k in missing if not k.startswith(("classifier.", "pooler."))]
print("loaded: %d missing (%d outside the head), %d unexpected" % (len(missing), len(real), len(unexpected)))
assert len(real) <= 5, real[:5]
model.save_pretrained(dst, safe_serialization=True)
AutoTokenizer.from_pretrained(src).save_pretrained(dst)
print("saved ->", dst)
PYX
fi

# ---- stage 1: does the DEPLOY path work on deberta? ------------------------------------------
SKIP_DEBERTA=0
if [ ! -s "$STATE/deploy_ok" ]; then
  say "stage 1: deploy check (sweep -> prune -> onnx -> int8)"
  if python3 tools/sweep_vocab_rerank.py --data "$DATA" --tokenizer "$DEBERTA_DIR" \
        --out "$STATE/keep_ids_deberta.json" --limit 200000 > "$STATE/sweep.log" 2>&1 \
     && python3 tools/prune_vocab_rerank.py --model "$DEBERTA_DIR" \
        --keep "$STATE/keep_ids_deberta.json" --data "$DATA" --work "$STATE/onnx_deberta" \
        --max-len "$MAXLEN" > "$STATE/prune.log" 2>&1; then
    tail -6 "$STATE/sweep.log"; tail -12 "$STATE/prune.log"
    touch "$STATE/deploy_ok"; say "stage 1 PASSED -- deberta can ship"
  else
    tail -20 "$STATE/prune.log" 2>/dev/null || tail -20 "$STATE/sweep.log"
    say "stage 1 FAILED -- deberta cannot ship; SKIPPING its training"
    SKIP_DEBERTA=1
  fi
else
  say "stage 1 already passed"
fi

# ---- stage 2: reference -- the v40 checkpoint read under the v41 prompt -----------------------
say "stage 2: reference screen of $REF_MODEL under v41"
screen_model "$REF_MODEL" "$STATE/ref_v40ckpt_on_v41.json" "ref" || say "reference screen failed"

# ---- stage 3: deberta on v41 -----------------------------------------------------------------
if [ "$SKIP_DEBERTA" = 0 ]; then
  if train_one "$DEBERTA_DIR" /root/out/v41_deberta deberta; then
    screen_model /root/out/v41_deberta "$STATE/deberta_v41.json" "deberta" || say "deberta screen failed"
  fi
else
  say "stage 3 skipped"
fi

# ---- stage 4: the control, same data and budget ----------------------------------------------
if train_one "$GTE" /root/out/v41_gte gte; then
  screen_model /root/out/v41_gte "$STATE/gte_v41.json" "gte" || say "gte screen failed"
fi

# ---- stage 5: paired comparison ---------------------------------------------------------------
say "stage 5: paired summary"
python3 - "$STATE" <<'PYX'
import json, math, os, statistics as st, sys
S = sys.argv[1]
runs = [("v40ckpt-on-v41", "ref_v40ckpt_on_v41.json"),
        ("deberta-v41", "deberta_v41.json"), ("gte-v41", "gte_v41.json")]
got = {}
for name, f in runs:
    p = os.path.join(S, f)
    if os.path.exists(p):
        got[name] = json.load(open(p))["decks"]
for name, d in got.items():
    p = [v["p"] for v in d.values()]
    print("%-18s decks %d  mean %5.1f%%  median %5.1f%%  below50 %d"
          % (name, len(p), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))
names = list(got)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        ks = sorted(set(got[a]) & set(got[b]))
        if len(ks) < 2:
            continue
        dd = [got[b][k]["p"] - got[a][k]["p"] for k in ks]
        m = st.mean(dd); se = st.stdev(dd)/math.sqrt(len(dd))
        print("  paired %-18s -> %-18s n=%d %+.4f +- %.4f  t %+.2f"
              % (a, b, len(ks), m, se, m/se if se else 0))
PYX
say "================ DONE"
