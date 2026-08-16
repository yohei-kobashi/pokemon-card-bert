#!/usr/bin/env bash
# Re-run the round-5 gate on all ELEVEN decks, then start round-6 collection from whichever
# checkpoint the gate says is current.
#
# Why this exists. The round-5 gate launched three shards into a GPU that still held the
# training process, so four ~12 GiB residents met a 47.38 GiB card and two shards died inside
# resize_token_embeddings -- one with CUDA OOM, one with CUSOLVER_STATUS_INTERNAL_ERROR from the
# same exhaustion. The verdict was then computed from the one surviving shard and reported as
# "up 3/3", which is three decks out of eleven, and the +1.79pt mean was carried by crustle_geco
# alone (+11.1 against -0.9 and +2.6). The protagonist deck itself was in a dead shard.
#
# Three things are different here:
#   1. mirror_match.py now passes mean_resizing=False (shipped separately) -- the Cholesky init
#      that blew up allocated a large temporary and then had its output overwritten four lines
#      later by domain_embeddings.pt, so removing it changes nothing but the peak.
#   2. the GPU is checked empty before launch and the shards are staggered, so three model loads
#      never peak together.
#   3. a missing shard is a HARD FAILURE. The pooling in dpo_round5.sh printed "shard unreadable"
#      and carried on, which is how a 3-deck verdict got read as an 11-deck one.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
STATE=/root/loop_dpo
MIRROR_SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
say() { echo "[rg5 $(date -u +%m-%d_%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 1. disk
# 90% full with 26G free, and round 6 writes traces plus a checkpoint. Everything removed here is
# either refuted (teacher9b*: 60x params, 0 gain -- see the teacher-9b-adds-nothing measurement),
# superseded (qwen3_4b_cf1 by cfb_v40; i2_r1..r4 by i2_r7, which is the PINNED DPO reference and
# is kept, as are i2_r5/r6), or a smoke test. All checkpoints are LoRA adapters on
# unsloth/Qwen3-4B-Base, which lives in ~/.cache/huggingface and is NOT touched.
say "disk before: $(df -h /root | awk 'NR==2{print $4" free ("$5" used)"}')"
for d in teacher9b_v39 teacher9b qwen3_4b_cf1 cfb_preflight2 smoke smoke2 smoke3 smoke5 smoke6 \
         i2_r1 i2_r2 i2_r3 i2_r4; do
    [ -d "/root/out/$d" ] && rm -rf "/root/out/$d" && say "  removed out/$d"
done
say "disk after:  $(df -h /root | awk 'NR==2{print $4" free ("$5" used)"}')"
for keep in /root/out/i2_r7/adapter_model.safetensors /root/out/dpo_r4/adapter_config.json \
            /root/out/dpo_r5/adapter_config.json; do
    [ -f "$keep" ] || { say "STOP: cleanup removed something it must not have ($keep)"; exit 1; }
done

# ---------------------------------------------------------------- 2. GPU must be empty
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 2000 ]; then
    say "STOP: GPU already holds ${used} MiB -- this is exactly what killed the round-5 gate"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    exit 1
fi
say "GPU clear (${used} MiB)"

# ---------------------------------------------------------------- 3. the gate
# Same decks, same --max-games 229, same --seed 1 as rounds 4 and 5: the adoption statistic is
# PAIRED against gate_dpo4, and a changed seed would silently turn it into an unpaired one.
DECKS=$(python3 -c "import sys;sys.path.insert(0,'tools');import rl_config;print(','.join(rl_config.STAGE_C_TARGETS))")
python3 - "$DECKS" > "$STATE/shards.txt" <<'PYX'
import sys
d = sys.argv[1].split(",")
for i in range(3):
    print(" ".join("--deck " + x for x in d[i::3]))
PYX
say "gate: 11 decks x 229 vs engine_v2, 3 shards"
rm -f "$STATE"/gate_dpo5b.*.json
j=0
while read -r DK; do
    [ -n "$DK" ] || continue
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DK --a engine --b "qwen:/root/out/dpo_r5" \
        --max-games 229 --mirror --seed 1 --mirror-so "$MIRROR_SO" \
        --out "$STATE/gate_dpo5b.$j.json" > "$STATE/gate_dpo5b.$j.log" 2>&1 &
    j=$((j+1))
    sleep 120     # stagger the model loads; the peak is at load, not during play
done < "$STATE/shards.txt"
say "launched $j gate shards"; wait

miss=0
for k in 0 1 2; do
    [ -s "$STATE/gate_dpo5b.$k.json" ] || { say "SHARD $k PRODUCED NO JSON"; tail -20 "$STATE/gate_dpo5b.$k.log"; miss=1; }
done
[ "$miss" -eq 0 ] || { say "STOP: incomplete gate -- refusing to decide adoption on a subset"; exit 1; }

# ---------------------------------------------------------------- 4. the verdict
python3 - > "$STATE/verdict_r5.txt" 2>&1 <<'PY'
import json, math, statistics as st
def load(p, n=3):
    d = {}
    for k in range(n):
        d.update(json.load(open("/root/loop_dpo/%s.%d.json" % (p, k)))["decks"])
    return d
cur = load("gate_dpo5b"); r4 = load("gate_dpo4")
base = json.load(open("/root/loop_stage1/gate_r1.json"))["decks"]
p = [v["p"] for v in cur.values()]
print("[gate] dpo_r5 | %d decks | mean %.1f%% | median %.1f%% | below50 %d"
      % (len(cur), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))
def sp(v, s):
    w, l = v["seat%d" % s]; return w/(w+l) if (w+l) else float("nan")
res = {}
def paired(key, name, ref):
    ks = sorted(set(cur) & set(ref))
    if not ks: return
    dd = [cur[k]["p"] - ref[k]["p"] for k in ks]
    se = st.stdev(dd)/math.sqrt(len(dd))
    t = st.mean(dd)/se if se else 0.0
    res[key] = (100*st.mean(dd), 100*se, t, len(ks))
    print("[paired vs %s, %d decks] %+.4f +- %.4f  t %+.2f  (up %d/%d)"
          % (name, len(ks), st.mean(dd), se, t, sum(1 for x in dd if x > 0), len(dd)))
    for s in (0, 1):
        d2 = [sp(cur[k], s) - sp(ref[k], s) for k in ks]
        print("      seat%d %+.2fpt +- %.2f" % (s, 100*st.mean(d2),
              100*st.stdev(d2)/math.sqrt(len(d2))))
paired("r4", "dpo_r4 (the adopted checkpoint)", r4)
paired("sft", "i2_r7 (cumulative)", base)
for k in sorted(set(cur) & set(r4), key=lambda k: cur[k]["p"]-r4[k]["p"]):
    print("   %-22s r4 %5.1f%% -> r5 %5.1f%%  (%+.1f)"
          % (k, 100*r4[k]["p"], 100*cur[k]["p"], 100*(cur[k]["p"]-r4[k]["p"])))

# Adoption. NOT_WORSE, not "significantly better": the same checkpoint re-scores 2.6pt apart on
# this gate, so demanding significance would reject every real gain as well as every fake one.
# dpo_r5 is adopted unless it is worse by more than the noise floor. Round 6 proceeds either
# way -- only the checkpoint it starts from changes.
d, se, t, n = res.get("r4", (0.0, 0.0, 0.0, 0))
worse = (d <= -2.0 and t <= -2.0)
adopt = "dpo_r4" if worse else "dpo_r5"
halt = st.mean(p) < 0.40      # not a checkpoint question: something in the harness is broken
print("VERDICT delta_vs_r4 %+.2fpt +- %.2f (t %+.2f, %d decks) -> ADOPT %s%s"
      % (d, se, t, n, adopt, "  [HALT: mean below 40%]" if halt else ""))
open("/root/loop_dpo/adopt.txt", "w").write(("HALT" if halt else adopt) + "\n")
PY
cat "$STATE/verdict_r5.txt"
ADOPT=$(cat "$STATE/adopt.txt" 2>/dev/null || echo HALT)
[ "$ADOPT" = "HALT" ] && { say "STOP: gate says the harness is broken, not the checkpoint"; exit 1; }
say "adopted checkpoint: $ADOPT"

# ---------------------------------------------------------------- 5. round 6 collection
sleep 60      # let the gate's three scorers release the card before three more load
say "round 6 collection from $ADOPT (opponents now include slowking)"
MODEL="qwen:/root/out/$ADOPT" bash /root/collect_r6.sh
say "REGATE_AND_R6_DONE"
