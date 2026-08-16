#!/bin/bash
# v40: everything at once. attach_q2 lands -> convert -> mix -> train -> measure.
#
# WHY NOT A/B. Time is short and each arm costs 5h of training plus 2h of screening. Four
# changes ship together and only the total is measured; the fallback is exact, because
# rerank_loop2 and its screen (mirror_r2.json) are on disk and untouched.
#
#   new prompt      menu shows one entry per ACT. The cross-encoder ranks the deduped candidate
#                   list, so it was being shown 6.88 options for 5.53 acts; on decisions
#                   offering an attach the menu's attach share is inflated +4 to +6pt with sd
#                   9-11pt, set by how many copies of the energy are in hand.
#   attach_q1+q2    playout-VALUED attach records. q1 answers which target, q2 also answers
#                   whether to attach at all -- the model scores a non-attach above every attach
#                   on 42.4% of decisions that have a clear best attach target.
#   margin loss     pairwise hinge with the margin proportional to the measured Q gap, so a
#                   value-neutral pair asks for nothing instead of a coin-flip label.
#   dagger_r1 only  round 1 gained +3.06pt paired; accumulating rounds 2-3 gave it back.
#
# The screen of the RESULT is compared to mirror_r2.json PAIRED per deck. Each model is screened
# with the prompt it was trained on -- rerank_loop2 predates menu_dedup, so it must NOT be
# re-screened under the new format.
set -u
REPO=/root/ptcg/repo
LOG=/root/chain_v40.log
Q2=$REPO/data/rerank/attach_q2.jsonl.gz
MIX=$REPO/data/rerank/v40_mix.jsonl.gz
OUT=/root/out/rerank_v40
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[v40 $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "############ waiting for attach_q2 ############"
while pgrep -f "[a]ttach_label.py --games" > /dev/null; do sleep 60; done
sleep 20
[ -s "$Q2" ] || { say "STOP: $Q2 is empty -- the generator produced nothing"; exit 1; }
say "attach_q2 ready: $(zcat $Q2 | wc -l) records"

say "=== 1/5 convert attach_q2 to the v40 prompt ==="
python3 tools/menu_dedup_pool.py --inp "$Q2" --out data/rerank/v40_attach_q2.jsonl.gz \
  || { say "STOP: conversion failed"; exit 1; }

say "=== 2/5 mix ==="
python3 tools/mix_v40.py --base data/rerank/v40_base.jsonl.gz \
  --dagger data/rerank/v40_dagger_r1.jsonl.gz \
  --valued data/rerank/v40_attach_q1.jsonl.gz,data/rerank/v40_attach_q2.jsonl.gz \
  --dagger-frac 0.10 --valued-frac 0.05 --out "$MIX" \
  || { say "STOP: mix failed"; exit 1; }

# A mix that silently lost the valued records would train as a plain imitation run and look
# like a fair test of the whole package. Refuse rather than discover it afterwards.
python3 - "$MIX" <<'PY' || { say "STOP: the mix carries no valued records"; exit 1; }
import gzip, json, sys
n = v = 0
for line in gzip.open(sys.argv[1], "rt"):
    n += 1
    if n % 7 == 0 and json.loads(line).get("qvals"):
        v += 1
print("[check] %d rows, %d of every 7th carry qvals (%.2f%% of the sample)"
      % (n, v, 100.0 * v / max(1, n // 7)))
raise SystemExit(0 if v else 1)
PY

say "=== 3/5 train (rerank_loop2's recipe + the value-margin term) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 tools/train_rerank.py --data "$MIX" --out "$OUT" \
  --deadline-h 5 --max-samples 600000 --lr 2e-5 --pair-batch 32 --accum 12 --max-len 768 \
  --eval-n 2000 --grad-ckpt --margin-weight 0.5 \
  || { say "STOP: training failed"; exit 1; }
[ -f "$OUT/model.safetensors" ] || { say "STOP: no model saved"; exit 1; }

say "=== 4/5 attach quality on the held-out valued set ==="
python3 tools/eval_attach.py --model "$OUT" --data data/rerank/v40_attach_held.jsonl.gz \
  2>&1 | grep -viE "warning|loading weights|dtype"

say "=== 5/5 mirror screen vs engine_v2, 4 shards ==="
python3 - 4 > /root/v40_shards.txt <<'PY'
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
  PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "hf:$OUT" \
      --max-games 40 --out /root/mirror_v40.$i.json > /root/v40_screen_$i.log 2>&1 &
  i=$((i+1))
done < /root/v40_shards.txt
say "launched $i screen shards"
wait

python3 - $i <<'PY'
import json, statistics, math, collections, sys
out = {}
for k in range(int(sys.argv[1])):
    try:
        out.update(json.load(open("/root/mirror_v40.%d.json" % k))["decks"])
    except Exception as e:
        print("shard %d unreadable: %s" % (k, e))
if not out:
    raise SystemExit("no shard produced a result")
json.dump({"decks": out}, open("/root/mirror_v40.json", "w"))
old = json.load(open("/root/loop_rerank/mirror_r2.json"))["decks"]
p = [v["p"] for v in out.values()]
c = collections.Counter(v["verdict"] for v in out.values())
print("[v40]  decks %d | median %.1f%% | mean %.1f%% | WORSE %d | below50 %d"
      % (len(p), 100 * statistics.median(p), 100 * sum(p) / len(p), c["WORSE"],
         sum(1 for x in p if x < .5)))
ks = sorted(set(old) & set(out))
d = [out[k]["p"] - old[k]["p"] for k in ks]
m = sum(d) / len(d); se = statistics.pstdev(d) / len(d) ** .5
w = lambda z, k: z[k]["verdict"] == "WORSE"
b01 = sum(1 for k in ks if not w(old, k) and w(out, k))
b10 = sum(1 for k in ks if w(old, k) and not w(out, k))
t = b01 + b10
pv = min(1.0, sum(math.comb(t, x) for x in range(0, min(b01, b10) + 1)) / 2 ** t * 2) if t else 1.0
print("[v40 vs rerank_loop2, PAIRED on %d decks] %+.4f +- %.4f  t %+.2f | "
      "became WORSE %d, left WORSE %d, exact p %.3f" % (len(ks), m, se, m / se, b01, b10, pv))
print("VERDICT: %s" % ("v40 WINS -- promote it" if m > 0 and b01 <= b10 else
                       "keep rerank_loop2"))
PY
say "CHAIN DONE"
