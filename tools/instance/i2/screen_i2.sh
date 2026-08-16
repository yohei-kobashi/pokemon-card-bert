#!/bin/bash
# Measure instance2's scheme-B model when its training ends. The chain that trains it has no
# evaluation step, so without this we would spend 5.3 hours and end up with a checkpoint and no
# number.
#
# THREE shards, not four. Four 4B scorers asked for 48 GB on a 47.4 GB card and shard 2 died of
# CUDA OOM, which is why the scheme-A baseline covers 47 of 63 decks.
#
# The SAME 47 DECKS the scheme-A baseline covers, so the comparison is paired. Screening the
# other 16 would add coverage but nothing comparable, and each deck costs ~7 minutes.
set -u
REPO=/root/ptcg/repo
NEW=/root/out/qwen3_4b_cfb_v40
LOG=/root/screen_i2.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[si2 $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "############ waiting for the scheme-B training ############"
while pgrep -f "[s]ft_teacher.py --model unsloth/Qwen3-4B-Base --data $REPO/data/sft/cf_b_v40" > /dev/null; do sleep 120; done
sleep 20
[ -f "$NEW/domain_embeddings.pt" ] || { say "STOP: $NEW has no domain_embeddings.pt -- the run did not reach its final save, so the added rows are gone"; exit 1; }
say "checkpoint ready: $NEW"

python3 - > /root/si2_shards.txt <<'PY'
import json
d = sorted(json.load(open("/root/mirror_cf.json"))["decks"])
for i in range(3):
    print(" ".join("--deck " + x for x in d[i::3]))
PY
say "screening $(wc -l < /root/si2_shards.txt) shards over $(json=/root/mirror_cf.json python3 -c "import json,os;print(len(json.load(open(os.environ['json']))['decks']))") decks"

i=0
while read -r DECKS; do
  [ -n "$DECKS" ] || continue
  PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "qwen:$NEW" \
      --max-games 40 --out /root/mirror_i2v40.$i.json > /root/si2_$i.log 2>&1 &
  i=$((i+1))
done < /root/si2_shards.txt
say "launched $i shards"
wait

python3 - $i <<'PY'
import json, statistics, math, collections, sys
out = {}
for k in range(int(sys.argv[1])):
    try:
        out.update(json.load(open("/root/mirror_i2v40.%d.json" % k))["decks"])
    except Exception as e:
        print("shard %d unreadable: %s" % (k, e))
if not out:
    raise SystemExit("no shard produced a result")
json.dump({"decks": out}, open("/root/mirror_i2v40.json", "w"))
old = json.load(open("/root/mirror_cf.json"))["decks"]
p = [v["p"] for v in out.values()]
c = collections.Counter(v["verdict"] for v in out.values())
print("[scheme B v40] decks %d | median %.1f%% | mean %.1f%% | %s | below50 %d"
      % (len(p), 100 * statistics.median(p), 100 * sum(p) / len(p), dict(c),
         sum(1 for x in p if x < .5)))
ks = sorted(set(old) & set(out))
d = [out[k]["p"] - old[k]["p"] for k in ks]
m = sum(d) / len(d); se = statistics.pstdev(d) / len(d) ** .5
w = lambda z, k: z[k]["verdict"] == "WORSE"
b01 = sum(1 for k in ks if not w(old, k) and w(out, k))
b10 = sum(1 for k in ks if w(old, k) and not w(out, k))
t = b01 + b10
pv = min(1.0, sum(math.comb(t, x) for x in range(0, min(b01, b10) + 1)) / 2 ** t * 2) if t else 1.0
print("[v40 vs scheme-A, PAIRED on %d decks] %+.4f +- %.4f  t %+.2f | became WORSE %d, left %d, p %.3f"
      % (len(ks), m, se, m / se, b01, b10, pv))
o = sorted(zip(ks, d), key=lambda x: x[1])
print("  biggest drops:", ", ".join("%s %+.0fpt" % (k, 100 * v) for k, v in o[:5]))
print("  biggest gains:", ", ".join("%s %+.0fpt" % (k, 100 * v) for k, v in o[-5:]))
print("VERDICT: %s" % ("scheme B v40 WINS" if m > 0 else "keep the scheme-A checkpoint"))
PY
say "SCREEN DONE"
