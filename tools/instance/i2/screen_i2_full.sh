#!/bin/bash
# Screen instance2's scheme-B model on ALL 63 decks, and fill in the 16 decks the scheme-A
# baseline is missing so the pair is complete and every future run compares against the same set.
#
# The baseline has a hole because I ran the scheme-A screen as 4 shards: four 4B scorers asked
# for 48 GB on a 47.4 GB card and one died of CUDA OOM. THREE shards (36 GB) is the ceiling here.
#
# Each shard does its share of BOTH jobs, sequentially, so the two models never hold the card at
# the same time and the 79 deck-screens spread evenly instead of leaving one shard idle.
#
# COST, from the scheme-A screen's own timing (47 decks / 4 shards in 2h36m = ~13 min per deck):
# 79 deck-screens over 3 shards is ~26 each, so ~5.7 hours. That is the price of a complete
# baseline; the 47 decks already resolve ~1.4pt, so this buys coverage, not precision.
set -u
REPO=/root/ptcg/repo
NEW=/root/out/qwen3_4b_cfb_v40
OLD=/root/out/qwen3_4b_cf1
LOG=/root/screen_i2.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[si2 $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "############ waiting for the scheme-B training ############"
while pgrep -f "[s]ft_teacher.py --model unsloth/Qwen3-4B-Base --data $REPO/data/sft/cf_b_v40" > /dev/null; do sleep 120; done
sleep 20
[ -f "$NEW/domain_embeddings.pt" ] || { say "STOP: $NEW has no domain_embeddings.pt -- the run never reached its final save, so the added rows are gone"; exit 1; }
say "checkpoint ready: $NEW"

python3 - <<'PY'
import json
import sys
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
import library
allc = sorted(library.list_decks())
have = set(json.load(open("/root/mirror_cf.json"))["decks"])
miss = [d for d in allc if d not in have]
print("all %d | scheme-A baseline has %d | missing %d" % (len(allc), len(have), len(miss)),
      file=sys.stderr)
for i in range(3):
    with open("/root/si2_new_%d.txt" % i, "w") as f:
        f.write(" ".join("--deck " + x for x in allc[i::3]))
    with open("/root/si2_old_%d.txt" % i, "w") as f:
        f.write(" ".join("--deck " + x for x in miss[i::3]))
PY

for i in 0 1 2; do
  (
    N=$(cat /root/si2_new_$i.txt); O=$(cat /root/si2_old_$i.txt)
    [ -n "$N" ] && PYTHONPATH=cg-lib python3 tools/mirror_match.py $N --a engine \
        --b "qwen:$NEW" --max-games 40 --out /root/mirror_i2v40.$i.json > /root/si2_new_$i.log 2>&1
    [ -n "$O" ] && PYTHONPATH=cg-lib python3 tools/mirror_match.py $O --a engine \
        --b "qwen:$OLD" --max-games 40 --out /root/mirror_cf_fill.$i.json > /root/si2_old_$i.log 2>&1
  ) &
done
say "launched 3 shards (scheme-B on all 63, then scheme-A on the missing 16)"
wait
say "shards finished"

python3 - <<'PY'
import json, statistics, math, collections
new, fill = {}, {}
for k in range(3):
    for path, sink in (("/root/mirror_i2v40.%d.json" % k, new),
                       ("/root/mirror_cf_fill.%d.json" % k, fill)):
        try:
            sink.update(json.load(open(path))["decks"])
        except Exception as e:
            print("unreadable %s: %s" % (path, e))
if not new:
    raise SystemExit("no scheme-B shard produced a result")
old = json.load(open("/root/mirror_cf.json"))["decks"]
old.update(fill)                      # the completed 63-deck scheme-A baseline
json.dump({"decks": old}, open("/root/mirror_cf_full.json", "w"))
json.dump({"decks": new}, open("/root/mirror_i2v40.json", "w"))
print("scheme-A baseline now covers %d decks (was %d, filled %d)"
      % (len(old), len(old) - len(fill), len(fill)))

for nm, d in (("scheme-A", old), ("scheme-B v40", new)):
    p = [v["p"] for v in d.values()]
    c = collections.Counter(v["verdict"] for v in d.values())
    print("[%-12s] decks %d | median %.1f%% | mean %.1f%% | %s | below50 %d"
          % (nm, len(p), 100*statistics.median(p), 100*sum(p)/len(p), dict(c),
             sum(1 for x in p if x < .5)))

ks = sorted(set(old) & set(new))
d = [new[k]["p"] - old[k]["p"] for k in ks]
m = sum(d)/len(d); se = statistics.pstdev(d)/len(d)**.5
w = lambda z, k: z[k]["verdict"] == "WORSE"
b01 = sum(1 for k in ks if not w(old, k) and w(new, k))
b10 = sum(1 for k in ks if w(old, k) and not w(new, k))
t = b01 + b10
pv = min(1.0, sum(math.comb(t, x) for x in range(0, min(b01, b10)+1))/2**t*2) if t else 1.0
print("[v40 vs scheme-A, PAIRED on %d decks] %+.4f +- %.4f  t %+.2f | became WORSE %d, left %d, p %.3f"
      % (len(ks), m, se, m/se, b01, b10, pv))
o = sorted(zip(ks, d), key=lambda x: x[1])
print("  biggest drops:", ", ".join("%s %+.0fpt" % (k, 100*v) for k, v in o[:5]))
print("  biggest gains:", ", ".join("%s %+.0fpt" % (k, 100*v) for k, v in o[-5:]))
print("VERDICT: %s" % ("scheme B v40 WINS" if m > 0 and b01 <= b10 else "keep the scheme-A checkpoint"))
PY
say "SCREEN DONE"
