#!/usr/bin/env bash
# Is the 300-row probe's floor a DATA problem? The trainer memorises ONE row to 0.007, but 300
# rows stall at 0.985 against a floor of 0.278 -- that gap is contradiction, not capacity.
#
# The prime suspect is the attack labels. slowking's reference agents fire their signature
# attack on 11.4% of the decisions where it is legal, and d41_r8 fires Phantom Dive on 12.0% --
# the same number, because a turn offers the attack at every decision and ends after ONE. So
# `phantom_dive` (3,215 rows) and `lock_early` (1,141 rows) demand an attack at boards where the
# right play is to keep building, and the SAME board appears elsewhere labelled "develop".
# Strip those two rules and re-probe: if the floor drops, the contradiction was theirs.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
python3 - <<'PY'
import gzip, json, math, random, statistics as st
src, dst = "/root/rl/plan_r4.jsonl.gz", "/root/rl/plan_noatk.jsonl.gz"
ATK = {"phantom_dive", "lock_early"}
kept = drop = 0
with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as g:
    for line in f:
        r = json.loads(line)
        if set(r["rules"]) & ATK:
            drop += 1
            continue
        g.write(line); kept += 1
rows = [json.loads(x) for x in gzip.open(dst, "rt")]
random.Random(0).shuffle(rows)
H = lambda wc: (lambda p: -sum(q*math.log(q) for q in p))([x/sum(wc) for x in wc if x > 0])
print("kept %d rows, dropped %d attack-labelled | FLOOR probe300 %.4f | all %.4f"
      % (kept, drop, st.mean([H(r["wc"]) for r in rows[:300]]), st.mean([H(r["wc"]) for r in rows])))
PY
for cfg in "5e-6 8" "2e-5 8"; do
  set -- $cfg
  printf "NO-ATTACK  lr=%-6s accum=%-3s  " "$1" "$2"
  python3 tools/dusk_plan_train.py --data /root/rl/plan_noatk.jsonl.gz --model /root/out/d41_r8 \
    --out /root/out/plan_probe --probe --lr "$1" --epochs 30 --accum "$2" 2>&1 \
    | grep -aE "^FINAL|^PROBE" | tr "\n" " "
  echo
done
echo "--- control: the SAME two settings on the full data, for a like-for-like read ---"
for cfg in "5e-6 8"; do
  set -- $cfg
  printf "FULL       lr=%-6s accum=%-3s  " "$1" "$2"
  python3 tools/dusk_plan_train.py --data /root/rl/plan_r4.jsonl.gz --model /root/out/d41_r8 \
    --out /root/out/plan_probe --probe --lr "$1" --epochs 30 --accum "$2" 2>&1 \
    | grep -aE "^FINAL|^PROBE" | tr "\n" " "
  echo
done
echo NOATK_DONE
