#!/usr/bin/env bash
# Re-evaluate the reranker checkpoints on the 11 decks they were actually TRAINED for.
# The loop's 65-deck screen spends 54/65 of its games on decks no submission will ever use,
# and at 40 games/deck the 11 that matter swing +-8pt, so neither the level nor the ranking
# survives. Same method as the loop (mirror, --a engine --b hf:<dir>), 150 games/deck, no
# per-deck time cap -- the cap would let a slower model play fewer games and bias the compare.
set -u
say() { echo "[e11 $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
STATE=/root/eval11; mkdir -p "$STATE"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
GAMES=${GAMES:-150}

# round 8's training is the last thing worth finishing; round 9's screen is another 65-deck
# measurement of exactly the kind under question, so it is replaced by this one.
say "waiting for round 8 training to finish"
for i in $(seq 1 120); do
  grep -aq "round 8 done ->" /root/loop_deberta41/loop.log && break
  sleep 60
done
grep -aq "round 8 done ->" /root/loop_deberta41/loop.log || { say "round 8 never finished -- proceeding anyway"; }
pkill -f d41_run.sh; sleep 3; pkill -f "mirror_match.py"; sleep 5
say "d41 loop stopped; GPU free"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

DECKS=$(python3 -c "import sys;sys.path.insert(0,'tools');import rl_config;print(' '.join('--deck '+d for d in rl_config.STAGE_C_TARGETS))")
MODELS="v41_gte d41_r4 d41_r5 d41_r6 d41_r7 d41_r8"
j=0
for M in $MODELS; do
  [ -d "/root/out/$M" ] || { say "skip $M (missing)"; continue; }
  PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "hf:/root/out/$M" \
      --max-games "$GAMES" --mirror --seed 1 --mirror-so "$SO" \
      --out "$STATE/e11_$M.json" > "$STATE/e11_$M.log" 2>&1 &
  j=$((j+1))
done
say "launched $j model shards, $GAMES games/deck on 11 decks"
wait
python3 - <<'PY'
import json, math, os, statistics as st, sys
sys.path.insert(0, "/root/ptcg/repo/tools")
import rl_config
T = rl_config.STAGE_C_TARGETS
M = {}
for m in ("v41_gte", "d41_r4", "d41_r5", "d41_r6", "d41_r7", "d41_r8"):
    fn = "/root/eval11/e11_%s.json" % m
    if os.path.exists(fn):
        M[m] = json.load(open(fn))["decks"]
print("%-10s %8s %8s %9s %7s" % ("model", "mean", "median", "below50", "games"))
for m, d in M.items():
    p = [d[k]["p"] for k in T if k in d]
    print("%-10s %7.1f%% %7.1f%% %8d/%d %7d"
          % (m, 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5), len(p),
             sum(d[k]["w"]+d[k]["l"] for k in T if k in d)))
def paired(a, b):
    ks = [k for k in T if k in M.get(a, {}) and k in M.get(b, {})]
    if not ks: return
    dd = [M[b][k]["p"] - M[a][k]["p"] for k in ks]
    se = st.stdev(dd)/math.sqrt(len(dd))
    print("  %-10s -> %-10s %+.2fpt +- %.2f  t %+.2f  (up %d/%d)"
          % (a, b, 100*st.mean(dd), 100*se, st.mean(dd)/se if se else 0,
             sum(1 for x in dd if x > 0), len(dd)))
print("\nbackbone question (gte -> DeBERTa), same v41 format:")
for m in ("d41_r4", "d41_r5", "d41_r6", "d41_r7", "d41_r8"):
    if m in M: paired("v41_gte", m)
print("\nconsecutive DeBERTa rounds:")
ds = [m for m in ("d41_r4", "d41_r5", "d41_r6", "d41_r7", "d41_r8") if m in M]
for a, b in zip(ds, ds[1:]): paired(a, b)
print("\nper deck:")
hdr = "  %-20s" % "deck" + "".join(" %9s" % m for m in M)
print(hdr)
for k in T:
    print("  %-20s" % k + "".join(" %8.1f%%" % (100*M[m][k]["p"]) if k in M[m] else "        -" for m in M))
PY
say EVAL11_DONE
