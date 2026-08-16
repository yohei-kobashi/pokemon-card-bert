#!/usr/bin/env bash
# Dusknoir-only SFT, continued from d41_r8 (best of the 6 checkpoints on this deck: 42.7%).
# The mix is dusknoir-piloted rows against the 11, quota'd by the 2026-08-08 leaderboard scout
# rather than uniformly -- marnie_grimmsnarl is 29.4% of the ladder and one eleventh of a
# uniform mix, so uniform would under-train the most common opponent by 3x.
set -u
say() { echo "[dsft $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
STATE=/root/dusk; mkdir -p "$STATE"
SRC=$REPO/data/rerank/v41_dusk.jsonl.gz
MIX=$REPO/data/rerank/dusk_mix.jsonl.gz
ROWS=${ROWS:-300000}
FROM=${FROM:-/root/out/d41_r8}
OUT=${OUT:-/root/out/dusk_r1}

say "raising generator throughput (11 matchups needs far more games/matchup than 2080 did)"
for p in $(pgrep -f gen_pool_v41 2>/dev/null); do kill "$p" 2>/dev/null; done
sleep 3
DECKS=$(python3 -c "import sys;sys.path.insert(0,'tools');import rl_config;print(','.join(rl_config.STAGE_C_TARGETS))")
GAMES=600 WORKERS=28 DECKS="$DECKS" PAIR_WITH=dragapult_dusknoir BASE="$SRC" \
  TARGET_ROWS=40000000 setsid nohup bash tools/gen_pool_v41.sh > /root/gen_dusk.log 2>&1 < /dev/null &
say "generator restarted with GAMES=600"

say "building the leaderboard-quota'd mix ($ROWS rows)"
python3 - "$SRC" "$MIX" "$ROWS" <<'PY'
import gzip, json, random, sys, collections
src, dst, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
# top-153 leaderboard shares (2026-08-08), renormalised over the ten opponents we generate.
# The dusknoir mirror is deliberately absent: zero teams on the ladder play this deck.
W = {"marnie_grimmsnarl": 32.8, "dudunsparce_box": 16.8, "alakazam_nz": 12.4,
     "ogerpon_mono": 10.3, "dragapult": 5.8, "alakazam": 5.8, "crustle_geco": 5.1,
     "crustle": 4.4, "mega_lucario_tr": 3.7, "cynthia_garchomp": 2.9}
tot = sum(W.values())
quota = {k: int(want * v / tot) for k, v in W.items()}
# reservoir per opponent so one pass over 4.4M rows suffices
res = {k: [] for k in W}
seen = collections.Counter()
rng = random.Random(11)
n = 0
with gzip.open(src, "rt") as f:
    for line in f:
        n += 1
        o = line.find('"opp":')
        if o < 0: continue
        d = json.loads(line)
        k = d.get("opp")
        if k not in quota: continue
        seen[k] += 1
        r = res[k]
        if len(r) < quota[k]:
            r.append(line)
        else:
            j = rng.randrange(seen[k])
            if j < quota[k]:
                r[j] = line
out = [ln for k in res for ln in res[k]]
rng.shuffle(out)
with gzip.open(dst, "wt") as g:
    g.writelines(out)
print("[mix] scanned %d rows -> %d written" % (n, len(out)))
for k in sorted(W, key=lambda x: -W[x]):
    print("   %-22s quota %6d  available %7d  taken %6d" % (k, quota[k], seen[k], len(res[k])))
PY
[ -s "$MIX" ] || { say "mix build FAILED"; exit 1; }

# The 11-deck re-evaluation owns the GPU with six concurrent shards; starting a training run
# on top of it would slow both and make the eval's timings meaningless.
say "waiting for the 11-deck evaluation to release the GPU"
for i in $(seq 1 90); do
  grep -aq EVAL11_DONE /root/eval11.log && break
  sleep 60
done
grep -aq EVAL11_DONE /root/eval11.log || say "eval11 still running after 90 min -- proceeding anyway"

say "continuing $FROM -> $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"; cp -r "$FROM"/. "$OUT"/ && rm -f "$OUT/rr_progress.json"
[ -f "$OUT/model.safetensors" ] || { say "STOP: $OUT not a usable checkpoint"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 tools/train_rerank.py --data "$MIX" --out "$OUT" --resume --deadline-h 4 \
  --max-samples "$ROWS" --lr 1e-5 --pair-batch 32 --accum 2 --max-len 512 \
  --eval-n 2000 --grad-ckpt --margin-weight 0.5 > "$STATE/train_r1.log" 2>&1 \
  || { say "train FAILED"; tail -6 "$STATE/train_r1.log"; exit 1; }
grep -aE "eval|FINAL|saved" "$STATE/train_r1.log" | tail -4

say "evaluating: dragapult_dusknoir only, 150 games vs engine_v2"
PYTHONPATH=cg-lib python3 tools/mirror_match.py --deck dragapult_dusknoir \
  --a engine --b "hf:$OUT" --max-games 150 --mirror --seed 1 \
  --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
  --out "$STATE/eval_r1.json" 2>&1 | tail -8
say "baseline for comparison: d41_r8 = 42.7% (64-86)"
say DUSK_SFT_DONE
