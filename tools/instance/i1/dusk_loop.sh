#!/usr/bin/env bash
# Train dragapult_dusknoir toward parity with engine_v2, one gated round at a time.
#
# WHY A GATE AND NOT A SCHEDULE. Round 1 narrowed the mix to this one deck and came back
# -3.7pt (45.2 -> 41.5 at 400 games), so "more focused data" is not automatically progress.
# But the d41 checkpoints differ by 12.4pt ON THIS DECK (r8 45.2 vs r6 32.8) at the same
# recipe, which is far above the 2.5pt measurement SE -- round-to-round variance is the
# biggest lever available, and the way to use it is to keep the best, not to average.
#
# So each round: build a mix, continue the CURRENT BEST, then measure the challenger AND
# re-measure the incumbent on the same 400 seeds. Adopt only on a win. Re-measuring the
# incumbent every round costs 20 minutes and stops the loop ratcheting up on noise.
#
# ANCHOR. Round 1 was 100% dusknoir; the anchor fraction mixes general 11-deck rows back in,
# because a model that was trained on eleven decks and is then shown only one drifts
# ([[narrow-dagger-overfits]] measured the same shape). It alternates so the loop samples both.
set -u
say() { echo "[dl $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
STATE=/root/dusk; mkdir -p "$STATE"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
DUSK=$REPO/data/rerank/v41_dusk.jsonl.gz
BASE=$REPO/data/rerank/v41_base11.jsonl.gz
DEADLINE=${DEADLINE:-1786924740}          # 2026-08-16 23:59 UTC
ROWS=${ROWS:-400000}
GATE_GAMES=${GATE_GAMES:-400}

BEST=$(cat "$STATE/best.txt" 2>/dev/null || echo /root/out/d41_r8)
BESTP=$(cat "$STATE/best_p.txt" 2>/dev/null || echo 0.452)
R=$(cat "$STATE/round.txt" 2>/dev/null || echo 1)
say "start: best=$BEST p=$BESTP round=$R"

while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  R=$((R + 1)); echo "$R" > "$STATE/round.txt"
  # Weight interpolation between d41_r8 and the round-1 fine-tune DIPPED (36.7-40.4% against
  # 45.2 / 41.5 at the endpoints), so the two are NOT linearly connected: one 300k-row epoch at
  # lr 1e-5 walked the model out of its basin, and no post-hoc average can undo that. Constrain
  # the training instead. L2-SP anchors the weights to this round's own init, which is the axis
  # rehearsal cannot reach -- rehearsal only protects what the replayed rows happen to cover.
  case $((R % 4)) in
    0) ANCHOR=0.40; LR=5e-6; L2SP=1e-3;;
    1) ANCHOR=0.25; LR=1e-5; L2SP=1e-3;;
    2) ANCHOR=0.40; LR=5e-6; L2SP=1e-2;;
    *) ANCHOR=0.25; LR=5e-6; L2SP=0;;
  esac
  OUT=/root/out/dusk_r$R
  say "=== round $R | from $BEST | anchor $ANCHOR | lr $LR | l2sp $L2SP | $ROWS rows ==="

  MIX=$REPO/data/rerank/dusk_mix_r$R.jsonl.gz
  python3 - "$DUSK" "$BASE" "$MIX" "$ROWS" "$ANCHOR" "$R" <<'PY'
import gzip, json, random, sys, collections
dusk, base, dst, want, anchor, rnd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
# top-153 leaderboard shares (2026-08-08), renormalised over the ten opponents we generate.
W = {"marnie_grimmsnarl": 32.8, "dudunsparce_box": 16.8, "alakazam_nz": 12.4,
     "ogerpon_mono": 10.3, "dragapult": 5.8, "alakazam": 5.8, "crustle_geco": 5.1,
     "crustle": 4.4, "mega_lucario_tr": 3.7, "cynthia_garchomp": 2.9}
n_d = int(want * (1 - anchor)); n_a = want - n_d
tot = sum(W.values())
quota = {k: int(n_d * v / tot) for k, v in W.items()}
rng = random.Random(1000 + rnd)
res = {k: [] for k in W}; seen = collections.Counter(); scanned = 0
with gzip.open(dusk, "rt") as f:
    for line in f:
        scanned += 1
        d = json.loads(line)
        k = d.get("opp")
        if k not in quota or d.get("deck") != "dragapult_dusknoir":
            continue
        seen[k] += 1
        r = res[k]
        if len(r) < quota[k]:
            r.append(line)
        else:
            j = rng.randrange(seen[k])
            if j < quota[k]:
                r[j] = line
out = [ln for k in res for ln in res[k]]
# anchor: a plain reservoir over the eleven-deck pool, no quota -- its job is to keep the
# model's general distribution alive, not to teach a matchup.
anc = []; m = 0
with gzip.open(base, "rt") as f:
    for line in f:
        m += 1
        if len(anc) < n_a:
            anc.append(line)
        else:
            j = rng.randrange(m)
            if j < n_a:
                anc[j] = line
out += anc
rng.shuffle(out)
with gzip.open(dst, "wt") as g:
    g.writelines(out)
print("[mix] dusknoir %d + anchor %d = %d rows (scanned %d dusk / %d base)"
      % (len(out) - len(anc), len(anc), len(out), scanned, m), flush=True)
PY
  [ -s "$MIX" ] || { say "mix FAILED"; break; }

  rm -rf "$OUT"; mkdir -p "$OUT"; cp -r "$BEST"/. "$OUT"/ && rm -f "$OUT/rr_progress.json"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 tools/train_rerank.py --data "$MIX" --out "$OUT" --resume --deadline-h 5 \
    --max-samples "$ROWS" --lr "$LR" --pair-batch 32 --accum 2 --max-len 512 \
    --eval-n 2000 --grad-ckpt --margin-weight 0.5 --l2sp "$L2SP" > "$STATE/train_r$R.log" 2>&1 \
    || { say "train FAILED"; tail -5 "$STATE/train_r$R.log"; break; }
  grep -aE "FINAL" "$STATE/train_r$R.log" | tail -2
  rm -f "$MIX"

  say "gate: challenger and incumbent, $GATE_GAMES games each, same seeds"
  for tag in chal inc; do
    [ "$tag" = chal ] && M="$OUT" || M="$BEST"
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py --deck dragapult_dusknoir \
      --a engine --b "hf:$M" --max-games "$GATE_GAMES" --mirror --seed 1 --mirror-so "$SO" \
      --out "$STATE/g_${R}_$tag.json" > "$STATE/g_${R}_$tag.log" 2>&1 &
  done
  wait
  python3 - "$R" "$BESTP" <<'PY' > "$STATE/verdict_$R.txt"
import json, sys, math
R, bp = sys.argv[1], float(sys.argv[2])
def rd(t):
    d = json.load(open("/root/dusk/g_%s_%s.json" % (R, t)))["decks"]["dragapult_dusknoir"]
    return d["p"], d["w"], d["l"]
c, cw, cl = rd("chal"); i, iw, il = rd("inc")
se = math.sqrt(c * (1 - c) / max(1, cw + cl) + i * (1 - i) / max(1, iw + il))
print("challenger %.1f%% (%d-%d) | incumbent %.1f%% (%d-%d) | delta %+.1fpt +- %.1f"
      % (100 * c, cw, cl, 100 * i, iw, il, 100 * (c - i), 100 * se))
print("ADOPT" if c > i else "KEEP")
print("%.4f" % max(c, i))
PY
  cat "$STATE/verdict_$R.txt"
  if grep -q ADOPT "$STATE/verdict_$R.txt"; then
    BEST="$OUT"; say "adopted $OUT"
  else
    rm -rf "$OUT"; say "kept $BEST"
  fi
  BESTP=$(tail -1 "$STATE/verdict_$R.txt")
  echo "$BEST" > "$STATE/best.txt"; echo "$BESTP" > "$STATE/best_p.txt"
  say "best now $BEST at $(python3 -c "print('%.1f%%' % (100*float('$BESTP')))")"
  awk -v p="$BESTP" 'BEGIN{ if (p+0 >= 0.50) print "[dl] PARITY REACHED" }'
done
say DUSK_LOOP_DONE
