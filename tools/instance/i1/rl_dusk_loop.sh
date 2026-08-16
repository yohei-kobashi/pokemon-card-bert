#!/usr/bin/env bash
# instance1: shaped-return policy gradient on mirror self-play. No engine_v2 in the training
# path -- rl_pg_train refuses the round if more than 2% of decisions were answered by the
# agent's engine_v2 fallback. engine_v2 remains the GATE, because an independent yardstick is
# the only reason we could tell that three SFT continuation rounds lost (45.5 -> 41.5 -> 30.7).
#
# The reference is PINNED at the starting checkpoint and never re-anchored. Re-anchoring each
# round is what removed the only thing holding instance2's SFT in place.
set -u
say() { echo "[rl $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
STATE=/root/rl; mkdir -p "$STATE"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
DECK=dragapult_dusknoir
REF=${REF:-/root/out/d41_r8}
SEEDS=${SEEDS:-30}
GROUP=${GROUP:-4}
TEMP=${TEMP:-1.0}
GATE_GAMES=${GATE_GAMES:-400}
DEADLINE=${DEADLINE:-1786924740}          # 2026-08-16 23:59 UTC

BEST=$(cat "$STATE/best.txt" 2>/dev/null || echo "$REF")
BESTP=$(cat "$STATE/best_p.txt" 2>/dev/null || echo 0.455)
R=$(cat "$STATE/round.txt" 2>/dev/null || echo 0)
say "start | best=$BEST ($BESTP) | reference PINNED at $REF | round=$R"

while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  R=$((R + 1)); echo "$R" > "$STATE/round.txt"
  OUT=/root/out/rl_r$R
  ROLL=$STATE/roll_$R.jsonl.gz
  say "=== round $R | rollout from $BEST | $SEEDS deals x $GROUP | temp $TEMP ==="

  PYTHONPATH=cg-lib python3 tools/rl_rollout.py --model "$BEST" --deck "$DECK" \
    --seeds "$SEEDS" --group "$GROUP" --temp "$TEMP" --seed-base $((700000 + R * 1000)) \
    --mirror-so "$SO" --out "$ROLL" > "$STATE/roll_$R.log" 2>&1 \
    || { say "ROLLOUT FAILED"; tail -5 "$STATE/roll_$R.log"; break; }
  grep -aE "group divergence|WARNING" "$STATE/roll_$R.log" | tail -2

  say "train"
  PYTHONPATH=cg-lib python3 tools/rl_pg_train.py --data "$ROLL" --model "$BEST" \
    --ref "$REF" --out "$OUT" --epochs 1 --lr 5e-6 --beta-kl 0.05 --beta-h 0.01 \
    > "$STATE/train_$R.log" 2>&1 \
    || { say "TRAIN FAILED"; tail -6 "$STATE/train_$R.log"; break; }
  grep -aE "\[pairing\]|\[data\]|FINAL|saved" "$STATE/train_$R.log" | tail -5
  [ -f "$OUT/model.safetensors" ] || { say "no checkpoint written"; break; }
  rm -f "$ROLL"

  say "gate: challenger and incumbent, $GATE_GAMES games each, same seeds, vs engine_v2"
  for tag in chal inc; do
    [ "$tag" = chal ] && M="$OUT" || M="$BEST"
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py --deck "$DECK" \
      --a engine --b "hf:$M" --max-games "$GATE_GAMES" --mirror --seed 1 --mirror-so "$SO" \
      --out "$STATE/g_${R}_$tag.json" > "$STATE/g_${R}_$tag.log" 2>&1 &
  done
  wait
  python3 - "$R" <<'PY' > "$STATE/verdict_$R.txt"
import json, sys, math
R = sys.argv[1]
def rd(t):
    d = json.load(open("/root/rl/g_%s_%s.json" % (R, t)))["decks"]["dragapult_dusknoir"]
    return d["p"], d["w"], d["l"]
c, cw, cl = rd("chal"); i, iw, il = rd("inc")
se = math.sqrt(c*(1-c)/max(1, cw+cl) + i*(1-i)/max(1, iw+il))
print("challenger %.1f%% (%d-%d) | incumbent %.1f%% (%d-%d) | delta %+.1fpt +- %.1f"
      % (100*c, cw, cl, 100*i, iw, il, 100*(c-i), 100*se))
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
done
say RL_LOOP_DONE
