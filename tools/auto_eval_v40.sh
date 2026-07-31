#!/bin/bash
# Fires when the v40 training exits: ablation, then the same 63-deck mirror screen v39 got,
# then a side-by-side.
#
# --max-games 150 matches the v39 screen exactly. A shorter cap would still give comparable
# WORSE verdicts (those resolve in well under 60 games) but not comparable undecideds, and the
# undecided band is where a real improvement would first appear.
set -u
REPO=/root/ptcg/repo
MODEL=${MODEL:-/root/out/rerank_gte_v40}
GAMES=${GAMES:-150}
LOG=/root/eval_v40.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[eval $(date -u +%H:%M:%S)] $*"; }

say "waiting for train_rerank to exit"
while pgrep -f "tools/train_rerank" > /dev/null; do sleep 120; done
sleep 30

if [ ! -f "$MODEL/model.safetensors" ] && [ ! -f "$MODEL/pytorch_model.bin" ]; then
  say "STOP: no model at $MODEL -- training did not reach a save, so there is nothing to score."
  exit 1
fi
say "model ready: $MODEL"

say "=== ablation (does it read DECK[] any better than v39?) ==="
python3 tools/ablate_rerank.py --models "$MODEL" \
    --data data/rerank/v40_mix.rerank.jsonl.gz --eval-n 2000 --max-len 768 --by-turn \
    || say "ablation failed (continuing to the mirror, which is the real gate)"

say "=== 63-deck mirror screen, ${GAMES} games/deck ==="
DECKS=$(PYTHONPATH=cg-lib python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'cg-lib')
import library; print(' '.join('--deck '+d for d in sorted(library.list_decks())))")
PYTHONPATH=cg-lib python3 tools/mirror_match.py $DECKS --a engine --b "hf:$MODEL" \
    --max-games "$GAMES" --out /root/mirror_v40.json || { say "screen FAILED"; exit 1; }

say "=== v39 vs v40 ==="
python3 - <<'PY'
import json
a = json.load(open("/root/mirror_fleet.json"))["decks"]      # v39
b = json.load(open("/root/mirror_v40.json"))["decks"]        # v40
common = sorted(set(a) & set(b))
import statistics
pa = [a[k]["p"] for k in common]
pb = [b[k]["p"] for k in common]
d = [b[k]["p"] - a[k]["p"] for k in common]
print("decks compared %d" % len(common))
print("  v39 median %.1f%%  mean %.1f%%  WORSE %d"
      % (100*statistics.median(pa), 100*sum(pa)/len(pa),
         sum(1 for k in common if a[k]["verdict"] == "WORSE")))
print("  v40 median %.1f%%  mean %.1f%%  WORSE %d"
      % (100*statistics.median(pb), 100*sum(pb)/len(pb),
         sum(1 for k in common if b[k]["verdict"] == "WORSE")))
print("  paired change: mean %+.1f pp  improved %d/%d decks"
      % (100*sum(d)/len(d), sum(1 for x in d if x > 0), len(d)))
print("\nbiggest moves:")
for k in sorted(common, key=lambda k: -(b[k]["p"] - a[k]["p"]))[:6]:
    print("  %-22s %.1f%% -> %.1f%%  (%+.1f)" % (k, 100*a[k]["p"], 100*b[k]["p"],
                                                 100*(b[k]["p"] - a[k]["p"])))
for k in sorted(common, key=lambda k: (b[k]["p"] - a[k]["p"]))[:6]:
    print("  %-22s %.1f%% -> %.1f%%  (%+.1f)" % (k, 100*a[k]["p"], 100*b[k]["p"],
                                                 100*(b[k]["p"] - a[k]["p"])))
PY
say "EVAL DONE"
