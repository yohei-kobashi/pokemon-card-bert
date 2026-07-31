#!/bin/bash
# Unattended: 9B SFT finishes -> mirror screen -> pick target decks -> DAgger -> re-SFT.
#
# Sizing. The 9B scores a decision in ~0.14 s against the reranker's 0.018 s, so ~10 s of model
# time per game. The reranker's 63-deck / 150-game screen took 2.5 h; the same shape here would
# take a day. The screen is therefore capped at --max-games 60 (a collapsed deck resolves in
# under 40) and collection runs only on the decks the screen flags.
#
# Target selection CASCADES, so a run where nothing is proven WORSE still produces work:
#   1. verdict WORSE
#   2. win rate below 45%   -- losing by more than the equivalence margin, even if unproven
#   3. win rate below 50%   -- losing at all
#   4. the weakest N decks   -- there is always something to improve
# The first non-empty tier wins, and which tier fired is printed, because "we trained on tier 3"
# and "we trained on tier 1" mean very different things about the model.
set -u
REPO=/root/ptcg/repo
ADAPTER=${ADAPTER:-/root/out/teacher9b_v39}
BASE_SFT=${BASE_SFT:-/root/ptcg/repo/data/sft/v39_0731.jsonl.gz}
SCREEN_GAMES=${SCREEN_GAMES:-60}
COLLECT_GAMES=${COLLECT_GAMES:-24}
MAX_TARGETS=${MAX_TARGETS:-20}
RATIO=${RATIO:-0.5}
LOG=/root/chain9b.log
cd "$REPO"
exec >> "$LOG" 2>&1

say() { echo "[chain $(date -u +%H:%M:%S)] $*"; }

say "waiting for the SFT to finish"
while pgrep -f "sft_teacher.py" | grep -qv "^$$\$"; do
  pgrep -f "tools/instance/sft_teacher.py" > /dev/null || break
  sleep 120
done
sleep 60

# The 2,971 domain-token embedding rows are written only at the END of training and live
# nowhere else -- no adapter, no checkpoint carries them. Without the file the model cannot be
# loaded correctly at all, so stop rather than screen a half-built model.
if [ ! -f "$ADAPTER/domain_embeddings.pt" ]; then
  say "STOP: $ADAPTER/domain_embeddings.pt missing -- the SFT did not reach its final save, so"
  say "      the added embedding rows are lost and the checkpoint cannot be scored."
  exit 1
fi
say "adapter ready: $ADAPTER"

say "=== 1/4 mirror screen (qwen, ${SCREEN_GAMES} games/deck) ==="
DECKS=$(PYTHONPATH=cg-lib python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'cg-lib')
import library; print(' '.join('--deck '+d for d in sorted(library.list_decks())))")
PYTHONPATH=cg-lib python3 tools/mirror_match.py $DECKS --a engine --b "qwen:$ADAPTER" \
    --max-games "$SCREEN_GAMES" --out /root/mirror_9b.json || { say "screen FAILED"; exit 1; }

say "=== 2/4 target selection ==="
TARGETS=$(python3 -c "
import json
d=json.load(open('/root/mirror_9b.json'))['decks']
tiers=[('WORSE',      [k for k,v in d.items() if v['verdict']=='WORSE']),
       ('below 45%',  [k for k,v in d.items() if v['p']<0.45]),
       ('below 50%',  [k for k,v in d.items() if v['p']<0.50]),
       ('weakest',    sorted(d, key=lambda k: d[k]['p'])[:$MAX_TARGETS])]
for name, ks in tiers:
    if ks:
        ks=sorted(ks, key=lambda k: d[k]['p'])[:$MAX_TARGETS]
        import sys; print('TIER=%s'%name, file=sys.stderr)
        print(','.join(ks)); break
")
say "targets: $TARGETS"
[ -n "$TARGETS" ] || { say "no targets -- stopping"; exit 1; }

say "=== 3/4 DAgger collection (${COLLECT_GAMES} games/deck) ==="
PYTHONPATH=cg-lib python3 tools/collect_dagger.py --decks "$TARGETS" \
    --model "qwen:$ADAPTER" --games "$COLLECT_GAMES" \
    --out /root/ptcg/repo/data/rerank/dagger_9b.jsonl.gz || { say "collect FAILED"; exit 1; }

python3 tools/dagger_to_sft.py --dagger /root/ptcg/repo/data/rerank/dagger_9b.jsonl.gz \
    --base "$BASE_SFT" --ratio "$RATIO" \
    --out /root/ptcg/repo/data/sft/v39_dagger.jsonl.gz || { say "convert FAILED"; exit 1; }

say "=== 4/4 re-SFT ==="
python3 tools/instance/sft_teacher.py --domain-tokens \
    --data /root/ptcg/repo/data/sft/v39_dagger.jsonl.gz \
    --out /root/out/teacher9b_v39d --limit 150000 --epochs 1 --bsz 8 --accum 4 \
    --eval-n 4000 --save-steps 400
say "CHAIN DONE"
