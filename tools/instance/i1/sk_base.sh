#!/usr/bin/env bash
# Baseline the slowking rebuild against the ladder it will actually meet. Arm A is the list
# HEAD shipped (Metagross/Zeraora ammo, no Ciphermaniac's); arm B is the top-20 list. Same
# pilot in both, so this isolates the DECK.
set -u
say() { echo "[skb $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
FIELD=marnie_grimmsnarl:26,dudunsparce_box:18,alakazam_nz:12,ogerpon_mono:9,dragapult:6,alakazam:5,crustle_geco:5,crustle:4,mega_lucario_tr:4,cynthia_garchomp:3
G=${G:-60}
mkdir -p /root/sk
cp decks/slowking.csv /root/sk/slowking_new.csv
git show HEAD:decks/slowking.csv > /root/sk/slowking_old.csv

say "arm A: HEAD decklist (old ammo)"
cp /root/sk/slowking_old.csv decks/slowking.csv
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/eval_deck_field.py \
  --deck slowking --field "$FIELD" --games "$G" --workers 26 \
  --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
  --label old_deck --out /root/sk/old_deck.json 2>&1 | tail -16

say "arm B: rebuilt decklist (Conkeldurr/Annihilape/Smoochum + Ciphermaniac's)"
cp /root/sk/slowking_new.csv decks/slowking.csv
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/eval_deck_field.py \
  --deck slowking --field "$FIELD" --games "$G" --workers 26 \
  --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
  --label new_deck --out /root/sk/new_deck.json 2>&1 | tail -16

python3 - <<'PY'
import json
a = json.load(open("/root/sk/old_deck.json")); b = json.load(open("/root/sk/new_deck.json"))
print("\n%-24s %8s %8s %8s" % ("opponent", "old", "new", "delta"))
for k in sorted(set(a["decks"]) | set(b["decks"])):
    x = a["decks"].get(k, {}).get("p"); y = b["decks"].get(k, {}).get("p")
    if x is None or y is None: continue
    print("%-24s %7.1f%% %7.1f%% %+7.1f" % (k, 100*x, 100*y, 100*(y-x)))
print("\nFIELD-WEIGHTED  %.1f%% -> %.1f%%  (%+.1fpt)"
      % (100*a["weighted"], 100*b["weighted"], 100*(b["weighted"]-a["weighted"])))
PY
say SK_BASE_DONE
