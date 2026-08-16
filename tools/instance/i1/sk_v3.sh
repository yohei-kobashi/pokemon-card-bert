#!/usr/bin/env bash
# Arm C: rebuilt deck + the pilot that can actually use it (slowking_v3).
set -u
say() { echo "[skv3 $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
cp /tmp/engine_v2.py /tmp/tuning.json /tmp/slowking.py agents/
cp /root/sk/slowking_new.csv decks/slowking.csv
FIELD=marnie_grimmsnarl:26,dudunsparce_box:18,alakazam_nz:12,ogerpon_mono:9,dragapult:6,alakazam:5,crustle_geco:5,crustle:4,mega_lucario_tr:4,cynthia_garchomp:3
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/eval_deck_field.py \
  --deck slowking --field "$FIELD" --games "${G:-60}" --workers 26 \
  --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
  --label new_deck_v3 --out /root/sk/new_v3.json 2>&1 | tail -16
python3 - <<'PY'
import json
o = json.load(open("/root/sk/old_deck.json")); b = json.load(open("/root/sk/new_deck.json"))
c = json.load(open("/root/sk/new_v3.json"))
print("\n%-24s %9s %9s %9s" % ("opponent", "A old", "B newdeck", "C new+v3"))
for k in sorted(o["decks"]):
    f = lambda d: 100*d["decks"][k]["p"] if k in d["decks"] else float("nan")
    print("%-24s %8.1f%% %8.1f%% %8.1f%%" % (k, f(o), f(b), f(c)))
print("\nFIELD-WEIGHTED  A %.1f%%   B %.1f%%   C %.1f%%   (C-A %+.1fpt)"
      % (100*o["weighted"], 100*b["weighted"], 100*c["weighted"],
         100*(c["weighted"]-o["weighted"])))
PY
say SK_V3_DONE
