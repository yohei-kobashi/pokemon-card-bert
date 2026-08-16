#!/usr/bin/env bash
set -u
say() { echo "[skd $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
cp /tmp/engine_v2.py agents/
cp /tmp/slowking_D.csv /root/sk/slowking_D.csv
FIELD=marnie_grimmsnarl:26,dudunsparce_box:18,alakazam_nz:12,ogerpon_mono:9,dragapult:6,alakazam:5,crustle_geco:5,crustle:4,mega_lucario_tr:4,cynthia_garchomp:3
run() {  # $1 label, $2 deck csv
  cp "$2" decks/slowking.csv
  rm -f /root/sk/stat_$1.txt
  ENGINE_SK_STAT=/root/sk/stat_$1.txt CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 \
    python3 tools/eval_deck_field.py --deck slowking --field "$FIELD" --games "${G:-60}" \
    --workers 26 --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
    --label "$1" --out /root/sk/$1.json 2>&1 | tail -4
  python3 - "$1" <<'PY'
import ast, sys, collections
tot = collections.Counter()
try:
    for line in open("/root/sk/stat_%s.txt" % sys.argv[1]):
        tot.update(ast.literal_eval(line.strip()))
except FileNotFoundError:
    pass
print("   mechanisms:", dict(tot) or "NONE FIRED")
PY
}
say "arm C2: rebuilt deck (10 energy) + v3 pilot, instrumented"
run c2 /root/sk/slowking_new.csv
say "arm D: rebuilt ammo + OLD energy suite (15) + v3 pilot"
run d /root/sk/slowking_D.csv
python3 - <<'PY'
import json
o = json.load(open("/root/sk/old_deck.json"))
for lbl in ("c2", "d"):
    try: x = json.load(open("/root/sk/%s.json" % lbl))
    except Exception: continue
    print("%-4s FIELD-WEIGHTED %.1f%%   (vs old deck %.1f%%  %+.1fpt)"
          % (lbl, 100*x["weighted"], 100*o["weighted"], 100*(x["weighted"]-o["weighted"])))
PY
say SK_D_DONE
