#!/usr/bin/env bash
set -u
say() { echo "[skgh $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
cp /tmp/slowking_G.csv /tmp/slowking_H.csv /root/sk/
FIELD=marnie_grimmsnarl:26,dudunsparce_box:18,alakazam_nz:12,ogerpon_mono:9,dragapult:6,alakazam:5,crustle_geco:5,crustle:4,mega_lucario_tr:4,cynthia_garchomp:3
run() {
  cp "$2" decks/slowking.csv
  rm -f /root/sk/stat_$1.txt
  ENGINE_SK_STAT=/root/sk/stat_$1.txt CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 \
    python3 tools/eval_deck_field.py --deck slowking --field "$FIELD" --games "${G:-60}" \
    --workers 26 --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
    --label "$1" --out /root/sk/$1.json 2>&1 | tail -3
  python3 - "$1" <<'PY'
import ast, sys, collections
tot = collections.Counter()
try:
    for line in open("/root/sk/stat_%s.txt" % sys.argv[1]): tot.update(ast.literal_eval(line.strip()))
except FileNotFoundError: pass
print("   mechanisms:", dict(tot) or "NONE")
PY
}
say "arm G: proven shell, ammo swapped only (15 energy, no Ciphermaniac)"
run g /root/sk/slowking_G.csv
say "arm H: G + Ciphermaniac x2 (13 energy)"
run h /root/sk/slowking_H.csv
python3 - <<'PY'
import json, os
lab = {"old_deck":"A old shell+old ammo","new_deck":"B new list, old pilot",
       "new_v3":"C new list, v3","e":"E new ammo 15E, v4","f":"F scouted list, v4",
       "g":"G shell+new ammo","h":"H G+Cipher"}
for k, name in lab.items():
    p = "/root/sk/%s.json" % k
    if os.path.exists(p):
        x = json.load(open(p))
        print("%-26s %.1f%%" % (name, 100*x["weighted"]))
PY
say SK_GH_DONE
