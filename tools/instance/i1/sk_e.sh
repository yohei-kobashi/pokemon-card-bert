#!/usr/bin/env bash
set -u
say() { echo "[ske $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
cp /tmp/engine_v2.py agents/
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
say "arm E: v4 pilot (Slowking primary + persistent load) on the 15-energy list"
run e /root/sk/slowking_D.csv
say "arm F: v4 pilot on the pure scouted list (10 energy)"
run f /root/sk/slowking_new.csv
say SK_E_DONE
