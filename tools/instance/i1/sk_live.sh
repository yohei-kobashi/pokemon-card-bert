#!/usr/bin/env bash
# The pilot written from the ladder agents' own replays, on both decklists, at 150 games.
# Arm A (shipped deck + shipped pilot) is the incumbent at 32.9%.
set -u
say() { echo "[skl $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
cp /tmp/engine_v2.py /tmp/tuning.json /tmp/slowking.py agents/
FIELD=marnie_grimmsnarl:26,dudunsparce_box:18,alakazam_nz:12,ogerpon_mono:9,dragapult:6,alakazam:5,crustle_geco:5,crustle:4,mega_lucario_tr:4,cynthia_garchomp:3
run() {
  cp "$2" decks/slowking.csv
  CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/eval_deck_field.py \
    --deck slowking --field "$FIELD" --games 150 --workers 24 \
    --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
    --label "$1" --out /root/sk/L_$1.json 2>&1 | tail -3
}
say "LIVE-scouted deck (Smoochum x2 / Conkeldurr / Annihilape / Ciphermaniac x4) + live pilot"
run scouted /root/sk/slowking_new.csv
say "shipped deck + live pilot (isolates the PILOT)"
run shipped /root/sk/slowking_old.csv
say "ammo-swap deck + live pilot"
run ammo /root/sk/slowking_G.csv
python3 -c "
import json, os
base=json.load(open('/root/sk/c_A150.json'))['weighted']
print()
print('%-26s %8s %8s' % ('arm','weighted','vs A150'))
print('%-26s %7.1f%% %8s' % ('A150 shipped+shipped', 100*base, '-'))
for k,lbl in (('scouted','scouted deck + live pilot'),('shipped','shipped deck + live pilot'),('ammo','ammo-swap + live pilot')):
    p='/root/sk/L_%s.json'%k
    if not os.path.exists(p): continue
    w=json.load(open(p))['weighted']
    print('%-26s %7.1f%% %+7.1f' % (lbl, 100*w, 100*(w-base)))
"
say SK_LIVE_DONE
