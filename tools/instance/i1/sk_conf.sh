#!/usr/bin/env bash
# Confirmation at 150 games/opponent (1,500 battles per arm): the 60-game screens read
# +-2-3pt and G won by 2.8, which is one SE. Also splits deck from pilot -- G's counters say
# the Seek loop barely runs, so the gain may be the decklist alone.
set -u
say() { echo "[skc $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
FIELD=marnie_grimmsnarl:26,dudunsparce_box:18,alakazam_nz:12,ogerpon_mono:9,dragapult:6,alakazam:5,crustle_geco:5,crustle:4,mega_lucario_tr:4,cynthia_garchomp:3
setl2() { python3 -c "
import json,sys
t=json.load(open('agents/tuning.json')); t['slowking']['l2']=sys.argv[1]
json.dump(t, open('agents/tuning.json','w'), indent=1, ensure_ascii=False, sort_keys=True)" "$1"; }
dump() { python3 -c "
import ast,sys,collections
tot=collections.Counter()
try:
    for line in open('/root/sk/stat_%s.txt'%sys.argv[1]): tot.update(ast.literal_eval(line.strip()))
except FileNotFoundError: pass
print('   mechanisms:', dict(tot) or 'NONE')" "$1"; }
run() {
  cp "$2" decks/slowking.csv
  setl2 "$3"
  rm -f /root/sk/stat_$1.txt
  ENGINE_SK_STAT=/root/sk/stat_$1.txt CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 \
    python3 tools/eval_deck_field.py --deck slowking --field "$FIELD" --games 150 \
    --workers 26 --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
    --label "$1" --out /root/sk/c_$1.json 2>&1 | tail -3
  dump "$1"
}
say "A150: current shipped deck + shipped pilot"
run A150 /root/sk/slowking_old.csv slowking_hybrid
say "G150: ammo-swapped deck + shipped pilot  (isolates the DECK)"
run G150 /root/sk/slowking_G.csv slowking_hybrid
say "V150: ammo-swapped deck + v4 pilot       (isolates the PILOT)"
run V150 /root/sk/slowking_G.csv slowking_v3
python3 -c "
import json
r={k: json.load(open('/root/sk/c_%s.json'%k)) for k in ('A150','G150','V150')}
print()
print('%-22s %9s %9s %9s' % ('opponent','A shipped','G deck','V deck+pilot'))
for k in sorted(r['A150']['decks']):
    f=lambda x: 100*r[x]['decks'][k]['p'] if k in r[x]['decks'] else float('nan')
    print('%-22s %8.1f%% %8.1f%% %8.1f%%' % (k, f('A150'), f('G150'), f('V150')))
a,g,v=(100*r[x]['weighted'] for x in ('A150','G150','V150'))
print()
print('FIELD-WEIGHTED   A %.1f%%   G %.1f%% (%+.1f)   V %.1f%% (%+.1f)' % (a,g,g-a,v,v-a))"
say SK_CONF_DONE
