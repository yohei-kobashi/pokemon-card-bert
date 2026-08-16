#!/usr/bin/env bash
# Narrow everything to ONE pilot: dragapult_dusknoir, versus the 11 Stage-C decks.
#  (1) carve the dusknoir-piloted rows out of the 38.5M-row pilot-11 pool -- training can start
#      on these immediately, no generation wait;
#  (2) free the disk the generator is gated on (MIN_FREE_GIB) by dropping THIS SESSION's own
#      probe artifacts, nothing of the user's;
#  (3) teach gen_pool_v41.sh to pass --decks so the opponent set is the 11, not all 63;
#  (4) relaunch generation as dusknoir-vs-the-11, pilot-filtered to dusknoir only.
set -u
say() { echo "[dusk $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
PILOT=dragapult_dusknoir
DECKS=$(python3 -c "import sys;sys.path.insert(0,'tools');import rl_config;print(','.join(rl_config.STAGE_C_TARGETS))")
say "pilot=$PILOT | opponents=$DECKS"

say "(1) carving dusknoir rows out of v41_base11"
python3 - <<'PY'
import gzip, json, collections
src = "/root/ptcg/repo/data/rerank/v41_base11.jsonl.gz"
dst = "/root/ptcg/repo/data/rerank/v41_dusk.jsonl.gz"
n = k = 0
opp = collections.Counter()
with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as g:
    for line in f:
        n += 1
        d = json.loads(line)
        if d.get("deck") == "dragapult_dusknoir":
            g.write(line); k += 1; opp[d.get("opp")] += 1
print("[dusk] kept %d of %d rows (%.1f%%)" % (k, n, 100.0*k/max(1, n)))
print("[dusk] opponent mix:")
for o, c in opp.most_common():
    print("   %-24s %6d (%.1f%%)" % (o, c, 100.0*c/max(1, k)))
PY
ls -la "$REPO/data/rerank/v41_dusk.jsonl.gz"

say "(2) freeing disk -- this session's own probe artifacts only"
df -Pk "$REPO" | awk 'NR==2 {printf "    before: %d GiB free\n", int($4/1048576)}'
for d in /root/out/probe_rote /root/out/dpo_probe /root/out/dpo_probe2 /root/out/dpo_probe3; do
  [ -d "$d" ] && { du -sh "$d"; rm -rf "$d"; }
done
df -Pk "$REPO" | awk 'NR==2 {printf "    after:  %d GiB free\n", int($4/1048576)}'

say "(3) patching gen_pool_v41.sh for --decks"
grep -q 'DECKS=${DECKS:-}' tools/gen_pool_v41.sh || {
  sed -i 's|^PAIR_WITH=${PAIR_WITH:-}|PAIR_WITH=${PAIR_WITH:-}\n# Restrict the deck UNIVERSE too. --pair-with alone keeps every matchup that involves the\n# focus deck, which against the full 63-deck library is 62 opponents, not the 11 we train for.\nDECKS=${DECKS:-}|' tools/gen_pool_v41.sh
  sed -i 's|^  \[ -n "$PAIR_WITH" \] \&\& PW="--pair-with $PAIR_WITH"|  [ -n "$PAIR_WITH" ] \&\& PW="--pair-with $PAIR_WITH"\n  [ -n "$DECKS" ] \&\& PW="$PW --decks $DECKS"|' tools/gen_pool_v41.sh
}
grep -n 'DECKS=${DECKS:-}\|--decks \$DECKS' tools/gen_pool_v41.sh

say "(4) restarting generation as dusknoir-vs-the-11"
pkill -f gen_pool_v41; sleep 3; pkill -f gen_selfplay; sleep 3
rm -f /root/ptcg/repo/.genv41.count 2>/dev/null || true
cd "$REPO" && DECKS="$DECKS" PAIR_WITH="$PILOT" BASE=$REPO/data/rerank/v41_dusk.jsonl.gz \
  TARGET_ROWS=40000000 setsid nohup bash tools/gen_pool_v41.sh > /root/gen_dusk.log 2>&1 < /dev/null &
sleep 30
tail -6 /root/gen_dusk.log
say DUSK_SETUP_DONE
