#!/bin/bash
# End-to-end smoke test of the seeded round, on a 3-deck test set.
#
# Runs every stage a real round runs EXCEPT training and mixing (both need the GPU box's corpora):
# mirror screen -> shard merge -> summary + paired line -> fingerprint guard -> tier/targets ->
# collect shards -> seeded collection with the anchor panel. The loop's embedded python is
# EXTRACTED FROM THE .sh AND RUN, not reimplemented, so this tests the shipped text.
#
#   bash tools/smoke_seeded.sh [loop-script]      default tools/dagger_loop_i2d.sh
set -u
LOOP=${1:-tools/dagger_loop_i2d.sh}
W=${W:-/tmp/claude-1000/-home-kobashi-ptcgabc/e170e336-467a-4c34-a3f3-022796a19879/scratchpad/smoke}
DECKS="crustle_stall dragapult alakazam"
PANEL="crustle_stall,dragapult"
PASS=0; FAIL=0
rm -rf "$W"; mkdir -p "$W"

ok()   { PASS=$((PASS+1)); echo "  PASS  $*"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL  $*"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# Pull an embedded heredoc out of the loop by a MARKER, not a line number, so the same test runs
# against either loop script and does not rot when either is edited.
extract() {
  local s
  s=$(grep -n -- "$1" "$LOOP" | head -1 | cut -d: -f1)
  [ -n "$s" ] || { echo "marker not found: $1" >&2; return 1; }
  sed -n "$((s+1)),\$p" "$LOOP" | sed "/^$2\$/q" | sed '$d'
}

echo "=== 0. static checks ==="
check "loop script parses"        "bash -n $LOOP"
check "collect_dagger parses"     "python3 -c 'import ast;ast.parse(open(\"tools/collect_dagger.py\").read())'"
check "mirror_match parses"       "python3 -c 'import ast;ast.parse(open(\"tools/mirror_match.py\").read())'"
check "engine .so present"        "[ -f data/kaggle_engine_ext/libcg_mirror.so ]"

echo "=== 1. mirror screen, 2 shards (the round's --a engine --b <pilot>) ==="
i=0
for SH in "crustle_stall dragapult" "alakazam"; do
  ARGS=""; for d in $SH; do ARGS="$ARGS --deck $d"; done
  PYTHONPATH=cg-lib python3 tools/mirror_match.py $ARGS --a engine --b noisy:0.15 \
     --mirror --seed 1 --mirror-so data/kaggle_engine_ext/libcg_mirror.so \
     --max-games 20 --alpha 1e-12 --beta 1e-12 --out "$W/mirror_r9.$i.json" > "$W/screen.$i.log" 2>&1
  i=$((i+1))
done
check "shard files written"        "[ -s $W/mirror_r9.0.json ] && [ -s $W/mirror_r9.1.json ]"
check "fingerprint recorded"       "grep -q 'shuffle_fp' $W/mirror_r9.0.json"
check "seeded (seed_base present)" "grep -q 'seed_base' $W/mirror_r9.0.json"

echo "=== 2. shard merge (extracted from the loop) ==="
extract 'python3 - "$j" "$SOUT"' PYX > "$W/merge.py"
python3 "$W/merge.py" 2 "$W/mirror_r9.json" "$W/mirror_r9" > "$W/merge.log" 2>&1
check "merged 3 decks"             "grep -q 'merged -> 3 decks' $W/merge.log"

echo "=== 3. previous round (a different pilot on the SAME seeds) ==="
i=0
for SH in "crustle_stall dragapult" "alakazam"; do
  ARGS=""; for d in $SH; do ARGS="$ARGS --deck $d"; done
  PYTHONPATH=cg-lib python3 tools/mirror_match.py $ARGS --a engine --b noisy:0.25 \
     --mirror --seed 1 --mirror-so data/kaggle_engine_ext/libcg_mirror.so \
     --max-games 20 --alpha 1e-12 --beta 1e-12 --out "$W/mirror_r8.$i.json" > /dev/null 2>&1
  i=$((i+1))
done
python3 "$W/merge.py" 2 "$W/mirror_r8.json" "$W/mirror_r8" > /dev/null 2>&1
check "previous-round screen built" "[ -s $W/mirror_r8.json ]"

echo "=== 4. summary + paired line (extracted from the loop) ==="
extract 'history.tsv" "$ROUND"' PY > "$W/summary.py"
python3 "$W/summary.py" "$W/mirror_r9.json" "$W/history.tsv" 9 "$W/mirror_r8.json" > "$W/summary.log" 2>&1
cat "$W/summary.log" | sed 's/^/      /'
check "screen summary printed"     "grep -q '\[screen\] round 9' $W/summary.log"
check "paired line printed"        "grep -q 'paired vs previous round' $W/summary.log"
check "history.tsv appended"       "[ -s $W/history.tsv ]"

echo "=== 5. fingerprint guard (tamper the previous round's engine identity) ==="
python3 - "$W/mirror_r8.json" "$W/mirror_r8_bad.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for v in d["decks"].values():
    v["shuffle_fp"] = "deadbeefdeadbeef"
json.dump(d, open(sys.argv[2], "w"))
EOF
python3 "$W/summary.py" "$W/mirror_r9.json" "$W/history.tsv" 9 "$W/mirror_r8_bad.json" > "$W/guard.log" 2>&1
check "guard refuses a changed engine" "grep -q 'REFUSING' $W/guard.log"
check "guard suppresses the paired line" "! grep -q 'paired vs previous round' $W/guard.log"

echo "=== 6. tier / target selection (extracted from the loop) ==="
# only the TARGETS=$(...) assignment -- taking more swept in `say` and an unterminated `if`,
# which is what made this "fail" while the loop's own logic was fine.
S=$(grep -n 'TARGETS=$(python3 -c' "$LOOP" | head -1 | cut -d: -f1)
E=$(awk -v s="$S" 'NR>s && $0 == "\")" {print NR; exit}' "$LOOP")
# Prepend the loop's own NAME=${NAME:-default} knobs rather than guessing which ones the tier
# block reads -- loop7 also uses MIN_TARGETS, and hardcoding one name here just moves the problem.
{ grep -E '^[A-Z_]+=\$\{[A-Z_]+:-' "$LOOP"; sed -n "${S},${E}p" "$LOOP"; echo 'echo "$TARGETS"'; } \
  | sed "s|\$MIR|$W/mirror_r9.json|" > "$W/tier.sh"
TARGETS=$(bash "$W/tier.sh" 2>"$W/tier.log" | tail -1)
echo "      targets: ${TARGETS:-<none>}"
check "tier produced targets"      "[ -n \"$TARGETS\" ]"

echo "=== 7. seeded collection with the fixed anchor panel ==="
PYTHONPATH=cg-lib python3 tools/collect_dagger.py --decks "crustle_stall,alakazam" \
   --model noisy:0.15 --games 4 --seed 901 --engine-seed-base 65700000 \
   --mirror-so data/kaggle_engine_ext/libcg_mirror.so \
   --anchor-decks "$PANEL" --anchor-games 4 --out "$W/dagger.jsonl.gz" > "$W/collect.log" 2>&1
grep -E '^\[seeded\]|^\[anchor\]|^\[fresh\]|^\[anchor-deck\]|^written' "$W/collect.log" | sed 's/^/      /'
check "collection wrote rows"      "[ -s $W/dagger.jsonl.gz ]"
check "anchor measured"            "grep -q '^\[anchor\] ' $W/collect.log"
check "anchor held out of the data" "grep -q 'held out' $W/collect.log"
check "per-deck anchor rates"      "[ \$(grep -c '^\[anchor-deck\]' $W/collect.log) -eq 2 ]"
check "no anchor rows in the file" "[ \$(python3 -c \"
import gzip,json
print(sum(1 for l in gzip.open('$W/dagger.jsonl.gz','rt') if json.loads(l)['anchor']))\") -eq 0 ]"
check "panel deck NOT in targets is still measured" "grep -q '^\[anchor-deck\] dragapult' $W/collect.log"

echo "=== 8. anchors are invariant to the target list ==="
PYTHONPATH=cg-lib python3 tools/collect_dagger.py --decks "zangoose" \
   --model noisy:0.15 --games 4 --seed 902 --engine-seed-base 88800000 \
   --mirror-so data/kaggle_engine_ext/libcg_mirror.so \
   --anchor-decks "$PANEL" --anchor-games 4 --out "$W/dagger2.jsonl.gz" > "$W/collect2.log" 2>&1
A1=$(grep '^\[anchor\] ' "$W/collect.log"  | sed 's/.*wrong \([0-9.]*\)%.*/\1/')
A2=$(grep '^\[anchor\] ' "$W/collect2.log" | sed 's/.*wrong \([0-9.]*\)%.*/\1/')
echo "      targets={crustle,alakazam} -> $A1%   targets={zangoose} -> $A2%"
check "anchor rate identical across target sets" "[ \"$A1\" = \"$A2\" ]"

echo
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
