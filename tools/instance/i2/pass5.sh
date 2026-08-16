#!/usr/bin/env bash
# The tail of pass4, resumed after the batching measurement took the card for a few minutes.
# Same decks, same order: the two opponents that still have no LoRA, then slowking's second
# round (the only deck in pass3 whose gain cleared noise: +14.50, t 3.71).
set -u
say() { echo "[pass5 $(date -u +%m-%d_%H:%M:%S)] $*"; }
FAIL=0
run() {
    local D=$1 N=$2
    say "================ $D round $N ================"
    if GAMES=${3:-150} PER_GAME=${4:-15} bash /root/deck_lora2.sh "$D" "$N"; then
        say "$D r$N done"
    else
        FAIL=$((FAIL+1)); say "$D r$N FAILED -- continuing"
    fi
}
run ethan_hooh 1
run slowking 2 100 60
say "PASS5_DONE ($FAIL failed)"
