#!/usr/bin/env bash
# Re-run of pass 2 (2026-08-12). Pass 2 lost all five decks in 13 minutes to ONE cause: the
# registry entry for dragapult_dusknoir had been rewritten to
#   planfilter:lethal_now,spread_aim,clops_hold,energy_line,energy_focus:hf:/root/out/mrl2_r5b
# by wrap_ship, but THIS machine still carried the 10-rule tools/dusk_plan.py. lethal_now and
# clops_hold did not exist here, so every collection shard died on "unknown plan rule" before
# playing a game, left 0-byte traces, and the brancher was handed nothing. dusk_plan.py is now
# the 16-rule version (md5 b26149cc...) and a 4-game smoke ran clean.
#
# slowking runs FIRST and at round 1: its pass-2 "round 2" could never have worked -- round 1
# never collected either. It was killed mid-collection during an unrelated rollback, and the old
# resume guard tested trace EXISTENCE, so the retry skipped straight to a branch request over
# three empty files and burned 2h waiting. deck_lora2.sh now resumes on size.
#
# One round per deck, five decks. slowking r2 is deliberately NOT here: a second round for one
# deck is worth less than a first round for the four that have none.
set -u
say() { echo "[pass3 $(date -u +%m-%d_%H:%M:%S)] $*"; }

FAIL=0
run() {
    local D=$1 N=$2
    say "================ $D round $N ================"
    if GAMES=${3:-150} PER_GAME=${4:-15} bash /root/deck_lora2.sh "$D" "$N"; then
        say "$D r$N done"
    else
        FAIL=$((FAIL+1))
        say "$D r$N FAILED -- continuing"
    fi
}

run slowking 1 100 60
run dudunsparce_box 2
run dragapult 2
run crustle_geco 1
run mega_lucario_tr 1
say "PASS3_DONE ($FAIL failed)"
