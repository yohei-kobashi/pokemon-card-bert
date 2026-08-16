#!/usr/bin/env bash
# Second pass of opponent adapters (user 2026-08-11): the three strongest LoRAs from pass 1
# (ogerpon 96.5%, alakazam_nz 89.0%, marnie 84.5%) are already good sparring partners and sit
# out. This pass trains the two that adopted but stayed beatable -- dudunsparce_box 67.5% and
# dragapult 58.5% -- plus slowking round 2, and ADDS the next two decks by live share off the
# 2026-08-11 top-500 scout: crustle_geco (23 teams, 4.5%, #5 overall) and mega_lucario_tr
# (16 teams -- tied with crustle at 3.2%, tie broken on the top-100, where it holds 5 teams to
# crustle's 3). alakazam_nz_fez (14) was passed over: it is the alakazam_nz shell with a tech
# swap, and that LoRA already wins at 89%.
#
# Every gate in this pass runs against the CURRENT sparring dusknoir in the registry
# (mrl2_r5b, plus the R5 deferral wrap if the v3 wrapper gate shipped it), so pass-2 numbers
# are not comparable with pass-1 numbers -- the opponent got harder in between, deliberately.
set -u
say() { echo "[pass2 $(date -u +%m-%d_%H:%M:%S)] $*"; }

# slowking round 1 (launched by after_pass) owns the GPU until it finishes
if pgrep -f "[d]eck_lora2.sh slowking 1" >/dev/null; then
    say "waiting for slowking round 1"
    while pgrep -f "[d]eck_lora2.sh slowking 1" >/dev/null; do sleep 120; done
    say "slowking round 1 finished"
fi

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

run dudunsparce_box 2
run dragapult 2
run crustle_geco 1
run mega_lucario_tr 1
# slowking r2 only exists if r1 adopted; deck_lora2 STOPs cleanly if the adopt file is missing
run slowking 2 100 60
say "PASS2_DONE ($FAIL failed)"
