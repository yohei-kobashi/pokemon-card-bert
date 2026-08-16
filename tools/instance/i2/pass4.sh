#!/usr/bin/env bash
# Fill the three holes in the 4B opponent set.
#
# instance1 now gates dusknoir against EIGHT decks, and the plan is to hand that gate over to
# these LoRAs once the engine_v2 phase converges (field_chain OPP_SPEC=engine -> reg). Five of
# the eight already have an adapter; archaludon, mega_abomasnow_sample and ethan_hooh do not, so
# a handover today would quietly leave three of eight cells on engine_v2 while the log said
# "reg". Train those three first -- they are worth more than a fourth round on a deck that
# already has one.
#
# The sparring dusknoir now carries the CURRENT wrapper from instance1 (prohibitions +
# lethal_now), so these adapters are trained against the pilot they will actually face.
#
# pass 3 verdicts, for context on what another round is worth: slowking r1 +14.50 (t 3.71) was
# the only real movement; dudunsparce_box r2 -1.50, dragapult r2 -0.50 and crustle_geco r1 +1.00
# were all inside noise and adopted only by the "not a collapse" rule. So slowking gets a second
# round here and the others do not.
set -u
say() { echo "[pass4 $(date -u +%m-%d_%H:%M:%S)] $*"; }
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
run archaludon 1
run mega_abomasnow_sample 1
run ethan_hooh 1
run slowking 2 100 60
say "PASS4_DONE ($FAIL failed)"
