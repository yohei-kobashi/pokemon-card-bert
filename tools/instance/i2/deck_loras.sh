#!/usr/bin/env bash
# Round 1 of the opponent adapters, one deck after another, after the last fleet round is done.
#
# SEQUENTIAL BY NECESSITY: a gate holds two Qwen-4B arms plus dusknoir's DeBERTa (~17 GiB), and
# the collection holds a Qwen and a DeBERTa. Two decks at once would fit on 48 GiB only until a
# gate overlapped a gate.
#
# A failing deck does NOT stop the others -- they are independent adapters, and losing four
# because the first one crashed is the wrong trade. Failures are printed and counted.
set -u
DECKS="$@"
[ -n "$DECKS" ] || { echo "usage: deck_loras.sh <deck> [<deck>...]"; exit 1; }
say() { echo "[deck_loras $(date -u +%m-%d_%H:%M:%S)] $*"; }

# The fleet loop owns the GPU until round 8 finishes; deck_lora's gpu_wait gives up after 20 min,
# so wait for the process itself rather than racing it.
if pgrep -f "round.sh 8" >/dev/null; then
    say "waiting for the last fleet round (round 8) to finish"
    while pgrep -f "round.sh 8" >/dev/null; do sleep 120; done
    say "round 8 finished"
fi

FAIL=0
for D in $DECKS; do
    say "================ $D ================"
    if bash /root/deck_lora.sh "$D" 1; then
        say "$D done"
    else
        FAIL=$((FAIL+1))
        say "$D FAILED -- continuing with the rest"
    fi
done
say "ALL_DECK_LORAS_DONE ($FAIL failed)"
