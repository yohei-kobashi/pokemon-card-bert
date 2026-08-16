#!/usr/bin/env bash
# Run DPO rounds back to back, stopping the moment one is rejected.
#
# "Fine -> keep going" is the user's instruction (2026-08-10); a rejected round -- the new
# checkpoint measurably worse than the adopted one -- is the definition of "not fine", so the
# loop stops there and leaves everything for a human read. A crashed round stops it too.
set -u
for N in "$@"; do
    echo "================ ROUND $N ================"
    bash /root/round.sh "$N" || { echo "ROUND $N FAILED -- stopping the chain"; exit 1; }
    A=$(cat /root/loop_dpo/adopt_r$N.txt 2>/dev/null || true)
    if [ "$A" != "dpo_r$N" ]; then
        echo "ROUND $N REJECTED (adopted: ${A:-none}) -- stopping the chain for a human read"
        exit 0
    fi
    echo "round $N adopted; continuing"
done
echo "ALL_ROUNDS_DONE"
