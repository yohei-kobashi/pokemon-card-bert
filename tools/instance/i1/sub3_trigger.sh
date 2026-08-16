#!/usr/bin/env bash
# After 12:00Z (21:00 JST), the FIRST completed round triggers the dusk_v3 build.
set -u
say() { echo "[trig $(date -u +%m-%d_%H:%M:%S)] $*" >> /root/sub3_run.log; }
grep -aq SUB3_BUILD_DONE /root/sub3_run.log 2>/dev/null && exit 0
until [ "$(date -u +%s)" -ge "$(date -u -d 2026-08-16T12:00:00Z +%s)" ]; do sleep 60; done
BASE=$(grep -ac winner /root/field_chain.log 2>/dev/null || echo 0)
say "armed at 12:00Z; winner lines so far: $BASE"
until [ "$(grep -ac winner /root/field_chain.log 2>/dev/null || echo 0)" -gt "$BASE" ]; do sleep 60; done
say "round completed: $(grep -a winner /root/field_chain.log | tail -1)"
sleep 200   # let the adoption/ship finish so the registry points at the final champion
bash /root/submit_dusk_v3.sh >> /root/sub3_run.log 2>&1
say "build exited rc=$?"
