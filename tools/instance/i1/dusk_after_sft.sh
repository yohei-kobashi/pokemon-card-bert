#!/bin/bash
# Unattended: wait for the single-deck SFT, gate it cross-deck, then branch.
#
#   no degradation -> ask which RULES the model cannot be taught (one rule at a time, 10 epochs)
#   degradation    -> ask WHY (play-time prompt / format effect / training effect)
#
# The branch is decided by s1 - r8 as a PAIRED difference over identical seeds and seats, not by
# either arm's win rate: `rl-gate-is-noisier-than-assumed` re-scored one checkpoint 2.6pt apart,
# so an unpaired comparison at this sample size decides nothing.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
LOG=/root/dusk_after_sft.log
say() { echo "[after $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for the SFT to finish"
while ! grep -qaE "^\[dusk-sft .*\] (done ->|train FAILED|no model saved|STOP)" /root/dusk_sft1.log 2>/dev/null; do
  sleep 60
done
TAIL=$(grep -aE "^\[dusk-sft .*\] (done ->|train FAILED|no model saved|STOP)" /root/dusk_sft1.log | tail -1)
say "SFT ended: $TAIL"
grep -aE "\[eval\]|\[ablation\]|step " /root/dusk_sft1.log | tail -6

if ! echo "$TAIL" | grep -q "done ->"; then
  # A failed train still leaves two of the three diagnostic questions answerable, and the night
  # is otherwise spent. Run what does not need dusk_s1 rather than exiting into nothing.
  say "the SFT did not finish -- running the diagnostic on what exists"
  bash /root/dusk_diag.sh
  exit 1
fi

say "GATE: dragapult_dusknoir vs the eleven, arms engine / r8 / s1"
bash /root/dusk_gate.sh 2>&1 | tail -40

V=/root/loop_dusk/gate1/verdict.json
if [ ! -s "$V" ]; then
  say "no verdict written -- treating that as a failure to measure, not as a pass"
  bash /root/dusk_diag.sh
  exit 1
fi
cat "$V"
DEG=$(python3 -c "import json;print(json.load(open('$V'))['degraded'])")
say "verdict degraded=$DEG"

if [ "$DEG" = "False" ]; then
  say "BRANCH A -- no degradation: probing the rules one at a time"
  bash /root/dusk_rule_probe.sh
else
  say "BRANCH B -- degraded: diagnosing"
  bash /root/dusk_diag.sh
fi
say "overnight run complete"
