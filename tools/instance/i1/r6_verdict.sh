#!/usr/bin/env bash
# Round 6's gate outlived its parent: bundle_gate.sh pkill'd "bash mirror_chain2.sh" and the
# gate_protagonist CHILD kept running, so the verdict would have been written to a JSON nobody
# reads. This applies the chain's own rule to that file and records the champion, so a round
# that cost 1800 games is not thrown away by how it was interrupted.
set -u
J=/root/loop_dusk/mrl2/gate_r6.json
say() { echo "[r6v $(date -u +%m-%d_%H:%M:%S)] $*"; }
say "waiting for $J"
while [ ! -s "$J" ]; do
    pgrep -f "seed 61600" >/dev/null || { say "round 6 gate died before writing"; exit 1; }
    sleep 60
done
sleep 5
python3 - "$J" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
a = j.get("arms", {})
best, bd = None, None
for k in ("a", "b"):
    d = (a.get(k) or {}).get("delta_vs_baseline")
    if d is None:
        continue
    print("  %-3s %5.1f%%  %+6.2f +- %.2f" % (k, a[k]["win_rate"], d, a[k]["se"]))
    if bd is None or d > bd:
        best, bd = k, d
if bd is not None and bd > 1.0:
    p = "/root/out/mrl2_r6%s" % best
    open("/root/loop_dusk/mrl2/current.txt", "w").write(p)
    print("round 6 ADOPT %s" % p)
else:
    print("round 6: no challenger cleared +1.0pt -- champion unchanged")
PY
say "R6_VERDICT_DONE"
