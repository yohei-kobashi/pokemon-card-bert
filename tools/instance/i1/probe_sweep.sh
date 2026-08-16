#!/usr/bin/env bash
# The probe stalled at 1.335 against a target-entropy floor of 0.593 -- it is not memorising
# 300 rows, so before spending a real round find out whether that is the step size or the
# number of updates. accum=8 at 300 rows x 8 epochs is only 300 optimizer steps, which is the
# update-starvation shape instance1 already diagnosed once (accum 12 -> ~4.2k steps/round).
set -u
cd /root/ptcg/repo
for cfg in "3e-5 8 8" "1e-4 8 1" "3e-5 20 1"; do
  set -- $cfg
  echo "=== lr=$1 epochs=$2 accum=$3 (floor 0.593) ==="
  PYTHONPATH=cg-lib python3 - "$1" "$2" "$3" <<'PY'
import subprocess, sys
lr, ep, ac = sys.argv[1:4]
r = subprocess.run(["python3","tools/dusk_plan_train.py","--data","/root/rl/plan_r1.jsonl.gz",
   "--model","/root/out/d41_r8","--out","/root/out/plan_probe","--probe",
   "--lr",lr,"--epochs",ep,"--accum",ac], capture_output=True, text=True, env={**__import__("os").environ,"PYTHONPATH":"cg-lib"})
for ln in r.stdout.splitlines():
    if "FINAL" in ln or "PROBE" in ln or "step 2" in ln: print("   ", ln)
PY
done
echo SWEEP_DONE
