#!/usr/bin/env bash
# Interpolate dusk_s1 with plan_dusk and find the largest step toward the plan that keeps the
# win rate.
#
# plan_dusk was trained FROM dusk_s1 with an L2-SP anchor at dusk_s1, so theta_plan = theta_s1 +
# delta and the merge theta_s1 + alpha*delta is exactly "apply a fraction of the plan update".
# That makes this sweep the CHEAP version of the experiment the collapse suggested -- retrain
# with fewer steps / a stronger anchor -- with no retraining at all.
#
# The endpoints, measured on 1650 paired games and 1902 held-out plan rows:
#   alpha 0 (s1)    win 18.9%   plan conformance 61.5%
#   alpha 1 (plan)  win  5.2%   plan conformance 93.6%
# ||plan - s1|| / ||s1|| is 0.0031, so a THIRD of one percent of weight movement cost 13.8pt of
# win rate. Sample the small end densely; the interesting region is unlikely to be near 1.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
S1=/root/out/dusk_s1
PLAN=/root/out/plan_dusk
DATA=/root/rl/plan_dusk.jsonl.gz
OUT=/root/out/merge
mkdir -p "$OUT"
say() { echo "[merge $(date -u +%m-%d_%H:%M:%S)] $*"; }

ALPHAS="${ALPHAS:-0.05 0.10 0.20 0.35 0.50 0.75}"

for A in $ALPHAS; do
    D=$OUT/a$A
    if [ ! -f "$D/model.safetensors" ]; then
        python3 - "$S1" "$PLAN" "$D" "$A" <<'PY'
import os, shutil, sys, torch
from safetensors.torch import load_file, save_file
s1, plan, out, a = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
os.makedirs(out, exist_ok=True)
# Everything except the weights comes from s1: same tokenizer, same config, same label head
# shape. Copying rather than symlinking so the directory is self-contained for a scorer that
# may be loaded on another machine.
for fn in os.listdir(s1):
    if fn != "model.safetensors" and os.path.isfile(os.path.join(s1, fn)):
        shutil.copy(os.path.join(s1, fn), os.path.join(out, fn))
A_, B_ = load_file(os.path.join(s1, "model.safetensors")), load_file(os.path.join(plan, "model.safetensors"))
merged = {k: (A_[k].float() * (1.0 - a) + B_[k].float() * a).to(A_[k].dtype) for k in A_}
save_file(merged, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
print("[merge] alpha %.2f -> %s" % (a, out))
PY
    fi
    # Conformance on the SAME held-out 5% every time: dusk_plan_train shuffles with Random(0)
    # and splits before touching the model, so the split does not move with alpha. --epochs 0
    # runs no steps; the "before" number IS the merged model's conformance.
    python3 tools/dusk_plan_train.py --data "$DATA" --model "$D" --out /root/out/discard_m \
        --epochs 0 --l2sp 0 > "$OUT/conf_a$A.log" 2>&1
    C=$(grep -a "conformance before" "$OUT/conf_a$A.log" | tail -1 | grep -oE "[0-9.]+%" | tail -1)
    say "alpha $A  plan-conformance ${C:-?}"
    rm -rf /root/out/discard_m
done

say "=== conformance vs alpha ==="
echo "  alpha 0.00 (s1)    61.5%   [win 18.9%]"
for A in $ALPHAS; do
    C=$(grep -a "conformance before" "$OUT/conf_a$A.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+%" | tail -1)
    echo "  alpha $A          ${C:-?}"
done
echo "  alpha 1.00 (plan)  93.6%   [win  5.2%]"
say MERGE_SWEEP_DONE
