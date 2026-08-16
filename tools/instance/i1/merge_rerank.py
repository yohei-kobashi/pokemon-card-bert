"""Linearly interpolate two reranker checkpoints:  theta(a) = (1-a)*v35 + a*v36.

Safe here for a specific reason: v36 was CONTINUED from v35, so the two sit on one training
trajectory in one basin -- measured ||v36-v35||/||v35|| = 0.0032. Interpolating unrelated runs
would not be safe; interpolating along a trajectory is closer to choosing where to stop.

Why bother when a single metric usually moves monotonically between endpoints (so the best a
would just be an endpoint): the two models differ in OPPOSITE directions per matchup -- v36 is
-11.7pt vs alakazam_nz but +7.7 vs archaludon and +3.7 vs dragapult. The submission is judged
on a live-frequency-weighted COMPOSITE of seven matchups, and a composite of non-monotone
components can have an interior optimum.

Score-ensembling the two models instead is not an option: two INT8 copies are 208 MiB
compressed against a 197.66 MiB cap, and double the per-decision latency. A merge costs
nothing at deploy.

The tokenizer/config are copied from v35; they are identical between the two (same 53,339
vocab, same 22 layers), so either source works.
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

SRC_A = sys.argv[1]                       # a=0 end
SRC_B = sys.argv[2]                       # a=1 end
OUT_ROOT = sys.argv[3]
ALPHAS = [float(x) for x in sys.argv[4].split(",")]

wa = load_file(os.path.join(SRC_A, "model.safetensors"))
wb = load_file(os.path.join(SRC_B, "model.safetensors"))
assert set(wa) == set(wb), "checkpoints do not share tensor names"

for a in ALPHAS:
    out = "%s_a%02d" % (OUT_ROOT, round(a * 100))
    os.makedirs(out, exist_ok=True)
    merged = {}
    for k in wa:
        x, y = wa[k], wb[k]
        if x.shape != y.shape:
            raise SystemExit("shape mismatch on %s" % k)
        if x.is_floating_point():
            merged[k] = ((1.0 - a) * x.float() + a * y.float()).to(x.dtype)
        else:
            merged[k] = x.clone()          # ints/buffers: identical, no meaningful average
    save_file(merged, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(SRC_A, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, fn))
    d = sum((merged[k].float() - wa[k].float()).pow(2).sum().item()
            for k in list(wa)[:400]) ** 0.5
    n = sum(wa[k].float().pow(2).sum().item() for k in list(wa)[:400]) ** 0.5
    print("a=%.2f -> %s   ||merged-v35||/||v35|| = %.4f" % (a, out, d / n), flush=True)
