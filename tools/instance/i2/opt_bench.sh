#!/bin/bash
# One benchmark session over every option still alive after today's measurements.
#
# DEAD, measured, not retried here:
#   HF padded batching        1.05x at batch 4, WORSE at 32 (left-pad wastes real compute)
#   score memoisation         0.1% repeat rate over 49,532 real decisions
#   cross-decision prefix KV  DECK[] renders the REMAINING deck: 152 distinct prefixes in 154
#                             decisions of one deck. This also caps what vLLM's prefix caching
#                             can do here -- it helps within a decision, never across.
#
# ALIVE:
#   fp8       Ada sm89 has FP8 tensor cores at ~2x bf16 and the prefill is compute-bound at
#             ~52 of ~91 TFLOPS. Also halves the weights, which is what limits the screen to 3
#             shards on a 47.4 GiB card while 3 single-threaded game loops use ~3 of 13.44 cores.
#   compile   65.8 ms measured against ~40 ms of prefill compute = ~25 ms of Python and launch
#             overhead per decision.
#   vllm      reference number. Adopted only if it clearly beats the tuned in-process stack,
#             because using it means restructuring the game loop to keep many games in flight.
#
# Every speed number is paired with an ARGMAX AGREEMENT number. A faster scorer that picks
# differently is not a faster scorer, it is a different pilot.
set -u
REPO=/root/ptcg/repo
CKPT=/root/out/qwen3_4b_cfb_v40
DATA=$REPO/data/sft/cf_b_v40.jsonl.gz
EXPORT=/root/export/cfb_v40_merged
LOG=/root/opt_bench.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo; echo "######## $* ########"; }

say "1/4 speed"
for v in "" "--fp8" "--compile-mode default" "--fp8 --compile-mode default"; do
  python3 tools/bench_scorer_final.py --ckpt "$CKPT" --data "$DATA" --n 120 --merge $v 2>&1 \
    | grep -E "^RESULT|FP8|compile|logits_to_keep=" || true
done

say "2/4 argmax agreement vs the path the baselines were made with"
for var in "hf,1,1,1," "hf,1,1,0,default" "hf,1,1,1,default"; do
  python3 tools/check_scorer_equiv.py --ckpt "$CKPT" --data "$DATA" --n 300 --variant "$var" 2>&1 \
    | grep -E "variant:|argmax agreement|flips:|gap > 0.05" || true
done

say "3/4 VRAM per process (decides how many shards fit in 47.4 GiB)"
for v in "" "--fp8"; do
  python3 - "$CKPT" $v <<'PY' || true
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
from tools.mirror_match import QwenScorer
fp8 = "--fp8" in sys.argv
sc = QwenScorer(sys.argv[1], backend="hf", merge=True, kv=True, fp8=fp8)
sc._score_card_first("[ACT]\nDECK win[c1] T1.1 ME A[-] pz0 dk60 bm5 H[] | OP A[-] pz0 dk60 bm5 h0 "
                     "|| SEL MAIN n1-1 :: 0=end 1=attach:c7@ACTIVE", ["end", "attach:c7@ACTIVE"])
print("VRAM fp8=%s: %.2f GiB allocated, %.2f GiB reserved -> %d shards fit in 47.4"
      % (fp8, torch.cuda.max_memory_allocated()/2**30, torch.cuda.max_memory_reserved()/2**30,
         int(47.4 / max(0.1, torch.cuda.max_memory_reserved()/2**30))), flush=True)
PY
done

say "5/5 training throughput sweep -- the hardware decision depends on this"
bash /root/train_sweep.sh || true

say "4/5 vLLM reference"
if /root/vllmenv/bin/python -c "import torch,vllm; assert torch.cuda.is_available()" 2>/dev/null; then
  python3 tools/check_export_equiv.py --ckpt "$CKPT" --export "$EXPORT" --data "$DATA" --n 200 \
    && /root/vllmenv/bin/python tools/bench_vllm.py --model "$EXPORT" --data "$DATA" \
         --n 500 --topk 64 2>&1 | grep -E "^\[|decisions/s|covered|top " \
    || echo "vLLM skipped: the export does not match the in-memory model"
else
  echo "vLLM skipped: the venv has no working CUDA (driver 570.124.04 = CUDA 12.8)"
fi
say "BENCH DONE"
