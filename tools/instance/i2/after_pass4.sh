#!/usr/bin/env bash
# instance2: when pass4 finishes and the card is IDLE, measure properly, then serve.
#
# Why the idle wait is not optional. The first batch sweep was run while archaludon's gate held
# the GPU at 100%, which makes every absolute number meaningless: the measured "peak" bf16
# throughput came out at 53.8 TFLOP/s, roughly half what this card should do, and a saturated
# card cannot show whether a bigger batch would have found idle capacity. Only the RATIO of
# model throughput to pure-GEMM throughput survived that, because both were scaled alike.
#
# Order: verify correctness -> measure the ceiling -> measure the decision split -> serve.
set -u
cd /root/ptcg/repo
LOG=/root/after_pass4.log
ADAPTER=${ADAPTER:-/root/out/lora_marnie_grimmsnarl_r1}
say() { echo "[after $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for PASS4_DONE"
while ! grep -aq PASS4_DONE /root/pass4.log 2>/dev/null; do
    if ! pgrep -f "/root/pass4.sh" > /dev/null; then
        say "pass4.sh is gone without PASS4_DONE -- it died. Continuing; the adapters that DID"
        say "finish are still worth serving, and the missing ones show up in adapters.py check."
        break
    fi
    sleep 120
done
say "pass4 over: $(tail -1 /root/pass4.log)"

# Idle means BOTH: memory released and the SMs quiet. Memory alone is not enough -- a process
# can be between allocations while still driving the card.
say "waiting for an idle card"
for _ in $(seq 90); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 3000 ] && [ "$G" -lt 15 ] && break
    sleep 20
done
say "GPU ${U} MiB / ${G}% busy"

say "=== 1. every adapter must reproduce its standalone checkpoint ==="
PYTHONPATH=cg-lib:tools:. python3 -u tools/score_server.py \
    --from-registry --verify all --bench 40 --no-serve >> "$LOG" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    say "VERIFY FAILED (rc $RC) -- NOT serving. instance1 must stay on engine_v2."
    exit 1
fi

say "=== 2. batch sweep on an IDLE card (the contended one proved nothing) ==="
PYTHONPATH=cg-lib:tools:. python3 -u tools/bench_prefill.py \
    --adapter "$ADAPTER" --tokens 250,368,512,800 --batches 1,2,4,8,16,32 --attn sdpa >> "$LOG" 2>&1

say "=== 3. can merged weights be served? (1.35-1.52x, if switching is safe) ==="
# Under contention this needed ~7 GiB it could not get, and unmerge was measured to corrupt the
# base (log-probs off by ~1.4 and not returning). Both questions need the idle card: whether
# merge-from-a-pristine-snapshot is EXACT, and what a switch really costs.
PYTHONPATH=cg-lib:tools:. python3 -u tools/bench_merge_switch.py \
    --adapters /root/out/lora_marnie_grimmsnarl_r1,/root/out/lora_alakazam_nz_r1 \
    --tokens 368 --cycles 5 --pristine-on-gpu >> "$LOG" 2>&1

say "=== 4. where a decision's time goes: GPU forward vs Python ==="
PYTHONPATH=cg-lib:tools:. python3 -u tools/profile_decision.py \
    --adapter "$ADAPTER" --tokens 368 >> "$LOG" 2>&1
PYTHONPATH=cg-lib:tools:. python3 -u tools/profile_decision.py \
    --adapter "$ADAPTER" --tokens 368 --merge >> "$LOG" 2>&1

say "measurements done. starting the server"
exec bash tools/i2_score_serve.sh >> "$LOG" 2>&1
