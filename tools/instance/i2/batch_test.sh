#!/usr/bin/env bash
# Stop pass4 at the end of its CURRENT deck round, measure batching on an idle card, then let
# the remaining decks finish.
#
# Why a pause rather than "wait for pass4". The batching question cannot be answered on a shared
# card: "is there idle capacity a bigger batch could fill" has the answer "no" by construction
# while another process is saturating the GPU. The first sweep ran that way and produced a
# clean-looking table that meant nothing. pass4 has ~2 decks left, so waiting for all of it
# costs hours; stopping cleanly between decks costs minutes and gives a genuinely idle card.
#
# Merging is deliberately NOT tested any more. It was measured at 1.35-1.52x, but it needs a
# pristine snapshot per switch and the adapter set is large -- and `unmerge` was measured to
# corrupt the base weights outright (bf16 W +/- BA does not return). Not a path to take three
# days out.
set -u
cd /root/ptcg/repo
LOG=/root/batch_test.log
DECK_DONE=${DECK_DONE:-mega_abomasnow_sample r1}
ADAPTER=${ADAPTER:-/root/out/lora_marnie_grimmsnarl_r1}
say() { echo "[batch $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for the current round to end: '$DECK_DONE'"
while ! grep -aqE "$DECK_DONE (done|FAILED)|PASS4_DONE" /root/pass4.log 2>/dev/null; do
    pgrep -f "/root/pass4.sh" > /dev/null || { say "pass4 already gone"; break; }
    sleep 60
done
say "round over: $(tail -1 /root/pass4.log)"

# Stop the queue BEFORE the next deck grabs the card. Parent first so it cannot spawn another.
for P in $(pgrep -f "/root/pass4.sh"); do kill "$P" 2>/dev/null && say "stopped pass4 pid $P"; done
sleep 2
for P in $(pgrep -f "deck_lora2.sh"); do kill "$P" 2>/dev/null && say "stopped deck_lora2 pid $P"; done
sleep 2
for P in $(pgrep -f "gen_selfplay|dpo_branch|mirror_match|dusk_plan_train"); do
    kill "$P" 2>/dev/null && say "stopped worker pid $P"
done

say "waiting for the card to actually empty"
for _ in $(seq 60); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 2000 ] && [ "$G" -lt 10 ] && break
    sleep 10
done
say "GPU ${U} MiB / ${G}% -- starting the sweep (bench_prefill refuses if this is wrong)"

say "=== batching, on an idle card, at the token lengths real decisions have ==="
PYTHONPATH=cg-lib:tools:. python3 -u tools/bench_prefill.py \
    --adapter "$ADAPTER" --tokens 250,368,512,800 --batches 1,2,4,8,16,32 \
    --attn sdpa >> "$LOG" 2>&1
say "sweep rc=$?"

say "=== what a single decision costs, unbatched, on an idle card ==="
PYTHONPATH=cg-lib:tools:. python3 -u tools/profile_decision.py \
    --adapter "$ADAPTER" --tokens 368 >> "$LOG" 2>&1

say "=== every adapter must still reproduce its standalone checkpoint ==="
PYTHONPATH=cg-lib:tools:. python3 -u tools/score_server.py \
    --from-registry --verify all --no-serve >> "$LOG" 2>&1
say "verify rc=$?"

say "measurements done -- resuming the remaining LoRA rounds"
exec setsid --fork nohup bash /root/pass5.sh > /dev/null 2>&1 < /dev/null
