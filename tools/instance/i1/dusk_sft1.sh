#!/bin/bash
# instance1 goes single-deck: re-SFT d41_r8 on the dusknoir pool with the DECK[] segment gone
# and the opponents cut down to the eleven instance2 trains against.
#
# WHY THIS IS A CONTINUATION AND NOT A FRESH TRAIN. The prompt lost 57 tokens of a constant
# prefix (371 -> 314); everything the model actually reads -- board, menu, ID -- is byte-for-byte
# what it was. d41_r8 is eight warm-started rounds deep and none of that is worth discarding to
# absorb a prefix deletion. --l2sp anchors the weights to d41_r8 so the round adapts to the new
# rendering instead of wandering off it (`loops-warm-start-switch`: from-scratch rounds left
# nothing carrying progress).
#
# The pool: data/rerank/v41_dusk11.jsonl.gz, 3,041,159 rows, stride-sampled out of the 36.49M
# eligible rows of the 40.17M pool. DECK segment stripped on 100% of them, 0 misses.
set -u
REPO=/root/ptcg/repo
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=$REPO/data/rerank/v41_dusk11.jsonl.gz
FROM=/root/out/d41_r8
OUT=/root/out/dusk_s1
LOG=/root/dusk_sft1.log

say() { echo "[dusk-sft $(date -u +%m-%d_%H:%M:%S)] $*"; }

[ -s "$DATA" ] || { say "no data at $DATA"; exit 1; }
[ -f "$FROM/model.safetensors" ] || { say "$FROM is not a checkpoint"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
cp -r "$FROM"/. "$OUT"/ || { say "could not seed $OUT from $FROM"; exit 1; }
# rr_progress.json records how many samples the PREVIOUS round saw; carried over it would
# fast-forward straight past this round's entire mix.
rm -f "$OUT/rr_progress.json"

say "continuing $FROM -> $OUT on $(zcat "$DATA" | wc -l) rows"
# max-len stays 512. The shorter prompt would fit 448, but right-truncation deleting the board
# and menu is a failure this project has already shipped once (`rerank-prompt-truncation-bug`)
# and the speed it would buy is not worth re-opening it.
# --fp32 IS LOAD-BEARING. Without it the weights stay in the checkpoint's bf16, where an AdamW
# step of 1e-5 is smaller than the ulp of a typical weight and rounds away to nothing. Measured
# on eight rows of one rule, 40 passes, lr 1e-5: bf16 wanders 6/8 -> 1/8 correct and never
# fits; fp32 reaches 8/8 by pass five and holds. The flag keeps the matmuls in bf16 under
# autocast, so this costs throughput, not memory. Every "the model cannot learn this" result
# taken before this line was measured through that floor.
python3 tools/train_rerank.py --data "$DATA" --out "$OUT" --resume --fp32 \
    --deadline-h 8 --max-samples 400000 --lr 1e-5 \
    --pair-batch 32 --accum 2 --max-len 512 \
    --eval-n 2000 --grad-ckpt --margin-weight 0.5 --l2sp 1e-3 \
    || { say "train FAILED"; exit 1; }

[ -f "$OUT/model.safetensors" ] || { say "no model saved"; exit 1; }
# A --resume that silently fell through to the base model trains for hours and looks normal;
# instance2 lost 10 GPU-hours to exactly this.
grep -q "RESUME from $OUT" "$LOG" || { say "STOP: the run never reported RESUME -- it trained from base"; exit 1; }
say "done -> $OUT"
