#!/bin/bash
# Training-throughput sweep on the CURRENT card, before deciding to rent a different one.
#
# The A100 measurement said the workload is not matmul-bound: 312 TFLOPS against this card's
# ~70-90 bought only 1.66x, which sits between the two cards' memory-bandwidth ratio (2.1x) and
# 1.0x. Bandwidth-bound work is not fixed by buying tensor cores, so the knobs that matter are
# the ones that move less data per step.
#
# THE BIG ONE IS GRADIENT CHECKPOINTING. sft_teacher defaults to use_gradient_checkpointing=
# "unsloth", which recomputes the forward pass in the backward to save activation memory --
# roughly 30% extra compute. The run it is protecting peaked at 25.0 GiB of a 47.4 GiB card, so
# the memory it buys may not be needed at all. Turning it off is free speed IF it fits.
#
# BATCH SIZE second: on the A100, 8 -> 32 at a constant 32 rows per optimiser step was worth 10%
# (2.05 -> 1.86 s/it) purely from doing the same work in fewer, larger kernels.
#
# Every row keeps rows-per-optimiser-step at 32, so s/it compares directly and the optimisation
# trajectory is unchanged -- this measures throughput, not a different training run.
set -u
REPO=/root/ptcg/repo
DATA=$REPO/data/sft/i2_r1.jsonl.gz
VOCAB=$REPO/data/cardfirst_b_v39.json
# The first attempt at this sweep printed five headers and no timings: the loop deletes its mix
# after training (dagger_loop_i2.sh:214 rm -f "$MIX"), so every run died on a missing file and
# the grep that keeps only timing lines hid the error. Check the input before spending the GPU.
if [ ! -s "$DATA" ]; then
  echo "STOP: $DATA does not exist. Build a slice first, e.g."
  echo "  zcat $REPO/data/sft/v40_base_sft.jsonl.gz | head -200000 | gzip > $DATA"
  exit 1
fi
cd "$REPO"
export PYTHONPATH=/root/ptcg/repo/cg-lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "######## training throughput sweep (reference: bsz8 accum4 ckpt=on = 3.08 s/it) ########"
for CFG in "8 4 on" "8 4 off" "16 2 off" "32 1 on" "32 1 off"; do
  set -- $CFG
  B=$1; A=$2; CK=$3
  FLAG=""; [ "$CK" = "off" ] && FLAG="--no-grad-ckpt"
  echo "-------- bsz=$B accum=$A grad_ckpt=$CK --------"
  rm -rf /root/out/tsweep
  # The FULL output goes to a per-config file and the filter reads THAT, so a failure the filter
  # does not recognise still exists somewhere. The first version piped straight into a grep
  # listing the patterns success produces plus the one failure I happened to anticipate (OOM);
  # a missing-file traceback matched none of them, five configs died in silence, and the log
  # showed only headers and "DONE". A filter narrow enough to make output readable is narrow
  # enough to hide the failure.
  RAW=/root/tsweep_b${B}_a${A}_${CK}.log
  timeout 1200 python3 tools/instance/sft_teacher.py --model unsloth/Qwen3-4B-Base \
      --data "$DATA" --domain-tokens --card-first "$VOCAB" \
      --out /root/out/tsweep --limit 200000 --eval-n 0 --steps 50 \
      --bsz $B --accum $A --maxlen 896 --group-by-length --save-steps 100000 $FLAG \
      > "$RAW" 2>&1
  RC=$?                                  # the PYTHON status, not a grep that found nothing
  tr '\r' '\n' < "$RAW" | grep -aE "50/50 \[|\[peak\]" | tail -2
  if [ "$RC" -ne 0 ]; then
    echo "!! FAILED rc=$RC -- last lines of $RAW:"
    tr '\r' '\n' < "$RAW" | grep -av "^$" | tail -6
  fi
done
rm -rf /root/out/tsweep
echo "######## TRAIN SWEEP DONE ########"
