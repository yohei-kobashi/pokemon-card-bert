#!/bin/bash
# Free space on instance1. Only artifacts from lines of work that are CLOSED on the record are
# removed; no source, no config, and nothing the running loop reads.
#
# Explicitly NOT touched:
#   /root/out/rerank_gte_v39      the loop's own starting model
#   /root/out/rerank_loop*        round outputs, one of which is being written right now
#   data/rerank/loop_r2.*         the mix the current training is reading
#   data/selfplay                 the game logs the reranker pool is built FROM
#   data/rerank/raw, loop_rerank/raw   the pre-dedup pools -- the only way back if the
#                                 equivalence dedup turns out to hurt, and only 163 MB
#   /root/ptcg/repo, repo_fix     source
set -u
echo "BEFORE: $(df -h / | tail -1 | awk '{print $4" free ("$5" used)"}')"
free_it() { for p in "$@"; do [ -e "$p" ] && echo "  rm $(du -sh "$p" | cut -f1)\t$p" && rm -rf "$p"; done; }

echo "--- llama.cpp / GGUF decoder path (dead: Kaggle caps submissions at 197.66 MiB) ---"
free_it /root/sft.Q4_K_M.gguf /root/sft2.Q4_K_M.gguf /root/sftv2.Q4_K_M.gguf /root/sftv2.Q6_K.gguf \
        /root/crustle_lm_submission.tar.gz /root/subm_crustle

echo "--- superseded ONNX exports (v37 kept: it is the only export that still matches a model) ---"
free_it /root/onnx_v35 /root/onnx_v36

echo "--- refuted RL runs (rl-plateau-five-refutations: flat over 12 rounds, and the policies are"
echo "    decoder+LoRA against a prompt format that has since been replaced) ---"
free_it /root/out/rlDL2 /root/out/rlA /root/out/rlDL /root/out/rlBIG

echo "--- misc ---"
free_it /root/out/smoke /root/repo_sync2.tgz /root/out/rerank_gte_v35 /root/out/rerank_gte_v36

echo "AFTER:  $(df -h / | tail -1 | awk '{print $4" free ("$5" used)"}')"
echo "--- what remains, largest first ---"
du -sh /root/* 2>/dev/null | sort -hr | head -8
