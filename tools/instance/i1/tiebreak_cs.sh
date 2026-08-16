#!/usr/bin/env bash
# v35 vs v36 on the SUBMISSION deck only, at a sample size that can actually resolve them.
# The 9-cell run put crustle_stall at 67.2 (v35) vs 58.3 (v36) on 180 games each = 1.7 SE:
# suggestive, not decisive, and it is the one deck we would ship. 150 games/cell -> 450 per
# model, SE 2.4pt per deck, 3.4pt on the difference.
#
# EACH MODEL IS EVALUATED IN ITS OWN PROMPT FORMAT (v35 static/no-shuffle, v36
# remaining/shuffle). Using one format for both would measure a format mismatch, not a model.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
export DECKS="crustle_stall"
echo "=== tiebreak start $(date -u) ==="
DECKS="crustle_stall" tools/eval_rerank_par.sh /root/out/wr_v35_cs /root/out/rerank_gte_v35 torch "" 8 150 1000000 none static 0
echo "=== v35 done $(date -u) ==="
DECKS="crustle_stall" tools/eval_rerank_par.sh /root/out/wr_v36_cs /root/out/rerank_gte_v36 torch "" 8 150 1000000 none remaining 1
echo "=== TIEBREAK_ALL_DONE $(date -u) ==="
