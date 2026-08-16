#!/usr/bin/env bash
# Engine-piloted traces of the pokehubguide deck for the doctrine-seeded branch
# (user 2026-08-16). CPU-only -- the GPU belongs to the round trainer and lmab7.
# Shards land in /root/gen_in where the chain branch step already globs gtr_*; the
# 4-shard waiting cap keeps instance2 traces owning most of the GEN_MAX=12 slots.
set -u
LOG=/root/engd.log
say(){ echo "[engd $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
OPPS=marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh
STOP=$(date -u -d 2026-08-16T12:00:00Z +%s)
say "start: engine-piloted dragapult_dusknoir vs the 8-deck field"
R=1
while [ "$(date -u +%s)" -lt "$STOP" ]; do
  N=$(ls /root/gen_in/gtr_eng* 2>/dev/null | wc -l)
  if [ "$N" -ge 4 ]; then sleep 120; continue; fi
  SEED=$((900000 + R * 500))
  nice -n 10 python3 tools/lm_mirror_log.py --model engine --deck-model engine --fmt dusk \
      --protagonist dragapult_dusknoir --decks "$OPPS" --games 40 --seed "$SEED" \
      --out /root/eng_out/englog_r$R.jsonl.gz --trace-out /root/gen_in/.gtr_engr$R.part \
      --mirror-so /root/ptcg/repo/data/kaggle_engine_ext/libcg_mirror.so >> "$LOG" 2>&1 \
      && mv /root/gen_in/.gtr_engr$R.part /root/gen_in/gtr_engr$R.jsonl.gz \
      && say "shard engr$R ready (8x40x2 games)"
  R=$((R+1))
  sleep 10
done
say ENGD_DONE
