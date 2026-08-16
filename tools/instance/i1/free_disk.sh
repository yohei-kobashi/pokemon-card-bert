#!/usr/bin/env bash
# Free instance1's disk. The generator has been gated on MIN_FREE_GIB=12 for hours and the
# dusknoir loop writes a 365 MB checkpoint per round, so this has to buy real headroom, not a
# gigabyte. Everything removed here is either superseded by a measurement recorded elsewhere
# or a raw intermediate whose product still exists.
set -u
say() { echo "[disk $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo
df -Pk "$REPO" | awk 'NR==2 {printf "before: %d GiB free\n", int($4/1048576)}'

# 1. gte-lineage checkpoints. Measured today on the 11 decks at 150 games/deck: DeBERTa beats
#    gte by +7.8 to +12.4pt, t 3.8-5.5, on 10-11 of 11 decks. That line is closed, and the
#    numbers are in the memory file, not in these weights.
say "gte-lineage checkpoints"
du -shc /root/out/rerank_* /root/out/v41_gte /root/out/l6_r* 2>/dev/null | tail -1
rm -rf /root/out/rerank_* /root/out/v41_gte /root/out/l6_r*

# 2. DeBERTa rounds 1-3. The 11-deck evaluation covers r4..r8; r1-r3 predate it and lost.
say "d41_r1..r3 and the superseded dusknoir round"
du -shc /root/out/d41_r1 /root/out/d41_r2 /root/out/d41_r3 /root/out/dusk_r1 2>/dev/null | tail -1
rm -rf /root/out/d41_r1 /root/out/d41_r2 /root/out/d41_r3 /root/out/dusk_r1

# 3. v40-format pools. v40 is the gte-era prompt; nothing reads it now that PROMPT_FMT is v41,
#    and re-rendering from selfplay is how it would be rebuilt anyway.
say "v40 pools and pre-pilot-11 v41 base"
du -shc "$REPO"/data/rerank/v40_base.jsonl.gz "$REPO"/data/rerank/v40_mix.jsonl.gz \
        "$REPO"/data/rerank/v41_base.jsonl.gz "$REPO"/data/rerank/v34_full_v36w.rerank.jsonl.gz \
        "$REPO"/data/rerank/v39_0731*.gz "$REPO"/data/rerank/l4_r1.jsonl.gz 2>/dev/null | tail -1
rm -f "$REPO"/data/rerank/v40_base.jsonl.gz "$REPO"/data/rerank/v40_mix.jsonl.gz \
      "$REPO"/data/rerank/v41_base.jsonl.gz "$REPO"/data/rerank/v34_full_v36w.rerank.jsonl.gz \
      "$REPO"/data/rerank/v39_0731*.gz "$REPO"/data/rerank/l4_r1.jsonl.gz

# 4. July raw self-play intermediates. gen_pool deletes its own tag dirs; these are leftovers
#    from finished campaigns whose .rerank pools were built and consumed long ago.
say "July selfplay intermediates"
du -shc "$REPO"/data/selfplay/v34_full "$REPO"/data/selfplay/v39_0731 \
        "$REPO"/data/selfplay/teacher_0730 "$REPO"/data/selfplay/cur_0802 \
        "$REPO"/data/selfplay/curengine_0724 "$REPO"/data/selfplay/meta_topup_0723 \
        "$REPO"/data/selfplay/mega_starmie_v2 2>/dev/null | tail -1
rm -rf "$REPO"/data/selfplay/v34_full "$REPO"/data/selfplay/v39_0731 \
       "$REPO"/data/selfplay/teacher_0730 "$REPO"/data/selfplay/cur_0802 \
       "$REPO"/data/selfplay/curengine_0724 "$REPO"/data/selfplay/meta_topup_0723 \
       "$REPO"/data/selfplay/mega_starmie_v2

df -Pk "$REPO" | awk 'NR==2 {printf "after:  %d GiB free\n", int($4/1048576)}'
say "kept: d41_r4..r8, dusk_r2, v41_base11 (the loop's anchor), v41_dusk (the loop's data)"
ls -d /root/out/* | head -20
say DISK_DONE
