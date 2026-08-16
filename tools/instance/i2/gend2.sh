#!/usr/bin/env bash
# instance2's job from here to the deadline: produce RL traces for instance1, continuously.
#
# WHY THESE PILOTS.
#   dusknoir side = the DeBERTa champion behind the plan filter.  night6 asked whether the 4B had
#   become dramatically better and answered +1.09 +- 1.83 -- the eleventh consecutive RL round
#   inside +-2pt across both machines.  Without a dramatic gap there is no reason to pay the
#   off-policy cost: DPO teaches "prefer A over B HERE", and states visited by a different pilot
#   are not the states instance1's model will face.  [[narrow-dagger-overfits]] is the local
#   precedent -- target +11.9pt, fleet -2.75pt.
#   opponent side = each deck's own 4B LoRA (`reg`).  That is what those eleven adapters were
#   trained for, and it is the half of the matchup where off-policy is the POINT: the protagonist
#   should meet decks that are PLAYED, not merely held.  Every gate so far has faced engine_v2.
#
# instance2 cannot open a connection to instance1 (the vast proxy authenticates on account keys),
# so this writes shards and instance1's puller collects them. Nothing here reaches out.
set -u
LOG=/root/gend.log
REPO=/root/ptcg/repo
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
OUT=/root/gen_out
OPPS=${OPPS:-marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh}
GAMES=${GAMES:-40}                 # per deck per shard
SHARDS=${SHARDS:-2}
WRAP=${WRAP:-lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search,front_dive,promote_dive,promote_line,lethal_boss,candy_line,noir_critical,stadium_bump,hammer_now,spike_candy,spike_race}
# The champion moves now (MIN_GAIN=0.0 adopts on any positive delta), so the checkpoint is
# read fresh each round from the pointer instance1 writes AFTER the weights have landed.
CKPT_PTR=${CKPT_PTR:-/root/out/champion.txt}
STOP=$(date -u -d "${STOP_AFTER:-2026-08-16T23:00:00Z}" +%s)
# DUSK_* gates: without these, six of the WRAP rules exist in name only -- the
# [[plan-rule-audit-and-wrapper-bugs]] trap. Every trace before 2026-08-15 was
# generated with the prohibitions silently inert; fixed with the trio adoption.
export PLAN_UPTO1=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=cg-lib
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1 DUSK_TIPS=1 DUSK_SPIKE=1

say() { echo "[gen $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
mkdir -p "$OUT"
say "start. protagonist = planfilter + the champion from $CKPT_PTR, opponents = reg (per-deck 4B)"

R=$(cat /root/gend_round 2>/dev/null || echo 0)
SEEN=
while [ "$(date -u +%s)" -lt "$STOP" ]; do
    R=$((R + 1)); echo "$R" > /root/gend_round
    CKPT=$(cat "$CKPT_PTR" 2>/dev/null)
    if [ -z "$CKPT" ] || [ ! -s "$CKPT/model.safetensors" ]; then
        say "no usable champion at $CKPT_PTR yet -- waiting 120s"; sleep 120; R=$((R - 1)); continue
    fi
    [ "$CKPT" = "${SEEN:-}" ] || { say "champion for this round: $CKPT"; SEEN=$CKPT; }
    NDECK=$(echo "$OPPS" | tr , ' ' | wc -w)
    say "round $R: $SHARDS x $NDECK decks x $GAMES games"
    cd "$REPO"
    for S in $(seq 0 $((SHARDS - 1))); do
        nohup python3 tools/lm_mirror_log.py \
            --model "planfilter:$WRAP:hf:$CKPT" --deck-model reg --fmt dusk \
            --protagonist dragapult_dusknoir --decks "$OPPS" --games "$GAMES" \
            --seed $((300000 + R * 1000 + S * 100)) \
            --out "$OUT/gen_r${R}s$S.jsonl.gz" --trace-out "$OUT/.gtr_r${R}s${S}.part" \
            --mirror-so "$SO" > "$OUT/gen_r${R}s$S.log" 2>&1 &
    done
    wait
    # Rename only after the writer exits: the puller treats gtr_*.jsonl.gz as complete, and a
    # half-written file cost a whole night once already when it was read mid-transfer.
    for S in $(seq 0 $((SHARDS - 1))); do
        P="$OUT/.gtr_r${R}s${S}.part"
        if [ -s "$P" ] && gzip -t "$P" 2>/dev/null; then
            # the producing champion goes in the name: a round on instance1 can branch shards
            # from two or three different champions and nothing else records which
            mv -f "$P" "$OUT/gtr_r${R}s${S}_$(basename "$CKPT").jsonl.gz"
        else
            say "round $R shard $S produced no usable trace"; rm -f "$P"
        fi
    done
    say "round $R ready: $(ls -1 $OUT/gtr_r${R}s*.jsonl.gz 2>/dev/null | wc -l) shard(s) from $(basename "$CKPT"); $(( (STOP - $(date -u +%s)) / 60 )) min left"
    # the puller deletes what it has taken; this is only a backstop against a dead puller
    ls -1t "$OUT"/gtr_*.jsonl.gz 2>/dev/null | tail -n +25 | xargs -r rm -f
done
say "GEND_DONE (past stop time)"
