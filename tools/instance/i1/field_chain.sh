#!/usr/bin/env bash
# Dusknoir RL against the FIELD, not against itself.
#
# Every RL round to date collected and gated dragapult_dusknoir vs dragapult_dusknoir. That gate
# read 61-63% while the deck rated ~330 live, and on 2026-08-12 three pilots that the mirror
# ranks differently -- engine_v2, the bare champion, and the shipped planfilter build -- all came
# out at 27.5/31.3/32.5% against alakazam_nz + marnie_grimmsnarl, indistinguishable at n=80.
# The mirror cannot see the difference that decides games, and it never could: the deck does not
# meet itself on the ladder, and mirror mode cancels to exactly zero any flaw that hurts both
# seats ([[mirror-shuffle-mode]]). The same day, an engine_v2-vs-engine_v2 ranking against these
# two opponents reproduced the live score ordering across the fleet
# ([[deck-ranking-vs-current-top2]]) -- so local measurement works, pointed at the right target.
#
# So: collect against the field, branch OUR decisions only, gate against the field.
#   collect  champion pilots dusknoir vs engine_v2 alakazam_nz + marnie_grimmsnarl
#   branch   --only-deck dragapult_dusknoir; in the mirror both seats were us, here one is
#   train    the same a/b reward A/B (beta 0 = Q+prizes, beta 0.3 = + rule conformance)
#   gate     gate_protagonist vs the SAME two decks, paired on (seed, seat)
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RL_PRIZE_GAMMA=0.25
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1  # rules exist only under these flags
# doctrine SEEDS (branch-side only; kept out of WRAP and out of rww/rwl labels)
export DUSK_SPIKE=1 DUSK_CSPLIT=1 DUSK_WIDE=1 DUSK_TIPS=1
export DUSK_HDOC=1   # user doctrines (fez_early/double_pult), doctrine-seeded + label-excluded
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
STATE=/root/loop_dusk/field
DECK=dragapult_dusknoir
OPPS=${OPPS:-marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh}
R5=lethal_now,spread_aim,clops_hold,energy_line,energy_focus
PROH=clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,hammer_spare,draw_cap
# lethal_now is NOT a taste rule like phantom_dive -- it fires only when the line takes our
# LAST prizes, and unenforced the model left one on the table in 22 of 160 games. Restored on
# correctness grounds, at the front so it reads first.
# search_bottom (prohibition) and setup_search (positive) added 08-14 on +4.10 and
# +6.55 paired points over 6,000 games; both read only our own board.
# front_dive,promote_dive,promote_line: +4.75 +- 0.66 on the shipping pilot
# (lmab, 150x8, all 8 opponents up), adopted by the user 2026-08-15.
# lethal_boss: +0.58 +- 0.32 (lmab4, 150x8, non-negative on all 8), adopted 2026-08-15.
# tips+spike: +0.50/+0.17/+0.67 vs base (lmab7, pokehubguide deck), adopted 2026-08-16
WRAP_RULES=${WRAP_RULES:-lethal_now,$PROH,search_bottom,setup_search,front_dive,promote_dive,promote_line,lethal_boss,candy_line,noir_critical,stadium_bump,hammer_now,spike_candy,spike_race,lethal_line}
CUR=${CUR:-/root/out/mrl2_r5b}
FROM=${FROM:-1}
ROUNDS=${ROUNDS:-200}
# User directive 2026-08-12: repeat this same RL until the day before the deadline and
# meet a plateau with learning-rate / data-allocation changes, not with a new method.
# So the loop stops on the CLOCK, never on a run of misses.
STOP_AFTER=${STOP_AFTER:-2026-08-16T23:00:00Z}
# The opponent pilot. engine_v2 now; flip to "reg" once this converges, which points each
# opponent deck at its own Qwen-4B LoRA from instance2 -- that handoff is the whole reason
# those adapters are being trained, and it is a one-word change here.
OPP_SPEC=${OPP_SPEC:-engine}
# 250 per (arm, opponent) = 500 paired games per arm. The mirror chain used 600 and reported
# SE ~2.1; two opponents at 250 is the same order for the same wall-clock, and it buys the thing
# that matters more than another 100 games -- the games are against decks we actually play.
GATE_GAMES=${GATE_GAMES:-100}
# User directive 08-14: adopt on ANY positive delta. The +1.0pt bar refused ten straight
# rounds; the gate is paired and unbiased, so the point estimate is the right thing to act
# on when only a handful of rounds remain before STOP_AFTER.
export MIN_GAIN=${MIN_GAIN:-0.0}
# Setup potential, 08-14. qmin drops ~60% of pairs as coin flips; those are labelled instead by
# which candidate left us further along the human opening (line bodies / Drakloak / energy that
# pays {R}{P}), measured at OUR turn boundary inside the playout, capped at the template so it
# cannot be farmed. Measured reach: 16.5% of pairs separate, +22% usable rows. The label is
# deliberately weak (0.65) because FORCING the same preference cost -2.25pt.
# PHI_MIN=0 turns it off and reproduces the previous converter exactly.
PHI_MIN=${PHI_MIN:-0.10}
PHI_WC=${PHI_WC:-0.65}
COLLECT=${COLLECT:-36}                        # per opponent; 8 x 100 = 800 games/round
# 20, not 24: branchd2 is serving instance2's pass3 with 24 workers on 61.4 effective cores
# ([[vast-cpu-quotas]] -- nproc says 112 and lies). 24+24 would oversubscribe both.
WORKERS=${WORKERS:-20}
PFX="planfilter:$WRAP_RULES:"                  # prohibitions-only; gated 08-12 at +4.00pt
mkdir -p "$STATE"
say() { echo "[field $(date -u +%m-%d_%H:%M:%S)] $*"; }

rules_fp() {   # what the pilot IS, as one hash
    md5sum "$REPO/tools/dusk_plan.py" "$REPO/lm/plan_filter.py" 2>/dev/null \
        | awk '{print $1}' | md5sum | cut -d' ' -f1
}

prune_ckpts() {  # keep the champion and ONE fallback; a rejected arm is 746 MB of nothing
    local keep1="$1" keep2="$2" d
    for d in /root/out/fld_r*[ab]; do
        [ -d "$d" ] || continue
        [ "$d" = "$keep1" ] && continue
        [ "$d" = "$keep2" ] && continue
        rm -rf "$d"
    done
}

disk_ok() {      # refuse to START a train we cannot finish, rather than truncate one mid-write
    local free
    free=$(df -BG /root | awk 'NR==2{gsub("G","",$4); print $4}')
    [ "${free:-0}" -ge "${1:-5}" ]
}

ok_gz() {   # a gzip that fails its CRC / end-of-stream check is not a resumable artifact
    [ -s "$1" ] && gzip -t "$1" 2>/dev/null
}

gpu_wait() {
    local u _i=0
    for _ in $(seq 1 360); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$u" -le 2000 ] && return 0
        [ $((_i % 20)) -eq 0 ] && say "waiting for the GPU (${u} MiB in use)"
        _i=$((_i+1))
        sleep 30
    done
    say "STOP: GPU held ${u} MiB for 3 h -- something is wedged"; exit 1
}

# ------------------------------------------------------- 0. where we actually start
# The mirror said 61-63%. Record what the champion scores against the FIELD before any round, so
# every later number is read against the right baseline rather than against the mirror's.
if [ ! -s "$STATE/base.json" ]; then
    gpu_wait
    say "baseline: champion vs $OPPS, $GATE_GAMES games/opponent"
    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$OPPS" --games "$GATE_GAMES" \
        --seed 88000 --baseline cur --opp-spec "$OPP_SPEC" \
        --arm "cur=${PFX}hf:$CUR@dusk" \
        --mirror-so "$SO" --out "$STATE/base.json" > "$STATE/base.log" 2>&1 \
        || { say "baseline FAILED"; tail -8 "$STATE/base.log"; exit 1; }
    grep -aE "vs |win%|^cur " "$STATE/base.log" | tail -5
fi

MISSES=0
BUDGET=6000
APOW=${ALLOC_POWER:-1.0}
QMIN=${QMIN:-0.35}
TEMP=${TEMP:-0.25}
ALLOC_MIN=${ALLOC_MIN:-0.5}
ALLOC_MAX=${ALLOC_MAX:-2.0}
for R in $(seq "$FROM" "$ROUNDS"); do
    if [ "$(date -u +%s)" -ge "$(date -u -d "$STOP_AFTER" +%s)" ]; then
        say "reached STOP_AFTER ($STOP_AFTER) -- ending with champion $CUR"; break
    fi
    FP0=$(rules_fp)
    say "================ field round $R (champion $CUR) ================"
    TR=/root/fld_tr$R.jsonl.gz
    PAIRS=/root/fld_pairs$R.jsonl.gz

    if ! ok_gz "$TR"; then
        [ -e "$TR" ] && { say "discarding a truncated $TR"; rm -f "$TR" /root/fld_log$R.jsonl.gz; }
        gpu_wait
        # Spend the SAME total games unevenly: the matchups the champion loses get more of the
        # round. Measured on round 4, the pairs reaching training were 11.3-14.3% per opponent
        # against win rates spanning 3.3-49.3% -- the loop was learning the won matchups as hard
        # as the lost ones. Clamped to [ALLOC_MIN, ALLOC_MAX] x even, because concentrating a
        # round on one matchup is a move already measured and lost (narrow DAgger: +11.9pt on
        # the target, -2.75pt on the fleet).
        NOPP=$(echo "$OPPS" | tr ',' '\n' | grep -c .)
        PREV_GATE="$STATE/gate_r$((R-1)).json"; [ -f "$PREV_GATE" ] || PREV_GATE="$STATE/base.json"
        GPD=""
        if [ -f "$PREV_GATE" ]; then
            GPD=$(python3 tools/field_alloc.py --gate "$PREV_GATE" --arm cur \
                    --total $((COLLECT * NOPP)) --power "$APOW" \
                    --min-mult "$ALLOC_MIN" --max-mult "$ALLOC_MAX" \
                    --order "$OPPS" --report 2>>"$STATE/alloc$R.log") \
                || { say "alloc FAILED -- falling back to the even split"; tail -3 "$STATE/alloc$R.log"; GPD=""; }
        fi
        if [ -n "$GPD" ]; then
            say "alloc from $(basename "$PREV_GATE") (power $APOW): $GPD"
            sed 's/^/    /' "$STATE/alloc$R.log"
        else
            say "alloc: no previous gate -- even split, $COLLECT per opponent"
        fi
        say "collect: $((COLLECT * NOPP)) games over $OPPS, champion through the wrapper"
        python3 tools/lm_mirror_log.py --model "${PFX}hf:$CUR" --deck-model engine --fmt dusk \
            --protagonist "$DECK" --decks "$OPPS" --games "$COLLECT" \
            ${GPD:+--games-per-deck "$GPD"} \
            --seed $((600000 + R * 1000)) \
            --out /root/fld_log$R.jsonl.gz --trace-out "$TR" --mirror-so "$SO" \
            > "$STATE/collect$R.log" 2>&1 \
            || { say "collect FAILED"; tail -8 "$STATE/collect$R.log"; exit 1; }
        grep -aE "games \|" "$STATE/collect$R.log" | tail -3
    fi
    ok_gz "$TR" || { say "STOP: $TR is missing or truncated after collection"; exit 1; }

    if ! ok_gz "$PAIRS"; then
        [ -e "$PAIRS" ] && { say "discarding a truncated $PAIRS"; rm -f "$PAIRS"; }
        say "branch: our decisions only, rule weights on, R5 excluded"
        # instance2's shards, if its generator has delivered any. Capped per round so one big
        # backlog cannot make a single round's branch run for hours.
        TRALL="$TR"
        GEN=$(ls -1 /root/gen_in/gtr_*.jsonl.gz 2>/dev/null | head -${GEN_MAX:-3} | paste -sd,)
        if [ -n "$GEN" ]; then
            TRALL="$TR,$GEN"
            # Provenance, not a filter. User directive 08-14: traces from an older champion
            # are used as they are. What matters is that the log SAYS which policies produced
            # this round's pairs, so a surprising round can be read against the mix.
            say "branching with $(echo "$GEN" | tr , '\n' | wc -l) shard(s) from instance2: $(echo "$GEN" | tr , '\n' | sed 's/.*_\(fld_[a-z0-9]*\)\.jsonl\.gz/\1/' | sort | uniq -c | tr '\n' ' ')"
        fi
        CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/dpo_branch.py \
            --traces "$TRALL" --fmt dusk --only-deck "$DECK" --rule-weights \
            --rule-exclude "$WRAP_RULES,spike_candy,spike_race,crispin_split,third_loak,phantom_dive,lethal_line,fez_early,double_pult,spread_evolve" \
            --doctrine-rules spike_candy,spike_race,crispin_split,third_loak,fez_early,double_pult,stadium_bump --doctrine-per-game 2 \
            --budget "${BUDGET:-6000}" --per-game 15 --margin-min 0.01 --playouts 24 --workers "$WORKERS" \
            --out "$PAIRS" > "$STATE/branch$R.log" 2>&1 \
            || { say "branch FAILED"; tail -8 "$STATE/branch$R.log"; exit 1; }
        grep -aE "^wrote|selected" "$STATE/branch$R.log" | tail -2
        # Retire what was just branched: a shard left in place would be re-branched next round,
        # spending the budget on states this round already mined.
        if [ -n "$GEN" ]; then
            mkdir -p /root/gen_used
            echo "$GEN" | tr , '\n' | xargs -r -I{} mv -f {} /root/gen_used/ 2>/dev/null || true
        fi
    fi
    NP=$(zcat "$PAIRS" 2>/dev/null | wc -l)
    say "pairs: $NP"
    [ "$NP" -ge 500 ] || { say "STOP: only $NP pairs"; exit 1; }

    if [ "$(rules_fp)" != "$FP0" ]; then
        say "RULES CHANGED mid-round -- the traces were collected by a different pilot."
        say "Discarding round $R data and re-collecting under the new rules."
        rm -f "$TR" "$PAIRS" /root/fld_log$R.jsonl.gz
        continue
    fi
    # LR / EPOCH / DATA ladder, indexed by the current miss streak. Cycles, so a long plateau
    # keeps sampling the space instead of re-running one setting that has already failed. lr is
    # the first knob because [[mirror-rl-training-is-the-noise-source]] measured a 26pt row-order
    # swing at 1e-5 against 1.00pt at 2e-6 -- above ~4e-6 this training is a lottery, not a fit.
    case $((MISSES % 5)) in
        0) LR=2e-6; EP=0.5; BUDGET=5000; APOW=1.0; QMIN=0.35 ;;
        1) LR=4e-6; EP=0.5; BUDGET=5000; APOW=1.0; QMIN=0.35 ;;
        2) LR=2e-6; EP=1.0; BUDGET=5000; APOW=1.0; QMIN=0.50 ;;
        3) LR=2e-6; EP=0.5; BUDGET=7000; APOW=1.0; QMIN=0.35 ;;   # more branch points
        4) LR=1e-6; EP=1.0; BUDGET=7000; APOW=2.0; QMIN=0.50 ;;   # harder on the lost matchups
    esac
    say "knobs (miss streak $MISSES): lr $LR epochs $EP branch-budget $BUDGET alloc-power $APOW qmin $QMIN"

    for V in a b; do
        BETA=0.0; [ "$V" = "b" ] && BETA=0.3
        ROWS=$STATE/rows_r$R$V.jsonl.gz
        # The playout advantage |qw-ql| is the label's confidence. Measured over rounds 7-9
        # (4,519 pairs): training on ALL of them moved held-out conformance 54.3 -> 53.6, i.e.
        # DOWN, while >=0.35 moved it 52.1 -> 58.1 and >=0.60 drove the loss below ln(2) for the
        # first time. With 24 playouts the Q estimate has an SE near 0.2 and the median pair
        # margin is 0.26, so the low-confidence majority was outvoting the real signal -- which
        # is why nine rounds trained at exactly chance while still perturbing the weights enough
        # to cost ~3.6pt an arm.
        NR=0
        for Q in "$QMIN" 0.25 0.15 0.0; do
            python3 /root/mrl_convert.py --pairs "$PAIRS" --out "$ROWS" \
                --beta "$BETA" --temp "$TEMP" --qmin "$Q" \
                --phi-min "${PHI_MIN:-0.10}" --phi-wc "${PHI_WC:-0.65}" \
                | tee -a "$STATE/convert$R.log"
            NR=$(zcat "$ROWS" | wc -l)
            [ "$NR" -ge 500 ] && { QUSED=$Q; break; }
            say "qmin $Q leaves only $NR rows -- relaxing (a thin round must not end the run)"
        done
        [ "$NR" -ge 500 ] || { say "STOP: only $NR rows even unfiltered"; exit 1; }
        [ "$QUSED" = "$QMIN" ] || say "NOTE: round $R trained at qmin $QUSED, not $QMIN"
        # Expert rows from the user's play_server games (21 HumanvAI games vs ogerpon_mono and
        # mega_abomasnow, 14 won): listwise rows in the same prompt format, appended to every
        # round's mix. The gate decides if they help -- a challenger still has to beat the
        # champion on the 8-opponent field to ship.
        for HF_ROWS in /root/human_rows.jsonl.gz /root/human_dpo_rows.jsonl.gz /root/human_doctrine_rows.jsonl.gz; do
            [ -f "$HF_ROWS" ] && cat "$HF_ROWS" >> "$ROWS"
        done
        NR=$(zcat "$ROWS" | wc -l)
        say "human expert + dpo-divergence + doctrine rows appended -> $NR total rows"
        gpu_wait
        disk_ok 5 || { say "STOP: only $(df -BG /root | awk 'NR==2{print $4}') free -- refusing to start a train that cannot finish"; exit 1; }
        say "train $V: beta $BETA, $NR rows (qmin $QUSED temp $TEMP), lr $LR ep $EP l2sp 1e-2"
        python3 tools/dusk_plan_train.py --data "$ROWS" --model "$CUR" \
            --out /root/out/fld_r$R$V --lr "$LR" --epochs "$EP" --accum 4 --l2sp 1e-2 \
            > "$STATE/train$R$V.log" 2>&1 \
            || { say "train $V FAILED"; tail -6 "$STATE/train$R$V.log"; exit 1; }
        grep -aE "FINAL|\[eval\]" "$STATE/train$R$V.log" | tail -2
        [ -f "/root/out/fld_r$R$V/model.safetensors" ] || { say "STOP: no checkpoint $V"; exit 1; }
    done

    if [ "$(rules_fp)" != "$FP0" ]; then
        say "RULES CHANGED during training -- gate would score a pilot the data never used."
        say "Discarding round $R and re-collecting."
        rm -f "$TR" "$PAIRS" /root/fld_log$R.jsonl.gz
        continue
    fi
    gpu_wait
    say "gate: champion vs a vs b vs $OPPS, $GATE_GAMES games/opponent"
    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$OPPS" --games "$GATE_GAMES" \
        --seed $((81000 + R * 100)) --baseline cur --opp-spec "$OPP_SPEC" \
        --arm "cur=${PFX}hf:$CUR@dusk" \
        --arm "a=${PFX}hf:/root/out/fld_r${R}a@dusk" \
        --arm "b=${PFX}hf:/root/out/fld_r${R}b@dusk" \
        --mirror-so "$SO" --out "$STATE/gate_r$R.json" > "$STATE/gate_r$R.log" 2>&1 \
        || { say "gate FAILED"; tail -10 "$STATE/gate_r$R.log"; exit 1; }
    grep -aE "vs |delta|^arm|^a |^b |^cur " "$STATE/gate_r$R.log" | tail -10

    WIN=$(python3 - "$STATE/gate_r$R.json" <<'PY'
import json, os, sys
arms = json.load(open(sys.argv[1])).get("arms", {})
best, bd = None, None
for k in ("a", "b"):
    d = (arms.get(k) or {}).get("delta_vs_baseline")
    if d is not None and (bd is None or d > bd):
        best, bd = k, d
MIN = float(os.environ.get("MIN_GAIN", "0.0"))
print(best if (bd is not None and bd > MIN) else "none")
print("a %+.2f | b %+.2f" % ((arms.get("a") or {}).get("delta_vs_baseline", float("nan")),
                             (arms.get("b") or {}).get("delta_vs_baseline", float("nan"))),
      file=sys.stderr)
PY
)
    say "round $R winner: $WIN"
    PREV_CUR=${PREV_CUR:-}
    if [ "$WIN" = "none" ]; then
        MISSES=$((MISSES+1))
        say "round $R: no challenger was positive (> ${MIN_GAIN:-0.0}pt) -- champion stays $CUR ($MISSES in a row)"
        # No break. A miss advances the knob ladder below and the loop goes again; the previous
        # behaviour (stop after three) is what left the GPU idle for six hours on 08-11.
    else
        MISSES=0
        PREV_CUR=$CUR
        CUR=/root/out/fld_r$R$WIN
        echo "$CUR" > "$STATE/current.txt"
        say "new champion: $CUR"
        # The registry is what build_rerank_submission --from-registry reads, so a champion that
        # is not written here cannot reach a tarball.
        python3 tools/adapters.py set "$DECK" --target "hf:$(basename $CUR)" --fmt dusk \
            --wrap "planfilter:$WRAP_RULES" \
            --note "FIELD-gated champion, round $R arm $WIN (vs $OPPS, not the mirror)" || true
    fi
    prune_ckpts "$CUR" "$PREV_CUR"
    rm -f /root/fld_log$R.jsonl.gz "$STATE"/rows_r$R[ab].jsonl.gz
    say "disk: $(df -BG /root | awk 'NR==2{print $4}') free after pruning round $R"
done
say "FIELD_CHAIN_DONE (champion: $CUR)"
