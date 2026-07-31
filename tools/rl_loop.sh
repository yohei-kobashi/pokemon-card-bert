#!/bin/bash
# Post-SFT RL curriculum driver for the CROSS-ENCODER policy (docs/rl_design.md §8-9).
#
#   bash tools/rl_loop.sh <SFT_CHECKPOINT_DIR>
#   e.g. bash tools/rl_loop.sh /root/out/rerank_gte_v37
#
# GOAL (user, 2026-07-28): bring the LM up to engine_v2's level using games against
# engine_v2. That closes the loop tightly: all 63 agents/*.py run engine_v2, so the training
# opponent, the eval opponent and the baseline being chased are the SAME agent, and the
# reward (beat engine_v2) measures the goal with no proxy in between. Imitation accuracy
# disagreed with win rate three times (v34, v36, v37); win/loss cannot.
#
# Pipeline:
#   0.  GATE: policy vs engine_v2 on the held-out protocol. Too weak -> fix SFT first.
#   A.  broad climb   (P = all decks, headroom-weighted; O = engine_v2)  until plateau.
#   B.  meta realign  (P = all, meta-weighted; O = LIVE-meta frequencies) until plateau.
#   C.  targeted      (P = the TARGET deck only; O = the LIVE field, NOT narrowed).
#
# WHAT CHANGED ON 2026-07-28 (this file was written for the abandoned Qwen decoder):
#   * no base+LoRA split      -- the 149M reranker is fully fine-tuned; POLICY is a model dir
#   * no merge / init_adapter -- there is nothing to merge; round 0 starts at the SFT dir
#   * no QLoRA in Stage C     -- deploy quantisation is post-hoc INT8 ONNX, not training-time
#   * no batched infer server -- measured 32 games / 2461 decisions in 54 s sequentially, so a
#                                768-game round is ~21 min; rl_infer.py's adapter-grouping
#                                server existed to make a decoder affordable and is now dead
#   * no fla/triton preflight -- that stack was Qwen3.5's DeltaNet; a ModernBERT encoder needs
#                                only torch + transformers
#   * no peer data-parallel   -- 2x on a 21-minute round is not worth the failure surface
#   * eval through eval_rerank.py -- the same harness the submission decision uses
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/cg-lib"
export OMP_NUM_THREADS=1

SFT_CKPT="${1:?usage: rl_loop.sh <SFT_CHECKPOINT_DIR>}"
WORK="${RL_WORK:-out/rl}"; mkdir -p "$WORK"
TEMP="${RL_TEMP:-1.0}"                 # sampling temperature for the rollout policy
MCAP="${RL_MATCHUPS:-96}"              # (pilot,opp) pairs per round; x gpm = games/round
HEUR="${RL_HEURISTIC_FRAC:--1}"        # -1 = use the stage's own opp_agent_mix
OPP_MODEL="${RL_OPP_MODEL:-}"          # frozen policy for the LM share (needed if HEUR < 1)
# --- EVALUATION SIZING ------------------------------------------------------------
# The gate must resolve the change it is asked to judge. "<1pt over 3 rounds" against
# 9 cells x 60 games has SE 2.15pt per measurement and 3.0pt on a round-to-round
# difference, so the rule fires on noise -- exactly how 60-game cells produced three wrong
# calls on 2026-07-27 (v36 crustle_stall read 58.3 at 60 games and 65.9 at 900). So: a CHEAP
# per-round trend (the rollout's own win rate, free, 768 games) plus an EXPENSIVE held-out
# gate every RL_EVAL_EVERY rounds, sized to see 2pt.
#
# The decks are the Stage-C targets, not the historical trio: Stage A puts 75% of its pilot
# mass on them (rl_config._stageA_pilot_weights), and measuring three decks while training
# sixty-three is how a gate goes blind to what it gates.
EVAL_EVERY="${RL_EVAL_EVERY:-4}"       # run the held-out gate every N rounds
# 2026-07-29: raised 150 -> 600 after re-scoring ONE checkpoint twice gave -4.37 and -6.97.
# 21 cells x 150 games is SE 0.89pt per gate and 1.26pt on a difference of two, so every
# 1-2pt reading in the whole plateau series was under the instrument's resolution and one
# 2-sigma excursion (rlDL r4) was read as a win. 600/cell = 12,600 games -> SE 0.45pt,
# diff 0.63pt, ~1.9 h; gating every 4 rounds instead of every 2 keeps the cost ratio.
EVAL_GAMES="${RL_EVAL_GAMES:-600}"     # per cell; 7 decks x 3 opps x 600 = 12600 games
# --- DECISION-LEVEL GROUPS (2026-07-29) --------------------------------------------
# Game-level GRPO alone plateaued: 8,064 games, 0.0pt of gate movement, because one scalar
# has to explain ~70 decisions. BRANCH re-plays a few positions per game with each of the top
# candidates so the update can credit the decision itself. It costs no GPU -- the playouts run
# on the engine's native search tree with engine_v2, ~0.09 s each, on cores that sit idle
# while the GPU works. Set BRANCH=0 to reproduce the old game-level-only round exactly.
BRANCH="${RL_BRANCH:-8}"               # branch points per game (cap; placement is signal-weighted)
BRANCH_K="${RL_BRANCH_K:-4}"           # candidates compared at a branch point
BRANCH_PLAY="${RL_BRANCH_PLAYOUTS:-4}" # scenarios per branch point
BRANCH_W="${RL_BRANCH_WEIGHT:-1.0}"    # weight of the decision-level term in the loss
# The update is 61% of a round's wall clock (rollout 27 min / train 64 / gate 14 at 3x size).
# Terminal-only reward makes every decision in a game share one advantage, so a uniform sample
# of the flat decision list estimates the same mean gradient -- rl_train's own docstring argues
# this. Halving it buys ~32 min a round; the honest framing is "more rounds per hour", not a
# free lunch. Branched decisions are unaffected: they carry their own attributed signal.
DEC_FRAC="${RL_DECISION_FRAC:-0.5}"
EVAL_DECKS="${RL_EVAL_DECKS:-$(python -c 'import sys;sys.path.insert(0,"tools");import rl_config;print(",".join(rl_config.STAGE_C_TARGETS))')}"
EVAL_OPPS="${RL_EVAL_OPPS:-alakazam,crustle,dragapult}"
# PLATEAU: patience against the BEST gate, not a single step against the previous one.
# The old rule ("stop unless this gate beat the last one by >=2pt") demanded a 2pt gain EVERY
# gate interval, which is larger than the instrument's own difference noise -- so it fired on
# noise and killed both runs after their second gate: rlBIG at r4, and rlDL at r6 because the
# r4 gate had blipped 2pt high and the honest r6 reading looked like a collapse. Requiring a
# sustained failure to improve on the best tolerates one bad gate, which is exactly what a
# 0.63pt-noise measurement will produce from time to time.
PLATEAU_PT="${RL_PLATEAU_PT:-0.005}"   # improvement over the BEST gate that counts as progress
PATIENCE="${RL_PATIENCE:-2}"           # consecutive gates without such progress before stopping
BASE_DIR="${RL_BASELINE_DIR:-$WORK/baseline}"   # engine_v2 control on the SAME cells
BASE_GAMES="${RL_BASELINE_GAMES:-300}"
GATE_MIN="${RL_GATE_MIN:--0.25}"       # start only if the policy is within 25pt OF THE ENGINE

# The gate is a DIFFERENCE (LM - engine_v2 on identical cells), not an absolute win rate.
# The goal is "reach engine_v2's level", and absolutes cannot say that: over these decks the
# engine itself averages 37.4% and manages 24.0% with dragapult, so any fixed threshold is
# unreachable on half the grid and free on the other half. 0 = parity = target reached.

# The prompt format is pinned in tools/rl_config.PROMPT_FMT and must describe $SFT_CKPT.
# Print it once so a mismatch is visible in the log rather than silent in the gradient.
python - "$SFT_CKPT" <<'PY'
import sys, os
sys.path.insert(0, "tools")
import rl_config
print("policy      :", sys.argv[1])
print("PROMPT_FMT  :", rl_config.PROMPT_FMT)
print("match target: %.0f%% (engine units; LM deficit %.0fpt)"
      % (rl_config.MATCH_TARGET_WR, rl_config.LM_ENGINE_DEFICIT))
assert os.path.exists(os.path.join(sys.argv[1], "config.json")), "not a model dir"
PY

ensure_baseline () {
  mkdir -p "$BASE_DIR"
  local missing=0
  for d in ${EVAL_DECKS//,/ }; do
    for o in ${EVAL_OPPS//,/ }; do
      [ -f "$BASE_DIR/${d}__${o}.json" ] && continue
      missing=$((missing+1))
      CUDA_VISIBLE_DEVICES="" nice -n 5 python tools/rl_baseline_cell.py "$d" "$o" \
        "$BASE_GAMES" "$BASE_DIR/${d}__${o}.json" > "$BASE_DIR/${d}__${o}.log" 2>&1 &
    done
  done
  if [ "$missing" -gt 0 ]; then
    echo "computing $missing engine_v2 baseline cells ($BASE_GAMES games each, CPU)..."
    wait
  fi
  echo "baseline: $(ls "$BASE_DIR"/*.json 2>/dev/null | wc -l) cells in $BASE_DIR"
}

GLOSS=$(python -c 'import sys;sys.path.insert(0,"tools");import rl_config;print(rl_config.PROMPT_FMT["glossary"])')
DMODE=$(python -c 'import sys;sys.path.insert(0,"tools");import rl_config;print(rl_config.PROMPT_FMT["deck_mode"])')
DSHUF=$(python -c 'import sys;sys.path.insert(0,"tools");import rl_config;print(1 if rl_config.PROMPT_FMT["deck_shuffle"] else 0)')

# Held-out gate: (LM - engine_v2) on identical cells, as a fraction. ONE PROCESS PER CELL
# (tools/eval_rerank_par.sh) -- the gate is 21 cells and sequentially would cost more than the
# rollout it judges. Prompt format comes from rl_config.PROMPT_FMT, so the gate scores the
# policy exactly the way the rollout built it.
gate () {   # $1 policy dir  $2 games/cell  $3 outdir
  local out="$3"
  rm -rf "$out"
  DECKS="${EVAL_DECKS//,/ }" OPPS="${EVAL_OPPS//,/ }" \
    tools/eval_rerank_par.sh "$out" "$1" torch "" 8 "${2:-150}" 1000000 \
      "$GLOSS" "$DMODE" "$DSHUF" > "$out.log" 2>&1 || true
  python tools/rl_gate_score.py "$out" --baseline "$BASE_DIR" --verbose 2>>"$out.log"
}

POLICY="$SFT_CKPT"

# ---- 0. GATE (cached: a restart skips it once passed) -------------------------------
echo "=== [0] engine_v2 baseline for the eval grid ==="
ensure_baseline
echo "=== [0a] gate: SFT policy vs engine_v2 (DELTA) ==="
if [ -f "$WORK/gate_ok" ]; then
  echo "gate already passed (skip) = $(cat "$WORK/gate_ok")"
else
  D=$(gate "$POLICY" 60 "$WORK/gate0" || echo 0)
  echo "SFT delta vs engine_v2 = ${D}  (0 = parity, the Stage-A target)"
  sed -n "s/^gate: /  /p" "$WORK/gate0.log" | tail -1
  if python -c "import sys;sys.exit(0 if float('${D}' or 0)>=float('${GATE_MIN}') else 1)"; then
    echo "$D" > "$WORK/gate_ok"
  else
    echo "SFT more than ${GATE_MIN} below engine — fix SFT before RL"; exit 3
  fi
fi

# ---- generic stage runner -----------------------------------------------------------
# $1 stage  $2 target  $3 max_rounds
#
# TWO signals, deliberately different in cost and in what they mean:
#   * per round, FREE -- the rollout's own pilot win rate. On-policy and temperature-sampled,
#     so it is NOT the deploy number; it is a trend line that says whether anything is moving.
#   * every EVAL_EVERY rounds -- the held-out gate vs engine_v2 at EVAL_GAMES/cell. This is
#     the deploy metric (argmax, held-out decks) and the only thing the plateau rule reads.
# Judging a plateau on a per-round 540-game eval was the original design and it cannot work:
# the rule's threshold sat below the measurement's own noise.
run_stage () {
  local ST="$1" TGT="$2" MAXR="$3"
  local prev=0 have_prev=0 best=-99 stall=0 WB=""
  # Stage A only: up-weight WINNING decisions from low-winrate pilots. Matched sampling
  # already puts each pilot in a contrastive band (rl_config.MATCH_TARGET_WR); win-boost
  # further lifts the density of positive-play signal for decks that rarely win at all.
  [ "$ST" = "A" ] && WB="--win-boost"
  # Markers are keyed by stage AND target. Keyed by stage alone, Stage C's seven targets
  # shared one C_final.txt: the first wrote it and the other six printed "already complete"
  # without training anything (caught by the 2026-07-28 end-to-end loop test).
  local TAG="$ST${TGT:+_$TGT}"
  # MAXR=0 means "not this run". Return WITHOUT writing the completion marker: writing it
  # would make a later resume skip the stage as already done.
  if [ "$MAXR" -le 0 ]; then echo "$TAG skipped (0 rounds requested)"; return; fi
  if [ -f "$WORK/${TAG}_final.txt" ]; then
    POLICY=$(cat "$WORK/${TAG}_final.txt"); echo "$TAG already complete -> $POLICY"; return
  fi
  for R in $(seq 1 "$MAXR"); do
    local RO="$WORK/${TAG}_r${R}.jsonl.gz" NEW="$WORK/${TAG}_r${R}_policy"
    if [ -d "$NEW" ]; then POLICY="$NEW"; echo "$ST r$R already done, skip"; continue; fi
    echo "--- $ST r$R rollout (temp=$TEMP, matchups<=$MCAP, opponent=engine_v2) ---"
    python tools/rl_rollout.py --stage "$ST" ${TGT:+--target $TGT} \
        --model "$POLICY" --matchups "$MCAP" \
        ${OPP_MODEL:+--opp-model "$OPP_MODEL"} \
        $( [ "$HEUR" != "-1" ] && echo --heuristic-frac "$HEUR" ) \
        --branch-per-game "$BRANCH" --branch-k "$BRANCH_K" --branch-playouts "$BRANCH_PLAY" \
        --temperature "$TEMP" --out "$RO" --seed "$R" 2>&1 | tee "$WORK/${TAG}_r${R}_rollout.log"
    local ROLLWR
    ROLLWR=$(grep -oE 'pilot winrate [0-9.]+%' "$WORK/${TAG}_r${R}_rollout.log" | grep -oE '[0-9.]+' | tail -1)
    echo "$TAG r$R rollout winrate = ${ROLLWR:-?}%" | tee -a "$WORK/${TAG}_trend.txt"
    echo "--- $ST r$R GRPO+RAE+MARS update ---"
    python tools/rl_train.py --rollout "$RO" --model "$POLICY" --out "$NEW" --grad-ckpt $WB \
        --branch-weight "$BRANCH_W" --decision-frac "$DEC_FRAC"
    POLICY="$NEW"
    # held-out gate only every EVAL_EVERY rounds (and always on the last one)
    if [ $((R % EVAL_EVERY)) -ne 0 ] && [ "$R" -ne "$MAXR" ]; then continue; fi
    local EVWR
    EVWR=$(gate "$POLICY" "$EVAL_GAMES" "$WORK/${TAG}_r${R}_gate" || echo 0)
    echo "$TAG r$R GATE delta vs engine_v2 = $EVWR (0 = parity)" | tee -a "$WORK/${TAG}_gates.txt"
    sed -n "s/^gate: /  /p" "$WORK/${TAG}_r${R}_gate.log" | tail -1
    # `have_prev` is a separate flag, NOT a sentinel value of `prev`. The gate is a DELTA vs
    # engine_v2 and is therefore normally NEGATIVE, so the old `prev >= 0` guard (written when
    # the gate was an absolute win rate) could never be true and the plateau rule never fired
    # once -- the loop silently ran every round. Found 2026-07-28 when r6 showed +0.35pt and
    # did not stop.
    if [ "$have_prev" = "1" ]; then
      if [ "$(python -c "print(1 if ($EVWR) > (${best:--99}) + $PLATEAU_PT else 0)")" = "1" ]; then
        stall=0
      else
        stall=$((stall + 1))
        echo "$TAG no progress at r$R (gate $EVWR vs best ${best}), stall $stall/$PATIENCE"
        if [ "$stall" -ge "$PATIENCE" ]; then
          echo "$TAG PLATEAU at r$R ($PATIENCE gates with no gain over the best)"
          break
        fi
      fi
    fi
    # Collapse guard: a single bad gate is noise, but a gate this far under the best seen
    # means the run has turned. Used by the "extend to 18 rounds" path, where nobody is
    # watching -- without it a diverging run would burn every remaining round.
    if [ -n "${RL_ABORT_BELOW_BEST:-}" ] && [ "$have_prev" = "1" ]; then
      if [ "$(python -c "print(1 if (${EVWR:--99}) < (${best:--99}) - $RL_ABORT_BELOW_BEST else 0)")" = "1" ]; then
        echo "$TAG ABORT at r$R: gate $EVWR is more than $RL_ABORT_BELOW_BEST below the best ($best)"
        break
      fi
    fi
    best=$(python -c "print(max(${best:--99}, ${EVWR:--99}))")
    prev=$EVWR; have_prev=1
  done
  echo "$POLICY" > "$WORK/${TAG}_final.txt"
}

echo "=== [A] broad climb (P = all decks headroom-weighted, O = engine_v2) ==="
run_stage A "" "${RL_ROUNDS_A:-12}"
echo "=== [B] meta realignment (O reweighted to the live meta) ==="
run_stage B "" "${RL_ROUNDS_B:-10}"
for T in $(python -c 'import sys;sys.path.insert(0,"tools");import rl_config;print(" ".join(rl_config.STAGE_C_TARGETS))'); do
  echo "=== [C] targeted specialisation: $T ==="
  run_stage C "$T" "${RL_ROUNDS_C:-8}"
done
echo "=== RL_LOOP_DONE $(date -u) ==="
echo "final policy: $POLICY"
