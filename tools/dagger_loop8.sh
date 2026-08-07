#!/bin/bash
# loop7, moved onto the DeBERTa-v3-base backbone and the v41 prompt. The loop body is unchanged
# -- only the five things that would have been silently wrong are.
#
# WHY DEBERTA. The backbone bench ran both candidates on the SAME v41 data, the SAME 5-hour
# deadline and the SAME max-len 512, then screened them over 63 decks in mirror mode:
#
#     ref l6_r8 (v40-trained, read under v41)   36.7%
#     deberta-v41                               30.6%    paired vs ref  -6.11pt +- 1.82
#     gte-v41                                   23.0%    paired vs ref -13.93pt +- 1.78
#                                                        paired vs deberta -7.82pt +- 1.83  t -4.27
#
# The confound flagged before the run resolved the WRONG WAY for gte: it is the slower model that
# won. deberta saw 320,399 samples, gte saw 503,568 -- 1.57x more data for 7.8pt less play. So
# the sample-count excuse is not available and the difference is the backbone.
#
# Held-out says the opposite and is wrong. gte finished at loss 1.233 / top1 48.5% against
# deberta's 1.245 / 47.0%, i.e. it fits the imitation target BETTER and plays 7.8pt WORSE. Same
# lesson as `teacher-9b-adds-nothing`: imitation accuracy does not predict ranking ability, so
# this loop is steered by the screen and never by eval loss.
#
# BOTH v41 arms sit far below l6_r8, and that is not evidence against v41. l6_r8 is eight
# warm-started rounds deep; five hours from scratch is not the same quantity. That is exactly why
# this is a LOOP starting from /root/out/v41_deberta rather than another one-shot training.
#
# THE FIVE CHANGES, each of which fails silently if skipped:
#
#   1. BASE is v41_base.jsonl.gz. Pointing at v40_base trains a v41 model on v40 prompts; the
#      renderer would not complain, and the screen -- which renders v41 -- would just score low.
#   2. MAXLEN 512. NOT because longer errors -- that was measured wrong. deberta-v3 has
#      position_biased_input=False (no absolute position table at all) and log-bucketed relative
#      attention; a 1024-token forward runs fine. 512 stays because the v41 prompt FITS it:
#      measured over 700 rows across all 63 decks, pair tokens p50 352 / p99 492, over-512 rate
#      0.14% (max overshoot 4 tokens). A larger window would buy nothing and cost O(L^2).
#   3. OUTSTEM. loop7 wrote /root/out/l6_r$ROUND. l6_r8 is the REFERENCE checkpoint this whole
#      comparison is measured against, and round 8 here would have overwritten it.
#   4. VALUED_FRAC defaults to 0. The playout-valued attach files are v40-rendered
#      (v40_attach_q1/q2), so folding them in at 5% would put 5% of every round in the other
#      prompt format -- the exact mixture `prune_pool_fmt.py` exists to prevent. They are
#      regenerated separately in v41; set VALUED_FRAC=0.05 and point VALUED at the new files
#      once they exist.
#   5. DUAL_FIRST=0. That switch re-screens the previous checkpoint in mirror mode, for the one
#      round that crosses from stock to mirror screening. This loop is mirror from round 1, and
#      leaving it on would send round 2 hunting for a nonexistent round-0 checkpoint and drop
#      the paired line.
#
# Round 1's screen is seeded from the bench's deberta_v41.json: same model, same 40 games, same
# --mirror --seed 1, same libcg_mirror.so, same 4 shards. Re-running it would burn an hour to
# reproduce a file we already have.
#
# ------------------------------------------------------------------------------------------
# Original loop7 header follows.
#
# screen -> collect DAgger on what it loses -> CONTINUE TRAINING the same model -> screen again.
#
# THE CHANGE THAT MATTERS. loop3 and loop4 trained every round from
# `Alibaba-NLP/gte-reranker-modernbert-base`, from scratch. That is not a bug on its own -- it is
# textbook DAgger, and the design note said so: "From-base is the safe default and loses nothing
# -- the accumulated DAgger data carries the progress, not the weights."
#
# But loop3 also stopped accumulating the DAgger. From-scratch training AND no data aggregation
# removes the only thing carrying progress while keeping the design that depends on it, so every
# round was an independent draw with no path to compounding. The evidence, four rounds:
#
#     v40    38.6%     l3_r1  39.1%     l3_r2  36.3%     l4_r3  35.7%
#
# l3_r1 (144,180 rows) and l4_r3 (150,000 rows) are the same recipe at the same size and land
# 3.4pt apart, so most of that spread is retraining variance -- larger than the paired screen's
# 1.25pt SE, which measures deck noise and not the re-draw. A loop like that cannot detect a 3pt
# improvement without repeating each recipe several times.
#
# So: progress lives in the WEIGHTS now. Each round continues the previous checkpoint
# (train_rerank --resume) and still collects fresh DAgger against the model that will consume it.
# This is what instance2's decoder does -- the only track that has produced a clear win.
#
# LR DROPS TO 1e-5 for a continued round. At 2e-5 over ~500,000 samples a warm start would be
# washed out and the round would be a retrain in all but name.
#
# The cost, from the note that first proposed continuing: the model becomes a function of round
# ORDER rather than of the data, which makes attribution harder. That is accepted deliberately.
#
# 1. THE TARGET LADDER IS BACK TO WORSE-ONLY. loop4 added a floor of 8 targets because loop3's
#    round 2 trained on ns_zoroark alone and the fleet lost 2.75pt. That evidence was CONFOUNDED:
#    the same round retrained from scratch on a 67,220-row mix, and the 8-target round that
#    replaced it lost 3.37pt -- more, not less. Widening was never shown to help. With the loop
#    now continuing the weights, narrow targeting deserves to be judged on its own, so the ladder
#    reverts to loop3's: WORSE first, and only if that tier is empty fall through to below45,
#    below50, weakest. MIN_TARGETS survives as an opt-in floor, defaulting to 0 (off).
#
#    THE SIDE EFFECT, named rather than discovered later: fewer targets means fewer collected
#    records, and the round is now 1.2M rows instead of 150k, so the same collection is 8x more
#    diluted. 8 decks x 72 games produced 17,929 rows = 1.60% of the mix; one deck would give
#    ~0.2%. COLLECT_GAMES therefore scales INVERSELY with the target count, so the collection is
#    about the same size whether the screen names one deck or eight.
#
# 2. A FIXED ROUND SIZE (--total). loop3 set the mix size to nd/RATIO, so the amount the round
#    trained on was a function of how many decks were bad enough to collect from. As the loop
#    succeeded it starved itself:
#
#        round 1   3 targets   7,209 DAgger rows   ->  144,180-row mix   3.6 epochs
#        round 2   1 target    3,361 DAgger rows   ->   67,220-row mix   7.8 epochs
#
#    Round 2 also carried only 3,361 of the 21,600 playout-valued attach records -- the data that
#    produced the v40 win -- because the valued share is a fraction of that same shrinking total.
#    With --total the round size is constant and the DAgger is capped at its 5% share.
#
# loop3's own change, kept -- EACH ROUND USES ONLY ITS OWN DAgger. dagger_loop2 accumulated every
# round's collection, and that is where the run went backwards -- paired on the same 63 decks:
#
#     base   -> round 1 out   +3.06pt +- 1.15   0 decks became WORSE, 16 left   exact p 0.000
#     round1 -> round 2 out   -4.25pt +- 1.62  16 decks became WORSE,  3 left   exact p 0.004
#     base   -> round 2 out   -1.19pt +- 1.21   indistinguishable from the base  p 0.581
#
# Two rounds netted nothing. Round 1 trained on one collection; round 2 added a second, gathered
# by a pilot that was itself weak. The accumulation is not PROVEN to be the cause -- that round
# also re-deduped its first collection and drew a different base sample -- but it is the one
# thing the loop does that gets monotonically worse, and dropping it costs nothing to test.
#
# The valued-attach files are NOT DAgger and are NOT dropped: they are playout-measured labels
# that do not depend on which pilot collected them, so they stay in every round.
#
# RATIO is 0.05, down from 0.10, per instruction.
set -u
REPO=/root/ptcg/repo
KIND=${KIND:-deberta41}
MODEL=${MODEL:?set MODEL to the checkpoint this loop starts from}
# Where each round's checkpoint is written. MUST NOT be /root/out/l6_r* -- see change 3.
OUTSTEM=${OUTSTEM:-/root/out/d41_r}
# Prefer the pilot-11 pool: same rows the mix filter would keep, minus scanning 42.5M rows to
# find them (the reservoir reads the WHOLE base file every round). Falls back to the full pool
# + filter when the 11-file does not exist yet.
BASE=${BASE:-$([ -s /root/ptcg/repo/data/rerank/v41_base11.jsonl.gz ] \
  && echo /root/ptcg/repo/data/rerank/v41_base11.jsonl.gz \
  || echo /root/ptcg/repo/data/rerank/v41_base.jsonl.gz)}
VALUED=${VALUED:-$REPO/data/rerank/v40_attach_q1.jsonl.gz,$REPO/data/rerank/v40_attach_q2.jsonl.gz}
RATIO=${RATIO:-0.05}
# 0 until the attach files are re-rendered in v41; mix_v40.py still needs --valued to point at
# readable files, and at frac 0 it copies none of them.
VALUED_FRAC=${VALUED_FRAC:-0}
MAXLEN=${MAXLEN:-512}
SCREEN_GAMES=${SCREEN_GAMES:-40}
# 8, not 4. The screen is bound by its slowest shard and the shards were dealt alphabetically:
# round 4 came in at 4478 / 2225 / 2454 / 2381 s. With cost-balanced dealing (below) 8 shards
# put the predicted wall at ~1279 s against 4478 today. Past ~8 the single longest deck is the
# floor, so more shards buy nothing -- that is what --deck-seconds is for.
SHARDS=${SHARDS:-8}
# --- seeded round (loop7) -----------------------------------------------------------------
# Screen with the same shuffle for both seats and the SAME seeds every round, so a
# round-over-round difference is a policy difference. 63 decks: paired SE 1.58x tighter, and
# re-scoring one checkpoint moves 0.00pt instead of 2.6pt. mirror `p` != stock `p`, so the first
# seeded round re-screens the PREVIOUS checkpoint in mirror mode to keep the paired line valid.
MIRROR_SCREEN=${MIRROR_SCREEN:-1}
# Bound the per-deck tail. mega_venusaur spent 2071 s (18% of the whole screen) to return 2-7
# with 31 draws in 40 games; draws carry no information for the SPRT. 600 s is ~4x the median.
DECK_SECONDS=${DECK_SECONDS:-600}
SCREEN_SEED=${SCREEN_SEED:-1}
# 0, not 1: this loop's round-1 screen is already a mirror screen (see the header).
DUAL_FIRST=${DUAL_FIRST:-0}
# A FIXED panel, played every round regardless of which decks the tier targeted, with seeds
# keyed by the deck NAME. Both matter: keying by the deck's index in the target list made the
# same deck draw different games whenever the targets changed, and taking the panel from the
# targets meant a deck leaving the tier stopped being measured.
ANCHOR_PANEL=${ANCHOR_PANEL:-crustle_stall,alakazam,dragapult,mega_lucario,rockets_honchkrow,ns_zoroark,marnie_grimmsnarl,archaludon,hydrapple,chandelure,trevenant_control,zangoose}
ANCHOR_GAMES=${ANCHOR_GAMES:-8}
SEEDED_COLLECT=${SEEDED_COLLECT:-1}
MIRROR_SO=${MIRROR_SO:-/root/ptcg/repo/data/kaggle_engine_ext/libcg_mirror.so}
# Collection is NOT the expensive part of a round: 3 decks x 72 games took 5 minutes against a
# 30-minute screen and a 5-hour training. So collect wide and let the mixer cap the share.
COLLECT_GAMES=${COLLECT_GAMES:-72}
MAX_TARGETS=${MAX_TARGETS:-20}
MIN_TARGETS=${MIN_TARGETS:-0}
MAX_GAMES=${MAX_GAMES:-400}
# 300k, up from loop7's 150k. v41_base holds 11.45M rows and the starting checkpoint has seen
# 320k of them, so this model is data-starved rather than over-fit: at 150k a five-hour round is
# ~4 epochs over the same slice, at 300k it is ~1.3 epochs over twice as much fresh base.
TOTAL=${TOTAL:-300000}
# ---- the base taper -----------------------------------------------------------------------
# A round is 90% base, 5% DAgger, 5% valued, and it is bound by --deadline-h, not by the data:
# round 3 saw 310,529 rows of a 300,000-row mix in five hours = 1.03 epochs. So five of every
# five and a half hours go to re-delivering base rows the model has already been trained on, to
# carry 15,000 rows of new signal.
#
# The base cannot simply be cut: valued-only training has destroyed a run before, and 85% base
# is the floor that came out of it. So the base TAPERS -- 20,000 rows per round -- and stops at
# the floor. Gradual on purpose: a step change confounds "smaller base helped" with "smaller
# base hurt starting somewhere", and one round per step keeps those separable.
#
#     round   5       6       7       8       9+
#     base  250k    230k    210k    190k    170k   (floor: 170k/200k = 85.0%)
#     total 280k    260k    240k    220k    200k
#     epoch 270m    251m    231m    212m    193m   (at the measured 17.3 rec/s)
#
# DAgger and valued stay at 15,000 rows ABSOLUTE. They are the reason the round exists, and
# mix_v40 takes them as fractions of the total, so the fractions are recomputed each round --
# leaving them at 0.05 would shrink the signal along with the base, which is the one thing this
# change must not do.
DAGGER_N=${DAGGER_N:-15000}
VALUED_N=${VALUED_N:-15000}
BASE_N0=${BASE_N0:-270000}
BASE_STEP=${BASE_STEP:-20000}
BASE_MIN=${BASE_MIN:-170000}     # 170k/(170k+30k) = 85.0%, the documented floor
BASE_FROM=${BASE_FROM:-5}        # first round that tapers
DEADLINE_H=${DEADLINE_H:-24}
MARGIN=${MARGIN:-0.5}
# 2, not 12, from round 6. The flat rounds delivered ~4,200 optimizer updates each (50k
# backwards / accum 12) at lr 1e-5 on a warm-started checkpoint, and the update-starvation
# probe showed the same trainer MOVING (-0.20 train loss over 857 updates, train dipping below
# eval for the first time) once given updates. accum 12->2 is 6x the updates at ~unchanged wall
# clock -- opt.step is cheap next to forward+backward at this size.
ACCUM=${ACCUM:-2}
# The submission is drawn from STAGE_C_TARGETS; training the pilot side 63-wide spends 83% of
# every round on decks that cannot ship. Opponent side stays unrestricted inside the rows.
# Empty = old behaviour.
PILOT_DECKS=${PILOT_DECKS:-$(PYTHONPATH=$REPO/tools python3 -c "import rl_config; print(','.join(rl_config.STAGE_C_TARGETS))" 2>/dev/null || true)}
LR=${LR:-1e-5}
STATE=/root/loop_$KIND
LOG=$STATE/loop.log
mkdir -p "$STATE"
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[d41 $(date -u +%m-%d_%H:%M:%S) r$ROUND] $*"; }

T0=$(date +%s)
ROUND=${START_ROUND:-1}
# Screen one checkpoint on every deck: screen_model <checkpoint> <merged-out> <tag>
screen_model() {
  local SMODEL="$1" SOUT="$2" STAG="$3" j=0
  # SHARD BY MEASURED COST, NOT ALPHABETICALLY. `d[i::n]` over a sorted deck list is blind to how
  # long a deck takes, and the spread is not mild: in round 4 the four shards came in at 4478 /
  # 2225 / 2454 / 2381 s. The wall clock is the slowest shard, so the screen cost 1.24 h to do
  # 3.2 h of work that would have fit in 0.80 h if balanced -- 26 minutes lost to the deal.
  #
  # One deck causes most of it. mega_venusaur took 2071 s against a 153 s median, 7x the next
  # slowest, and returned 2-7 with THIRTY-ONE DRAWS out of 40 games: it runs to the turn cap
  # (retreat ping-pong, [[engine-retreat-pingpong]], which names venusaur) and draws carry no
  # information for the SPRT. It is 18% of the screen's compute for 9 decisive games.
  #
  # Longest-processing-time first: sort by last round's seconds, hand each deck to the shard with
  # the least work so far. LPT is within 4/3 of optimal, and the floor here is the single longest
  # deck -- so raising SHARDS past that point buys nothing, which is why the cap below exists.
  # No history (round 1, or the logs were cleaned) falls back to the old round-robin.
  python3 - "$SHARDS" "$STATE" > $STATE/shards.txt <<'PYX'
import glob, re, sys
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
import library
d = sorted(library.list_decks())
n, state = int(sys.argv[1]), sys.argv[2]
cost = {}
for f in glob.glob(state + "/screen_mirror_r*.log"):
    for line in open(f, errors="ignore"):
        m = re.match(r"(\S+)\s+B \d+-\d+ = .*?(\d+)s$", line.strip())
        if m:                      # later rounds overwrite earlier ones: freshest wins
            cost[m.group(1)] = int(m.group(2))
if not cost:
    for i in range(n):
        print(" ".join("--deck " + x for x in d[i::n]))
    raise SystemExit
med = sorted(cost.values())[len(cost) // 2]
load, bins = [0.0] * n, [[] for _ in range(n)]
for deck in sorted(d, key=lambda k: -cost.get(k, med)):
    i = min(range(n), key=lambda k: load[k])
    bins[i].append(deck); load[i] += cost.get(deck, med)
print("[shard] predicted %.0f-%.0f s across %d shards (was one bin of %.0f)"
      % (min(load), max(load), n, sum(load) / n), file=sys.stderr)
for b in bins:
    print(" ".join("--deck " + x for x in b))
PYX
  local MFLAG=""
  [ "$MIRROR_SCREEN" = 1 ] && MFLAG="--mirror --seed $SCREEN_SEED --mirror-so $MIRROR_SO"
  while read -r DECKS; do
    [ -n "$DECKS" ] || continue
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "hf:$SMODEL" \
        --max-games "$SCREEN_GAMES" --deck-seconds "$DECK_SECONDS" $MFLAG --out "$STATE/${STAG}.$j.json" \
        > "$STATE/screen_${STAG}.$j.log" 2>&1 &
    j=$((j+1))
  done < $STATE/shards.txt
  say "launched $j screen shards for $STAG ($SMODEL)"
  wait
  python3 - "$j" "$SOUT" "$STATE/${STAG}" <<'PYX'
import json, sys
n, out, stem = int(sys.argv[1]), sys.argv[2], sys.argv[3]
d = {}
for k in range(n):
    try:
        d.update(json.load(open("%s.%d.json" % (stem, k)))["decks"])
    except Exception as e:
        print("shard %d unreadable: %s" % (k, e))
if not d:
    raise SystemExit(1)
json.dump({"decks": d}, open(out, "w"))
print("merged -> %d decks" % len(d))
PYX
}

# PREV seeds the paired comparison. loop6 never set it, so the one line that can say whether a
# round moved anything was never printed.
PREV=""
if [ -s "$STATE/mirror_r$((ROUND-1)).mirror.json" ]; then
  PREV=$STATE/mirror_r$((ROUND-1)).mirror.json
elif [ -s "$STATE/mirror_r$((ROUND-1)).json" ]; then
  PREV=$STATE/mirror_r$((ROUND-1)).json
fi
[ -n "$PREV" ] && echo "[d41] seeding the paired comparison from $PREV"

while :; do
  HRS=$(( ($(date +%s) - T0) / 3600 ))
  if [ "$HRS" -ge "$DEADLINE_H" ]; then
    echo "[d41] deadline reached (${HRS}h) -- stopping before round $ROUND"; break
  fi
  say "=== round $ROUND | model $MODEL | ${HRS}h elapsed ==="

  # ---- screen every deck ----------------------------------------------------------------
  MIR=$STATE/mirror_r$ROUND.json
  if [ -s "$MIR" ]; then
    say "reusing existing screen $MIR"
  else
    screen_model "$MODEL" "$MIR" "mirror_r$ROUND" || { say "screen FAILED"; break; }
  fi

  # See i2d: a mirror `p` and a stock `p` are different quantities, so re-screen the checkpoint
  # the PREVIOUS round screened (round R screens what R-1 trained) once, in mirror mode.
  if [ "$MIRROR_SCREEN" = 1 ] && [ "$DUAL_FIRST" = 1 ] && [ -n "$PREV" ]; then
    BASEMIR=$STATE/mirror_r$((ROUND-1)).mirror.json
    BASEMODEL=$OUTSTEM$((ROUND-2))
    if [ -s "$BASEMIR" ]; then
      PREV=$BASEMIR
    elif [ -d "$BASEMODEL" ]; then
      say "re-screening the baseline $BASEMODEL in mirror mode so the paired line stays valid"
      if screen_model "$BASEMODEL" "$BASEMIR" "mirror_r$((ROUND-1))b"; then
        PREV=$BASEMIR
      else
        say "baseline re-screen FAILED -- no paired line this round"; PREV=""
      fi
    else
      say "no baseline checkpoint at $BASEMODEL -- no paired line this round"; PREV=""
    fi
    DUAL_FIRST=0
  fi

  python3 - "$MIR" "$STATE/history.tsv" "$ROUND" "$PREV" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))["decks"]
p = [v["p"] for v in d.values()]
row = (int(sys.argv[3]), len(p), statistics.median(p), sum(p)/len(p),
       sum(1 for v in d.values() if v["verdict"] == "WORSE"), sum(1 for x in p if x < .5))
with open(sys.argv[2], "a") as f:
    f.write("%d\t%d\t%.4f\t%.4f\t%d\t%d\n" % row)
print("[screen] round %d | decks %d | median %.1f%% | mean %.1f%% | WORSE %d | below50 %d"
      % (row[0], row[1], 100*row[2], 100*row[3], row[4], row[5]))
for k in sorted(d, key=lambda k: d[k]["p"])[:8]:
    print("   %-24s %.1f%%  %s" % (k, 100*d[k]["p"], d[k]["verdict"]))
# Paired against the PREVIOUS round on the decks both screened. With --mirror the two rounds
# replay the SAME seeds, so this difference is a policy difference and nothing else.
prev = sys.argv[4] if len(sys.argv) > 4 else ""
if prev:
    import math
    try:
        q = json.load(open(prev))["decks"]
    except Exception as e:
        print("[paired] previous screen unreadable: %s" % e); raise SystemExit(0)
    fp_now = {v.get("shuffle_fp") for v in d.values() if v.get("shuffle_fp")}
    fp_old = {v.get("shuffle_fp") for v in q.values() if v.get("shuffle_fp")}
    if fp_now and fp_old and fp_now != fp_old:
        print("[paired] REFUSING: shuffle fingerprint changed %s -> %s. The two rounds did not "
              "play the same games, so the paired difference is not a policy difference."
              % (sorted(fp_old), sorted(fp_now)))
        raise SystemExit(0)
    both = sorted(set(d) & set(q))
    if len(both) > 2:
        diff = [d[k]["p"] - q[k]["p"] for k in both]
        m = sum(diff) / len(diff)
        se = statistics.stdev(diff) / math.sqrt(len(diff))
        print("[paired vs previous round, %d decks] %+.4f +- %.4f  t %+.2f"
              % (len(both), m, se, m / se if se else 0.0))
        srt = sorted(both, key=lambda k: d[k]["p"] - q[k]["p"])
        print("  drops: " + ", ".join("%s %+.0fpt" % (k, 100*(d[k]["p"]-q[k]["p"])) for k in srt[:5]))
        print("  gains: " + ", ".join("%s %+.0fpt" % (k, 100*(d[k]["p"]-q[k]["p"])) for k in srt[-5:]))

PY

  # ---- targets --------------------------------------------------------------------------
  # WORSE only. The ladder falls through to a wider tier ONLY when WORSE is empty -- a round with
  # no failing deck still has to train on something. MIN_TARGETS defaults to 0 and is an opt-in
  # floor, not the behaviour.
  #
  # THE VERDICT IS RECOMPUTED FROM THE RAW GAME RECORD, NOT READ FROM THE FILE. In mirror mode
  # mirror_match runs the SPRT on DISCORDANT PAIRS -- 11 to 13 per deck against 40 games -- and
  # at that sample size the non-inferiority boundary is unreachable, so every deck comes back
  # "undecided" and the stored `verdict` is never WORSE. This loop has screened in mirror mode
  # since round 1, so the WORSE tier was ALWAYS empty and the ladder fell through to below45
  # every single round. Recomputed on w/l, deberta r1 has 5 WORSE decks and r3 has 7.
  #
  # That tier is not decoration: it is the one mechanism in this loop whose effect has ever been
  # seen. gte went WORSE 4 -> 0 across r2->r3 while its mean stayed at 40.2 -> 40.1 -- the bottom
  # tail moved and the average did not, which is exactly what targeting the tail should look like
  # and exactly what a mean-only readout hides.
  #
  # Only the TARGETING changes. The screen stays mirror, because the paired round-over-round
  # comparison is what the stop rule reads and mirror is what makes its null exactly 0
  # ([[mirror-shuffle-mode]]). w/l/d/p keep their raw meaning in mirror rounds by design.
  TARGETS=$(python3 -c "
import json, sys
sys.path.insert(0, '$REPO/tools')
from mirror_match import sprt          # stdlib-only at module level; no PYTHONPATH needed
d = json.load(open('$MIR'))['decks']
pilot = set('$PILOT_DECKS'.split(',')) - {''}
if pilot:
    d = {k: v for k, v in d.items() if k in pilot}   # only shippable decks are DAgger targets
by_p = sorted(d, key=lambda k: d[k]['p'])
def _worse(v):
    return sprt(v['w'], v['l'], 0.50, 0.55, 0.05, 0.05, 0.05)[2] == 'WORSE'
worse = [k for k in by_p if _worse(d[k])]
if worse:
    tier, pick = 'WORSE', list(worse)
else:
    for tier, ks in [('below45', [k for k in by_p if d[k]['p'] < 0.45]),
                     ('below50', [k for k in by_p if d[k]['p'] < 0.50]),
                     ('weakest', by_p)]:
        if ks:
            pick = list(ks); break
while len(pick) < $MIN_TARGETS:
    nxt = next((k for k in by_p if k not in pick), None)
    if nxt is None: break
    pick.append(nxt)
pick = sorted(pick, key=lambda k: d[k]['p'])[:$MAX_TARGETS]
print('TIER=%s WORSE=%d -> targets=%d' % (tier, len(worse), len(pick)), file=sys.stderr)
print(','.join(pick))
")
  say "targets: $TARGETS"
  [ -n "$TARGETS" ] || { say "no targets -- stopping"; break; }

  # ---- collect, THIS ROUND ONLY ---------------------------------------------------------
  # Games per deck scale inversely with the target count so the ROUND's collection is about the
  # same size either way. Measured: 8 decks x 72 games -> 17,929 records (~2,240 per deck), which
  # is 1.60% of a 1.2M-row mix. One deck at 72 games would be 0.2% and effectively absent.
  NT=$(echo "$TARGETS" | tr ',' '\n' | grep -c .)
  GAMES=$(( COLLECT_GAMES * 8 / NT ))
  [ "$GAMES" -gt "$MAX_GAMES" ] && GAMES=$MAX_GAMES
  [ "$GAMES" -lt "$COLLECT_GAMES" ] && GAMES=$COLLECT_GAMES
  say "$NT target(s) -> $GAMES games each"
  DAG=$STATE/dagger_r$ROUND.jsonl.gz
  # One process here (not sharded), so the base only has to differ per ROUND. collect_dagger
  # uses base + deck_index*1000 + game; 20 decks x 1000 + 400 games stays inside the 100,000 stride.
  SBASE=0
  [ "$SEEDED_COLLECT" = 1 ] && SBASE=$(( 100000 + ROUND * 100000 ))
  PYTHONPATH=cg-lib python3 tools/collect_dagger.py --decks "$TARGETS" \
      --model "hf:$MODEL" --games "$GAMES" --out "$DAG" \
      --engine-seed-base "$SBASE" --mirror-so "$MIRROR_SO" \
      --anchor-decks "$ANCHOR_PANEL" --anchor-games "$ANCHOR_GAMES" \
      || { say "collect FAILED -- stopping"; break; }

  MIX=$REPO/data/rerank/${KIND}_r$ROUND.jsonl.gz

  # Taper the base for this round, holding the DAgger and valued counts fixed.
  RSTEPS=0
  [ "$ROUND" -ge "$BASE_FROM" ] && RSTEPS=$(( ROUND - BASE_FROM + 1 ))
  BASE_N=$(( BASE_N0 - BASE_STEP * RSTEPS ))
  [ "$BASE_N" -lt "$BASE_MIN" ] && BASE_N=$BASE_MIN
  if [ "$RSTEPS" -gt 0 ]; then
    RTOTAL=$(( BASE_N + DAGGER_N + VALUED_N ))
    RRATIO=$(python3 -c "print('%.6f' % ($DAGGER_N / $RTOTAL))")
    # VALUED_FRAC 0 means the valued files are absent or in the wrong prompt format; the taper
    # must not switch them on behind that decision.
    RVFRAC=0
    [ "$(python3 -c "print(1 if float('$VALUED_FRAC') > 0 else 0)")" = 1 ] \
      && RVFRAC=$(python3 -c "print('%.6f' % ($VALUED_N / $RTOTAL))")
    say "base taper step $RSTEPS: base $BASE_N | total $RTOTAL | base share $(python3 -c "print('%.1f%%' % (100*$BASE_N/$RTOTAL))")"
  else
    RTOTAL=$TOTAL; RRATIO=$RATIO; RVFRAC=$VALUED_FRAC
  fi

  # A DIFFERENT base sample every round. The reservoir is seeded, so without this every round
  # trains on the same ~12% slice of the 2.87M pool and the rounds differ only by their DAgger --
  # the base contribution would be a constant, not a sample.
  python3 tools/mix_v40.py --base "$BASE" --dagger "$DAG" --valued "$VALUED" \
      --dagger-frac "$RRATIO" --valued-frac "$RVFRAC" --total "$RTOTAL" \
      --pilot-decks "$PILOT_DECKS" \
      --seed "$ROUND" --out "$MIX" \
      || { say "mix FAILED -- stopping"; break; }

  # ---- CONTINUE from the current model ---------------------------------------------------
  # --max-samples IS THE ROUND LENGTH NOW, and that is what makes the taper mean anything. It
  # used to be 600,000 against a 300,000-row mix, so the round always ran to --deadline-h 5 and a
  # smaller mix would have bought more EPOCHS, not a shorter round -- the opposite of the point.
  # At $RTOTAL the round is exactly one epoch and the deadline is a guard. Neutral at today's
  # size: round 3 stopped at 310,529 rows of 300,000 when the clock ran out, so one epoch is
  # what it was already doing.
  # train_rerank --resume reads the model out of --out, so the checkpoint being continued is
  # copied in first. rr_progress.json must NOT come with it: it records how many samples the
  # PREVIOUS round saw and would fast-forward past this round's entire mix.
  OUT=$OUTSTEM$ROUND
  rm -rf "$OUT"; mkdir -p "$OUT"
  cp -r "$MODEL"/. "$OUT"/ || { say "could not seed $OUT from $MODEL"; break; }
  rm -f "$OUT/rr_progress.json"
  [ -f "$OUT/config.json" ] && [ -f "$OUT/model.safetensors" ] \
      || { say "STOP: $OUT is not a usable checkpoint to continue from"; break; }
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 tools/train_rerank.py --data "$MIX" --out "$OUT" --resume \
      --deadline-h 5 --max-samples "$RTOTAL" --lr "$LR" --pair-batch 32 --accum "$ACCUM" --max-len "$MAXLEN" \
      --eval-n 2000 --grad-ckpt --margin-weight "$MARGIN" \
      || { say "train FAILED -- stopping"; break; }
  [ -f "$OUT/model.safetensors" ] || { say "no model saved -- stopping"; break; }
  # A --resume that silently fell through to the base model trains for five hours and looks
  # normal. instance2 lost 10 GPU-hours to exactly this class of failure.
  # $OUT carries the round number, so this line is unique to this round inside the shared log.
  grep -q "RESUME from $OUT" "$LOG" \
      || { say "STOP: the run did not report RESUME -- it trained from base"; break; }

  say "round $ROUND done -> $OUT"
  # ADVANCE THE PAIRED BASELINE. Without this line PREV keeps whatever it was seeded with
  # before the loop, so every "[paired vs previous round]" is really "vs that one old
  # checkpoint" -- and it inherits a constant offset from however that screen happened to
  # score. loop7 rounds 5-8 printed +1.2 to +2.2pt that way while the true round-over-round
  # deltas were -0.99 to +0.79 and the 4-round total was -0.71pt. dagger_loop_i2d.sh has
  # always had it; loop7 did not.
  PREV=$MIR
  MODEL=$OUT
  ROUND=$((ROUND+1))
  rm -f "$MIX"        # 200-400 MB each; the dagger file is what matters and it is kept
done
echo "[d41] LOOP ENDED"
