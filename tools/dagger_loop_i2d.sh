#!/bin/bash
# instance2's DAgger loop: screen 63 decks -> collect on the ones it loses -> retrain -> repeat.
#
# The decoder twin of tools/dagger_loop3.sh, with the two decisions that loop settled:
#   * EACH ROUND USES ONLY ITS OWN DAgger. Accumulating every round's collection is what ran the
#     reranker loop backwards (+3.06pt, then -4.25pt with 16 decks turning WORSE).
#   * THE BASE IS RESAMPLED EVERY ROUND (mix_sft_round.py --seed $ROUND). instance2 previously
#     re-used one frozen 193,919-row file, so the base was a constant, not a sample.
#
# The base pool itself is new: data/sft/v40_base_sft.jsonl.gz, 5,733,620 rows converted from
# instance1's reranker pool by tools/rerank_to_sft.py. Besides being 30x the old file, it is the
# first instance2 base rendered with menu_dedup -- the old one showed one entry per menu POSITION
# while rl_config.PROMPT_FMT (what mirror_match/collect_dagger actually render at inference) shows
# one per ACT, so every base row was training on a format the model never meets in play.
#
# THREE shards, not four: four 4B scorers ask for 48 GB on a 47.4 GB card and one dies of CUDA
# OOM, which is how the scheme-A baseline lost 16 decks. The tuned scorer reports 11.56 GiB
# RESERVED per process, which looks like four would fit -- but that figure excludes the ~0.5 GiB
# CUDA context each process also holds, and 4 x ~12.1 GiB is 48.4 GiB. Still three.
#
# BATCH 32 x ACCUM 1, kept from i2b but NOT for the reason it was adopted. An A100 sweep said
# +10.4% over 8 x 4; the two i2b rounds differ in nothing else and say otherwise:
#
#     round 1   bsz  8 accum 4   3.221 s/it  (32 rows/it)
#     round 2   bsz 32 accum 1   3.214 s/it              +0.2%, i.e. nothing
#
# The 5880 Ada is already saturated at bsz 8 on what binds here (the 154,733-wide logits), so a
# wider batch buys nothing. It costs nothing either -- 15.7 GiB of 48, the OOM fallback below has
# never fired -- so it stays. Do not count it as a speedup, and do not re-run that sweep on rented
# hardware expecting the answer to transfer.
#
# WHAT WAS TESTED AND REJECTED at inference, same session: FP8 (torchao) and torch.compile made
# scoring 14x and 23x SLOWER (902 ms and 1515 ms per decision against a 65.8 ms baseline).
#
# WHAT CHANGED FROM i2b -- THE COLLECTION NO LONGER STARVES AS THE PILOT IMPROVES.
# i2b collected a fixed 48 games per target. Round 1 got 49,532 rows (18.4% of the mix) and moved
# the screen +4.17pt; round 2 got 12,501 (5.3%). The cause is not the target COUNT -- both rounds
# had 9 -- it is that DAgger yield tracks the LM's error rate:
#
#     r1 shard2  mega_venusaur/dragapult/raging_bolt  60,828 decisions  wrong 58.7%  42,050 rows
#     r2 shard0  mega_feraligatr/hydreigon/manectric  10,114 decisions  wrong 30.0%   4,793 rows
#
# 85% of round 1's collection came from three decks the pilot was falling apart on; when the round
# fixed them, the next round's targets were merely mediocre and yielded a quarter as much. So the
# loop feeds itself least exactly when it starts working, and a round drifts towards being a
# re-run of the base SFT with a 5% garnish. Same shape as tools/dagger_loop6.sh's problem, but the
# fix there (scale games by target COUNT) is inert here because the count is pinned at 9.
#
# COLLECT UNTIL THE ROW TARGET IS MET, up to MAX_PASSES passes with different seeds. Targeting the
# quantity that actually matters beats guessing a games number: an easy round stops after one pass
# and costs nothing, a thin one keeps going. DAG_MIN is set at round 1's yield, the only round
# whose effect on the screen has been measured.
set -u
REPO=/root/ptcg/repo
MODEL=${MODEL:?set MODEL to the checkpoint this loop starts from}
BASEQ=${BASEQ:-unsloth/Qwen3-4B-Base}
BASE=${BASE:-$REPO/data/sft/v40_base_sft.jsonl.gz}
BASE_N=${BASE_N:-200000}
VALUED=${VALUED:-$REPO/data/sft/valued_q1.jsonl.gz,$REPO/data/sft/valued_q234.jsonl.gz}
VOCAB=${VOCAB:-$REPO/data/cardfirst_b_v39.json}
SCREEN_GAMES=${SCREEN_GAMES:-40}
COLLECT_GAMES=${COLLECT_GAMES:-48}
DAG_MIN=${DAG_MIN:-40000}
MAX_PASSES=${MAX_PASSES:-4}
MAX_TARGETS=${MAX_TARGETS:-9}
SHARDS=${SHARDS:-3}
# --- seeded round (i2d) -------------------------------------------------------------------
# MIRROR_SCREEN=1 screens with tools/mirror_match.py --mirror: identical decklist AND identical
# shuffle for both seats, and the same SCREEN_SEED every round, so a round-over-round difference
# is a policy difference and nothing else. Measured on 63 decks: the paired SE tightens 1.58x
# (= the same precision for 0.40x the games), and re-scoring one checkpoint moves 0.00pt instead
# of 2.6pt. NOTE mirror `p` and stock `p` are different quantities, so the first seeded round
# also runs a STOCK screen purely to keep its paired-vs-previous line comparable (DUAL_FIRST).
MIRROR_SCREEN=${MIRROR_SCREEN:-1}
SCREEN_SEED=${SCREEN_SEED:-1}
DUAL_FIRST=${DUAL_FIRST:-1}
# Anchored collection: this fraction of each deck's games replays FIXED seeds every round, so
# `[anchor] LM wrong x%` is a paired error rate on identical openings. It resolves in decisions
# (tens of thousands) instead of games (tens) -- a fast read on whether a 6.5h train did
# anything. It measures agreement with engine_v2, NOT win rate: leading indicator, not gate.
ANCHOR_PANEL=${ANCHOR_PANEL:-crustle_stall,alakazam,dragapult,mega_lucario,rockets_honchkrow,ns_zoroark,marnie_grimmsnarl,archaludon,hydrapple,chandelure,trevenant_control,zangoose}
ANCHOR_GAMES=${ANCHOR_GAMES:-8}
SEEDED_COLLECT=${SEEDED_COLLECT:-1}
MIRROR_SO=${MIRROR_SO:-/root/ptcg/repo/data/kaggle_engine_ext/libcg_mirror.so}
DEADLINE_H=${DEADLINE_H:-72}
STATE=/root/loop_i2
LOG=$STATE/loop.log
mkdir -p "$STATE"
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[i2l $(date -u +%m-%d_%H:%M:%S) r$ROUND] $*"; }

T0=$(date +%s)
ROUND=${START_ROUND:-1}

# A loop restarted at START_ROUND=N used to begin with PREV empty, so its first round printed no
# paired line -- and the paired line is the only statement in this script that can tell whether a
# round moved anything (the medians swing ~9pt on 40 games). round 1 -> 2 had to be differenced by
# hand for exactly this reason. Seed it from the previous round's screen when one is on disk.
PREV=""
if [ -s "$STATE/mirror_r$((ROUND-1)).json" ]; then
  PREV=$STATE/mirror_r$((ROUND-1)).json
  echo "[i2l] seeding the paired comparison from $PREV"
fi
# Screen one checkpoint on every deck: screen_model <checkpoint> <merged-out> <tag>
# Factored out because the first seeded round has to screen TWO checkpoints (see BASELINE below).
screen_model() {
  local SMODEL="$1" SOUT="$2" STAG="$3" j=0
  python3 - "$SHARDS" > $STATE/shards.txt <<'PYX'
import sys
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
import library
d = sorted(library.list_decks())
n = int(sys.argv[1])
for i in range(n):
    print(" ".join("--deck " + x for x in d[i::n]))
PYX
  local MFLAG=""
  [ "$MIRROR_SCREEN" = 1 ] && MFLAG="--mirror --seed $SCREEN_SEED --mirror-so $MIRROR_SO"
  while read -r DECKS; do
    [ -n "$DECKS" ] || continue
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DECKS --a engine --b "qwen:$SMODEL" \
        --max-games "$SCREEN_GAMES" $MFLAG --out "$STATE/${STAG}.$j.json" \
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

while :; do
  HRS=$(( ($(date +%s) - T0) / 3600 ))
  if [ "$HRS" -ge "$DEADLINE_H" ]; then
    echo "[i2l] deadline reached (${HRS}h) -- stopping before round $ROUND"; break
  fi
  say "=== round $ROUND | model $MODEL | ${HRS}h elapsed ==="

  # ---- screen every deck ----------------------------------------------------------------
  # Round 1 is pre-seeded from the finished v40 screen (/root/mirror_i2v40.json): it screened
  # exactly this checkpoint on all 63 decks, so re-running it would burn 5.7 h to reproduce it.
  MIR=$STATE/mirror_r$ROUND.json
  if [ -s "$MIR" ]; then
    say "reusing existing screen $MIR"
  else
    screen_model "$MODEL" "$MIR" "mirror_r$ROUND" || { say "screen FAILED"; break; }
  fi

  # BASELINE. A mirror `p` and a stock `p` are different quantities, so the first seeded round
  # cannot be compared against a screen the previous loop produced. Rather than lose the paired
  # line -- the only line that says whether a round moved anything -- re-screen the PREVIOUS
  # checkpoint in mirror mode once. Costs one extra screen, and from then on every comparison is
  # mirror-to-mirror. Round R screens the checkpoint round R-1 trained, so the checkpoint the
  # previous round screened is i2_r$((ROUND-2)).
  if [ "$MIRROR_SCREEN" = 1 ] && [ "$DUAL_FIRST" = 1 ] && [ -n "$PREV" ]; then
    BASEMIR=$STATE/mirror_r$((ROUND-1)).mirror.json
    BASEMODEL=/root/out/i2_r$((ROUND-2))
    if [ -s "$BASEMIR" ]; then
      PREV=$BASEMIR
    elif [ -d "$BASEMODEL" ]; then
      say "re-screening the baseline $BASEMODEL in mirror mode so the paired line stays valid"
      if screen_model "$BASEMODEL" "$BASEMIR" "mirror_r$((ROUND-1))b"; then
        PREV=$BASEMIR
      else
        say "baseline re-screen FAILED -- dropping the paired line for this round"
        PREV=""
      fi
    else
      say "no baseline checkpoint at $BASEMODEL -- no paired line this round"
      PREV=""
    fi
    DUAL_FIRST=0
  fi

  # Paired against the PREVIOUS round on the decks both screened. The medians in history.tsv
  # swing ~9pt on this sample size; the paired difference resolves 1.2-1.6pt, so it is the only
  # line here that can say whether a round moved anything.
  python3 - "$MIR" "$STATE/history.tsv" "$ROUND" "$PREV" <<'PY'
import json, math, statistics, sys
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
prev = sys.argv[4] if len(sys.argv) > 4 else ""
if prev:
    try:
        q = json.load(open(prev))["decks"]
    except Exception as e:
        print("[paired] previous screen unreadable: %s" % e); raise SystemExit(0)
    # If the two rounds did not play the same games, the paired difference is not a policy
    # difference. Compare the seed -> deck-order fingerprint, not the .so's sha256: the binary
    # hashes differently on each machine while dealing identical games.
    fp_now = {v.get("shuffle_fp") for v in d.values() if v.get("shuffle_fp")}
    fp_old = {v.get("shuffle_fp") for v in q.values() if v.get("shuffle_fp")}
    if fp_now and fp_old and fp_now != fp_old:
        print("[paired] REFUSING: shuffle fingerprint changed %s -> %s"
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
  # The Qwen screen resolves to `undecided` on every deck (SPRT never concludes at 40 games on a
  # pilot this close to the margin), so the WORSE tier is normally empty and the ladder falls
  # through to below45 -- that is expected here, not a failure.
  TARGETS=$(python3 -c "
import json, sys
d = json.load(open('$MIR'))['decks']
for name, ks in [('WORSE',   [k for k,v in d.items() if v['verdict']=='WORSE']),
                 ('below45', [k for k,v in d.items() if v['p']<0.45]),
                 ('below50', [k for k,v in d.items() if v['p']<0.50]),
                 ('weakest', sorted(d, key=lambda k: d[k]['p']))]:
    if ks:
        print('TIER=%s (%d decks)' % (name, len(ks)), file=sys.stderr)
        print(','.join(sorted(ks, key=lambda k: d[k]['p'])[:$MAX_TARGETS])); break
")
  say "targets: $TARGETS"
  [ -n "$TARGETS" ] || { say "no targets -- stopping"; break; }

  # ---- collect, THIS ROUND ONLY, sharded ------------------------------------------------
  DAG=$STATE/dagger_r$ROUND.jsonl.gz
  if [ -s "$DAG" ]; then
    say "reusing existing collection $DAG"
  else
    python3 - "$TARGETS" "$SHARDS" > $STATE/cshards.txt <<'PY'
import sys
t = [x for x in sys.argv[1].split(",") if x]
n = int(sys.argv[2])
for i in range(n):
    print(",".join(t[i::n]))
PY
    # Each pass is a fresh set of games under a different seed, not a re-read of the same ones, so
    # the rows compound. A pass on 9 targets costs ~22 min against a ~6.5 h train, and passes stop
    # the moment the target is met -- round 1's collection would have ended after one.
    PASS=1
    NDAG=0
    while [ "$PASS" -le "$MAX_PASSES" ]; do
      i=0
      while read -r DECKS; do
        [ -n "$DECKS" ] || continue
        # The seed base MUST differ per shard. i2c passed ROUND*100+PASS to all three, which was
        # harmless only because the engine ignored seeds; seeded, three shards on one base would
        # replay the SAME games and the round would collect a third of what it reports.
        # collect_dagger uses base + deck_index*1000 + game, so bases are spaced 100,000 apart.
        SBASE=0
        [ "$SEEDED_COLLECT" = 1 ] && SBASE=$(( 100000 + (((ROUND * 10 + PASS) * 16 + i) * 100000) ))
        # Anchors are a FIXED measurement set, collected on pass 1 only: their seeds do not
        # depend on PASS, so a later pass would replay the identical games for nothing. The panel
        # is also SPLIT across the shards -- handing the whole panel to each of the three would
        # play every anchor game three times.
        AG=0
        APANEL=""
        if [ "$PASS" = 1 ] && [ -n "$ANCHOR_PANEL" ]; then
          AG=$ANCHOR_GAMES
          APANEL=$(python3 -c "import sys; p='$ANCHOR_PANEL'.split(','); print(','.join(p[int(sys.argv[1])::int(sys.argv[2])]))" "$i" "$SHARDS")
        fi
        PYTHONPATH=cg-lib nohup python3 tools/collect_dagger.py --decks "$DECKS" \
            --model "qwen:$MODEL" --games "$COLLECT_GAMES" --seed "$((ROUND * 100 + PASS))" \
            --engine-seed-base "$SBASE" --mirror-so "$MIRROR_SO" \
            --anchor-decks "$APANEL" --anchor-games "$AG" \
            --out "$STATE/dagger_r$ROUND.p$PASS.$i.jsonl.gz" \
            > "$STATE/collect_r$ROUND.p$PASS.$i.log" 2>&1 &
        i=$((i+1))
      done < $STATE/cshards.txt
      say "collect pass $PASS/$MAX_PASSES: launched $i shards x $COLLECT_GAMES games"
      wait
      NDAG=$(zcat $STATE/dagger_r$ROUND.p*.[0-9].jsonl.gz 2>/dev/null | wc -l)
      say "collect pass $PASS: $NDAG rows in total (want $DAG_MIN)"
      [ "$NDAG" -ge "$DAG_MIN" ] && break
      PASS=$((PASS+1))
    done
    cat $STATE/dagger_r$ROUND.p*.[0-9].jsonl.gz > "$DAG" || { say "collect merge FAILED"; break; }
    rm -f $STATE/dagger_r$ROUND.p*.[0-9].jsonl.gz
    # Falling short is a finding, not a failure: it means the pilot no longer errs often enough on
    # its own worst decks to generate data. Say so and train anyway -- a thin round beats no round.
    [ "$NDAG" -ge "$DAG_MIN" ] || say "NOTE: $MAX_PASSES passes yielded $NDAG rows, under the $DAG_MIN target"
  fi
  NDAG=$(zcat "$DAG" | wc -l)
  [ "$NDAG" -gt 1000 ] || { say "collection is only $NDAG rows -- stopping"; break; }
  say "collected $NDAG rows"

  # ---- convert the collection to the decoder's schema -----------------------------------
  # collect_dagger writes the RERANK schema. Its `menu_index` field is the RAW option index and
  # is stale under menu_dedup; rerank_to_sft derives the target from the rendered menu instead
  # and verifies the identity per record.
  DAGSFT=$STATE/dagger_r$ROUND.sft.jsonl.gz
  python3 tools/rerank_to_sft.py --inp "$DAG" --out "$DAGSFT" \
      || { say "dagger conversion FAILED"; break; }

  MIX=$REPO/data/sft/i2_r$ROUND.jsonl.gz
  python3 tools/mix_sft_round.py --base "$BASE" --base-n "$BASE_N" \
      --dagger "$DAGSFT" --valued "$VALUED" --seed "$ROUND" --out "$MIX" \
      || { say "mix FAILED"; break; }

  # ---- train ----------------------------------------------------------------------------
  OUT=/root/out/i2_r$ROUND
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # A warm start that silently restores 0 LoRA tensors trains from scratch and looks like a
  # normal run for five hours. This preflight cost 10 GPU-hours to learn once.
  rm -rf /root/out/i2_pre_r$ROUND
  python3 tools/instance/sft_teacher.py --model "$BASEQ" --data "$MIX" \
      --domain-tokens --card-first "$VOCAB" --init-from "$MODEL" \
      --out /root/out/i2_pre_r$ROUND --limit 400 --eval-n 0 --steps 5 \
      --bsz 32 --accum 1 --maxlen 896 --group-by-length --save-steps 100000 2>&1 \
      | grep -E "^\[warm\]|^\[cardfirst\]|^\[data\]|REFUSING|Error" | tee $STATE/pre_r$ROUND.txt
  grep -qE "^\[warm\] embedding rows restored by name: [0-9]{4,}" $STATE/pre_r$ROUND.txt \
      || { say "STOP: too few embedding rows restored"; break; }
  grep -q "^\[warm\] LoRA tensors restored: 0 " $STATE/pre_r$ROUND.txt \
      && { say "STOP: the LoRA did not load"; break; }
  rm -rf /root/out/i2_pre_r$ROUND
  say "warm-start preflight OK"

  # bsz 32 is carried over from an A100 sweep where it peaked at 18.2 GiB. This card's round-1
  # run peaked at 25.0 GiB at bsz 8 (it also carries --eval-n 4000 and --init-from), so bsz 32
  # should land near 30 GiB of 47.4 -- but "should" is a projection from a different card, and if
  # it is wrong the round dies and the GPU idles until someone notices. Fall back rather than
  # stop: a slower round beats no round.
  TLOG=$STATE/train_r$ROUND.log
  python3 tools/instance/sft_teacher.py --model "$BASEQ" --data "$MIX" \
      --domain-tokens --card-first "$VOCAB" --init-from "$MODEL" \
      --out "$OUT" --limit 400000 --eval-n 4000 --epochs 1 \
      --bsz 32 --accum 1 --maxlen 896 --group-by-length --save-steps 1000 > "$TLOG" 2>&1
  if [ $? -ne 0 ]; then
    if grep -qiE "out of memory|CUDA error: out of memory" "$TLOG"; then
      say "bsz 32 ran out of memory on this card -- retrying at the proven bsz 8 x accum 4"
      rm -rf "$OUT"
      python3 tools/instance/sft_teacher.py --model "$BASEQ" --data "$MIX" \
          --domain-tokens --card-first "$VOCAB" --init-from "$MODEL" \
          --out "$OUT" --limit 400000 --eval-n 4000 --epochs 1 \
          --bsz 8 --accum 4 --maxlen 896 --group-by-length --save-steps 1000 >> "$TLOG" 2>&1 \
          || { say "train FAILED at bsz 8 too -- stopping. See $TLOG"; break; }
    else
      say "train FAILED for a reason other than memory -- stopping. Last lines:"
      tr '\r' '\n' < "$TLOG" | grep -av "^$" | tail -8
      break
    fi
  fi
  grep -aE "^\[peak\]|^\[saved\]" "$TLOG" | tail -2
  tr '\r' '\n' < "$TLOG" | grep -a "it\]" | tail -1
  [ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no domain_embeddings.pt -- the added rows are lost"; break; }

  say "round $ROUND done -> $OUT"
  PREV=$MIR
  MODEL=$OUT
  ROUND=$((ROUND+1))
  rm -f "$MIX"
done
echo "[i2l] LOOP ENDED"
