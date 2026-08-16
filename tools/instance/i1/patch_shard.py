"""Create tools/rl_loop_par.sh: rl_loop.sh with a data-parallel rollout.

Run from the repo root. Does NOT touch rl_loop.sh (bash may be executing it).
"""
import os
import shutil

SRC = "tools/rl_loop.sh"
DST = "tools/rl_loop_par.sh"
shutil.copyfile(SRC, DST)
s = open(DST).read()

OLD_VAR = 'DEC_FRAC="${RL_DECISION_FRAC:-0.5}"\n'
assert s.count(OLD_VAR) == 1, "DEC_FRAC anchor not unique"
NEW_VAR = OLD_VAR + '''# --- DATA-PARALLEL ROLLOUT (2026-07-30) ---------------------------------------------
# rl_rollout is sequential and its branch playouts run engine_v2 on ONE core, so ~20 min of a
# 36 min rollout is CPU-only with the GPU idle (measured 34% util / 111 W during rollout vs
# 100% / 328 W during the gate, which fans out to 21 processes). --nshards splits the matchup
# list into disjoint slices from the SAME seed; merge_rollouts.py concatenates records AND
# rewards in shard order and aborts on misalignment. 256 cores here, so shards cost only VRAM
# (~1.1 GB each). Set RL_SHARDS=1 to reproduce the sequential round exactly.
SHARDS="${RL_SHARDS:-6}"
'''
s = s.replace(OLD_VAR, NEW_VAR)

OLD_CALL = '''    python tools/rl_rollout.py --stage "$ST" ${TGT:+--target $TGT} \\
        --model "$POLICY" --matchups "$MCAP" \\
        ${OPP_MODEL:+--opp-model "$OPP_MODEL"} \\
        $( [ "$HEUR" != "-1" ] && echo --heuristic-frac "$HEUR" ) \\
        --branch-per-game "$BRANCH" --branch-k "$BRANCH_K" --branch-playouts "$BRANCH_PLAY" \\
        --temperature "$TEMP" --out "$RO" --seed "$R" 2>&1 | tee "$WORK/${TAG}_r${R}_rollout.log"
'''
assert s.count(OLD_CALL) == 1, "rollout call anchor not unique"

NEW_CALL = '''    if [ "$SHARDS" -le 1 ]; then
      python tools/rl_rollout.py --stage "$ST" ${TGT:+--target $TGT} \\
          --model "$POLICY" --matchups "$MCAP" \\
          ${OPP_MODEL:+--opp-model "$OPP_MODEL"} \\
          $( [ "$HEUR" != "-1" ] && echo --heuristic-frac "$HEUR" ) \\
          --branch-per-game "$BRANCH" --branch-k "$BRANCH_K" --branch-playouts "$BRANCH_PLAY" \\
          --temperature "$TEMP" --out "$RO" --seed "$R" 2>&1 | tee "$WORK/${TAG}_r${R}_rollout.log"
    else
      # every shard MUST share --seed and --matchups so they slice the SAME pair list
      local SH_FILES="" SH_PIDS="" SH_I SH_PID SH_F SH_RC=0
      for SH_I in $(seq 0 $((SHARDS - 1))); do
        python tools/rl_rollout.py --stage "$ST" ${TGT:+--target $TGT} \\
            --model "$POLICY" --matchups "$MCAP" \\
            ${OPP_MODEL:+--opp-model "$OPP_MODEL"} \\
            $( [ "$HEUR" != "-1" ] && echo --heuristic-frac "$HEUR" ) \\
            --branch-per-game "$BRANCH" --branch-k "$BRANCH_K" --branch-playouts "$BRANCH_PLAY" \\
            --temperature "$TEMP" --out "${RO%.jsonl.gz}_s${SH_I}.jsonl.gz" --seed "$R" \\
            --nshards "$SHARDS" --shard "$SH_I" \\
            > "$WORK/${TAG}_r${R}_rollout_s${SH_I}.log" 2>&1 &
        SH_PIDS="$SH_PIDS $!"
        SH_FILES="$SH_FILES ${RO%.jsonl.gz}_s${SH_I}.jsonl.gz"
      done
      for SH_PID in $SH_PIDS; do wait "$SH_PID" || SH_RC=1; done
      if [ "$SH_RC" != "0" ]; then echo "$TAG rollout shard FAILED at r$R"; return 1; fi
      python tools/merge_rollouts.py --out "$RO" $SH_FILES 2>&1 \\
        | tee "$WORK/${TAG}_r${R}_merge.log"
      # the trend line is scraped from this log, so rebuild the summary it needs
      python tools/rollout_summary.py "$RO" > "$WORK/${TAG}_r${R}_rollout.log"
      cat "$WORK/${TAG}_r${R}_rollout.log"
      for SH_F in $SH_FILES; do rm -f "$SH_F" "$SH_F.rewards.json"; done
    fi
'''
s = s.replace(OLD_CALL, NEW_CALL)
open(DST, "w").write(s)

SUMMARY = '''"""Re-print the one-line rollout summary that rl_loop.sh scrapes, for a MERGED rollout.

The sharded path in rl_loop_par.sh merges shard files with merge_rollouts.py, which prints
per-shard counts but not the aggregate line `pilot winrate NN.N%` that the loop greps for.
"""
import gzip
import json
import sys

path = sys.argv[1]
rw = json.load(open(path + ".rewards.json"))
n = len(rw)
w = sum(1 for g in rw if g["reward"] > 0)
d = sum(g["n_decisions"] for g in rw)
b = sum(1 for line in gzip.open(path, "rt") if json.loads(line).get("qvals"))
print("rollout: %d games, %d decisions, pilot winrate %.1f%%" % (n, d, 100.0 * w / max(1, n)))
print("rollout: %d branched decisions (%.1f/game)" % (b, b / max(1, n)))
'''
open("tools/rollout_summary.py", "w").write(SUMMARY)
print("wrote", DST, "and tools/rollout_summary.py")
print("size", os.path.getsize(DST), "vs", os.path.getsize(SRC))
