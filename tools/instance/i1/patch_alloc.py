import os, sys, re
p = "/root/field_chain.sh"
s = open(p).read()
orig = s

# 1) knobs: allocation sharpness joins lr / epochs / branch budget on the ladder
old_case = """    case $((MISSES % 5)) in
        0) LR=2e-6; EP=0.5; BUDGET=6000 ;;
        1) LR=4e-6; EP=0.5; BUDGET=6000 ;;
        2) LR=2e-6; EP=1.0; BUDGET=6000 ;;
        3) LR=2e-6; EP=0.5; BUDGET=12000 ;;   # data allocation: twice the branch points
        4) LR=1e-6; EP=1.0; BUDGET=12000 ;;
    esac"""
new_case = """    case $((MISSES % 5)) in
        0) LR=2e-6; EP=0.5; BUDGET=6000;  APOW=1.0 ;;
        1) LR=4e-6; EP=0.5; BUDGET=6000;  APOW=1.0 ;;
        2) LR=2e-6; EP=1.0; BUDGET=6000;  APOW=1.0 ;;
        3) LR=2e-6; EP=0.5; BUDGET=12000; APOW=1.0 ;;   # twice the branch points
        4) LR=1e-6; EP=1.0; BUDGET=12000; APOW=2.0 ;;   # and lean harder on the lost matchups
    esac"""
assert s.count(old_case) == 1, "knob case not found"
s = s.replace(old_case, new_case)
s = s.replace('say "knobs (miss streak $MISSES): lr $LR epochs $EP branch-budget $BUDGET"',
              'say "knobs (miss streak $MISSES): lr $LR epochs $EP branch-budget $BUDGET alloc-power $APOW"')

# 2) APOW must exist before the first round reaches the collect step: the ladder is evaluated
#    AFTER collection, so without a default the very first round would expand to an empty
#    --power and die under `set -u`.
s = s.replace('MISSES=0\nBUDGET=6000\n',
              'MISSES=0\nBUDGET=6000\nAPOW=${ALLOC_POWER:-1.0}\n'
              'ALLOC_MIN=${ALLOC_MIN:-0.5}\nALLOC_MAX=${ALLOC_MAX:-2.0}\n')

# 3) allocate the round's games from the PREVIOUS gate before collecting
old_collect = '''        gpu_wait
        say "collect: $COLLECT games vs each of $OPPS, champion through the wrapper"
        python3 tools/lm_mirror_log.py --model "${PFX}hf:$CUR" --deck-model engine --fmt dusk \\
            --protagonist "$DECK" --decks "$OPPS" --games "$COLLECT" \\'''
new_collect = '''        gpu_wait
        # Spend the SAME total games unevenly: the matchups the champion loses get more of the
        # round. Measured on round 4, the pairs reaching training were 11.3-14.3% per opponent
        # against win rates spanning 3.3-49.3% -- the loop was learning the won matchups as hard
        # as the lost ones. Clamped to [ALLOC_MIN, ALLOC_MAX] x even, because concentrating a
        # round on one matchup is a move already measured and lost (narrow DAgger: +11.9pt on
        # the target, -2.75pt on the fleet).
        NOPP=$(echo "$OPPS" | tr ',' '\\n' | grep -c .)
        PREV_GATE="$STATE/gate_r$((R-1)).json"; [ -f "$PREV_GATE" ] || PREV_GATE="$STATE/base.json"
        GPD=""
        if [ -f "$PREV_GATE" ]; then
            GPD=$(python3 tools/field_alloc.py --gate "$PREV_GATE" --arm cur \\
                    --total $((COLLECT * NOPP)) --power "$APOW" \\
                    --min-mult "$ALLOC_MIN" --max-mult "$ALLOC_MAX" \\
                    --order "$OPPS" --report 2>>"$STATE/alloc$R.log") \\
                || { say "alloc FAILED -- falling back to the even split"; tail -3 "$STATE/alloc$R.log"; GPD=""; }
        fi
        if [ -n "$GPD" ]; then
            say "alloc from $(basename "$PREV_GATE") (power $APOW): $GPD"
            sed 's/^/    /' "$STATE/alloc$R.log"
        else
            say "alloc: no previous gate -- even split, $COLLECT per opponent"
        fi
        say "collect: $((COLLECT * NOPP)) games over $OPPS, champion through the wrapper"
        python3 tools/lm_mirror_log.py --model "${PFX}hf:$CUR" --deck-model engine --fmt dusk \\
            --protagonist "$DECK" --decks "$OPPS" --games "$COLLECT" \\
            ${GPD:+--games-per-deck "$GPD"} \\'''
assert s.count(old_collect) == 1, "collect block not found"
s = s.replace(old_collect, new_collect)

assert s != orig
open(p + ".tmp", "w").write(s)
os.chmod(p + ".tmp", 0o755)
os.replace(p + ".tmp", p)
print("patched field_chain.sh (new inode; the running loop keeps the old one until restart)")
