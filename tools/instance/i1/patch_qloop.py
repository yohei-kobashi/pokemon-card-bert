import os
p = "/root/field_chain.sh"
s = open(p).read(); orig = s

# 1) qmin joins the ladder, and the branch budget doubles at the base so that FILTERING still
#    leaves enough rows. At qmin 0.35 only ~34% of pairs survive (measured: 505 of 1500).
old = """        0) LR=2e-6; EP=0.5; BUDGET=6000;  APOW=1.0 ;;
        1) LR=4e-6; EP=0.5; BUDGET=6000;  APOW=1.0 ;;
        2) LR=2e-6; EP=1.0; BUDGET=6000;  APOW=1.0 ;;
        3) LR=2e-6; EP=0.5; BUDGET=12000; APOW=1.0 ;;   # twice the branch points
        4) LR=1e-6; EP=1.0; BUDGET=12000; APOW=2.0 ;;   # and lean harder on the lost matchups"""
new = """        0) LR=2e-6; EP=0.5; BUDGET=12000; APOW=1.0; QMIN=0.35 ;;
        1) LR=4e-6; EP=0.5; BUDGET=12000; APOW=1.0; QMIN=0.35 ;;
        2) LR=2e-6; EP=1.0; BUDGET=12000; APOW=1.0; QMIN=0.50 ;;
        3) LR=2e-6; EP=0.5; BUDGET=18000; APOW=1.0; QMIN=0.35 ;;   # more branch points
        4) LR=1e-6; EP=1.0; BUDGET=18000; APOW=2.0; QMIN=0.50 ;;   # harder on the lost matchups"""
assert s.count(old) == 1, "ladder not found"
s = s.replace(old, new)
s = s.replace('branch-budget $BUDGET alloc-power $APOW"',
              'branch-budget $BUDGET alloc-power $APOW qmin $QMIN"')
s = s.replace("APOW=${ALLOC_POWER:-1.0}\n",
              "APOW=${ALLOC_POWER:-1.0}\nQMIN=${QMIN:-0.35}\nTEMP=${TEMP:-0.25}\n")

# 2) convert with the filter, and RELAX rather than die if the round comes up short. The old
#    guard exited the whole chain on <500 rows; with a filter in front of it that turns a thin
#    round into the end of the run.
old2 = """        python3 /root/mrl_convert.py --pairs "$PAIRS" --out "$ROWS" \\
            --beta "$BETA" --temp 0.5 | tee -a "$STATE/convert$R.log"
        NR=$(zcat "$ROWS" | wc -l)
        [ "$NR" -ge 500 ] || { say "STOP: only $NR rows"; exit 1; }"""
new2 = """        # The playout advantage |qw-ql| is the label's confidence. Measured over rounds 7-9
        # (4,519 pairs): training on ALL of them moved held-out conformance 54.3 -> 53.6, i.e.
        # DOWN, while >=0.35 moved it 52.1 -> 58.1 and >=0.60 drove the loss below ln(2) for the
        # first time. With 24 playouts the Q estimate has an SE near 0.2 and the median pair
        # margin is 0.26, so the low-confidence majority was outvoting the real signal -- which
        # is why nine rounds trained at exactly chance while still perturbing the weights enough
        # to cost ~3.6pt an arm.
        NR=0
        for Q in "$QMIN" 0.25 0.15 0.0; do
            python3 /root/mrl_convert.py --pairs "$PAIRS" --out "$ROWS" \\
                --beta "$BETA" --temp "$TEMP" --qmin "$Q" | tee -a "$STATE/convert$R.log"
            NR=$(zcat "$ROWS" | wc -l)
            [ "$NR" -ge 500 ] && { QUSED=$Q; break; }
            say "qmin $Q leaves only $NR rows -- relaxing (a thin round must not end the run)"
        done
        [ "$NR" -ge 500 ] || { say "STOP: only $NR rows even unfiltered"; exit 1; }
        [ "$QUSED" = "$QMIN" ] || say "NOTE: round $R trained at qmin $QUSED, not $QMIN"'"'"''"'"'"""
assert s.count(old2) == 1, "convert block not found"
s = s.replace(old2, new2.replace("'\"'\"''\"'\"'", ""))
s = s.replace('say "train $V: beta $BETA, $NR rows, lr $LR ep $EP l2sp 1e-2"',
              'say "train $V: beta $BETA, $NR rows (qmin $QUSED temp $TEMP), lr $LR ep $EP l2sp 1e-2"')
assert s != orig
open(p + ".tmp", "w").write(s); os.chmod(p + ".tmp", 0o755); os.replace(p + ".tmp", p)
print("patched field_chain.sh: qmin filter + relaxation + ladder")
