import os

p = "/root/field_chain.sh"
s = open(p).read()

# --- 1. run until the day before the deadline, not until three misses -----------------------
old = 'ROUNDS=${ROUNDS:-8}'
new = ('ROUNDS=${ROUNDS:-200}\n'
       '# User directive 2026-08-12: repeat this same RL until the day before the deadline and\n'
       '# meet a plateau with learning-rate / data-allocation changes, not with a new method.\n'
       '# So the loop stops on the CLOCK, never on a run of misses.\n'
       'STOP_AFTER=${STOP_AFTER:-2026-08-15T12:00:00Z}\n'
       '# The opponent pilot. engine_v2 now; flip to "reg" once this converges, which points each\n'
       '# opponent deck at its own Qwen-4B LoRA from instance2 -- that handoff is the whole reason\n'
       '# those adapters are being trained, and it is a one-word change here.\n'
       'OPP_SPEC=${OPP_SPEC:-engine}')
assert s.count(old) == 1, ("rounds", s.count(old))
s = s.replace(old, new)

for a, b in (('--opp-spec engine \\', '--opp-spec "$OPP_SPEC" \\'),):
    assert s.count(a) == 2, ("oppspec", s.count(a))
    s = s.replace(a, b)

# --- 2. a plateau changes the KNOBS, it does not end the run --------------------------------
old2 = '''    if [ "$WIN" = "none" ]; then
        MISSES=$((MISSES+1))
        say "round $R: no challenger cleared +1.0pt -- champion stays $CUR ($MISSES in a row)"
        [ "$MISSES" -ge 3 ] && { say "three misses -- stopping"; break; }'''
new2 = '''    if [ "$WIN" = "none" ]; then
        MISSES=$((MISSES+1))
        say "round $R: no challenger cleared +1.0pt -- champion stays $CUR ($MISSES in a row)"
        # No break. A miss advances the knob ladder below and the loop goes again; the previous
        # behaviour (stop after three) is what left the GPU idle for six hours on 08-11.'''
assert s.count(old2) == 1, ("miss", s.count(old2))
s = s.replace(old2, new2)

# --- 3. the knob ladder ---------------------------------------------------------------------
old3 = '''    for V in a b; do
        BETA=0.0; [ "$V" = "b" ] && BETA=0.3'''
new3 = '''    # LR / EPOCH / DATA ladder, indexed by the current miss streak. Cycles, so a long plateau
    # keeps sampling the space instead of re-running one setting that has already failed. lr is
    # the first knob because [[mirror-rl-training-is-the-noise-source]] measured a 26pt row-order
    # swing at 1e-5 against 1.00pt at 2e-6 -- above ~4e-6 this training is a lottery, not a fit.
    case $((MISSES % 5)) in
        0) LR=2e-6; EP=0.5; BUDGET=6000 ;;
        1) LR=4e-6; EP=0.5; BUDGET=6000 ;;
        2) LR=2e-6; EP=1.0; BUDGET=6000 ;;
        3) LR=2e-6; EP=0.5; BUDGET=12000 ;;   # data allocation: twice the branch points
        4) LR=1e-6; EP=1.0; BUDGET=12000 ;;
    esac
    say "knobs (miss streak $MISSES): lr $LR epochs $EP branch-budget $BUDGET"

    for V in a b; do
        BETA=0.0; [ "$V" = "b" ] && BETA=0.3'''
assert s.count(old3) == 1, ("ladder", s.count(old3))
s = s.replace(old3, new3)

old4 = '--out /root/out/fld_r$R$V --lr 2e-6 --epochs 0.5 --accum 4 --l2sp 1e-2 \\'
new4 = '--out /root/out/fld_r$R$V --lr "$LR" --epochs "$EP" --accum 4 --l2sp 1e-2 \\'
assert s.count(old4) == 1, ("lr", s.count(old4))
s = s.replace(old4, new4)

old5 = '--budget 6000 --per-game 15 --margin-min 0.01 --playouts 24 --workers "$WORKERS" \\'
new5 = '--budget "${BUDGET:-6000}" --per-game 15 --margin-min 0.01 --playouts 24 --workers "$WORKERS" \\'
assert s.count(old5) == 1, ("budget", s.count(old5))
s = s.replace(old5, new5)

# The branch runs BEFORE the ladder is evaluated in the loop body, so seed BUDGET for round 1.
old6 = 'MISSES=0\nfor R in $(seq "$FROM" "$ROUNDS"); do'
new6 = ('MISSES=0\nBUDGET=6000\nfor R in $(seq "$FROM" "$ROUNDS"); do\n'
        '    if [ "$(date -u +%s)" -ge "$(date -u -d "$STOP_AFTER" +%s)" ]; then\n'
        '        say "reached STOP_AFTER ($STOP_AFTER) -- ending with champion $CUR"; break\n'
        '    fi')
assert s.count(old6) == 1, ("loop", s.count(old6))
s = s.replace(old6, new6)

t = p + ".new"
open(t, "w").write(s)
os.chmod(t, 0o755)
os.replace(t, p)
print("field_chain: runs to STOP_AFTER, miss -> knob ladder, OPP_SPEC switch for the 4B handoff")
