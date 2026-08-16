import os

p = "/root/field_chain.sh"
s = open(p).read()

helper = '''rules_fp() {   # what the pilot IS, as one hash
    md5sum "$REPO/tools/dusk_plan.py" "$REPO/lm/plan_filter.py" 2>/dev/null \\
        | awk '{print $1}' | md5sum | cut -d' ' -f1
}

'''
anchor = "prune_ckpts() {"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor, 1)

# Snapshot at the top of the round.
old = '    say "================ field round $R (champion $CUR) ================"'
new = ('    FP0=$(rules_fp)\n'
       '    say "================ field round $R (champion $CUR) ================"')
assert s.count(old) == 1, ("snap", s.count(old))
s = s.replace(old, new)

# Check before spending the GPU on training, and again before the gate. Each python step
# re-imports dusk_plan from disk, so an edit landing mid-round trains and gates a pilot that is
# not the one the data was collected with -- and nothing anywhere errors. Redo the round instead.
guard = '''    if [ "$(rules_fp)" != "$FP0" ]; then
        say "RULES CHANGED mid-round -- the traces were collected by a different pilot."
        say "Discarding round $R data and re-collecting under the new rules."
        rm -f "$TR" "$PAIRS" /root/fld_log$R.jsonl.gz
        continue
    fi
'''
old2 = '''    # LR / EPOCH / DATA ladder, indexed by the current miss streak.'''
assert s.count(old2) == 1, ("guard1", s.count(old2))
s = s.replace(old2, guard + old2)

old3 = '''    gpu_wait
    say "gate: champion vs a vs b vs $OPPS, $GATE_GAMES games/opponent"'''
new3 = '''    if [ "$(rules_fp)" != "$FP0" ]; then
        say "RULES CHANGED during training -- gate would score a pilot the data never used."
        say "Discarding round $R and re-collecting."
        rm -f "$TR" "$PAIRS" /root/fld_log$R.jsonl.gz
        continue
    fi
    gpu_wait
    say "gate: champion vs a vs b vs $OPPS, $GATE_GAMES games/opponent"'''
assert s.count(old3) == 1, ("guard2", s.count(old3))
s = s.replace(old3, new3)

t = p + ".new"
open(t, "w").write(s)
os.chmod(t, 0o755)
os.replace(t, p)
print("field_chain: a rule edit landing mid-round now discards and re-collects that round")
