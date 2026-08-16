import os
import shutil

p = "/root/mirror_chain.sh"
s = open(p).read()

s = s.replace(
    'ROUNDS=${ROUNDS:-3}',
    'ROUNDS=${ROUNDS:-4}\n'
    'FROM=${FROM:-1}                     # first round number to run (resume)\n'
    'GATE_GAMES=${GATE_GAMES:-600}       # SE ~2.6pt; a DeBERTa-vs-engine game costs ~1.8 s',
    1)
s = s.replace('CUR=/root/out/dusk_s1\n', 'CUR=${CUR:-/root/out/dusk_s1}\n', 1)
s = s.replace('for R in $(seq 1 "$ROUNDS"); do',
              'MISSES=0\nfor R in $(seq "$FROM" "$ROUNDS"); do', 1)

old_g = ('    say "gate: cur vs new, mirror, 320 paired games each vs engine_v2"\n'
         '    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$DECK" --games 320 \\')
new_g = ('    say "gate: cur vs new, mirror, $GATE_GAMES paired games each vs engine_v2"\n'
         '    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$DECK" '
         '--games "$GATE_GAMES" \\')
assert old_g in s, "gate block not found"
s = s.replace(old_g, new_g, 1)

old_v = 'print("ADOPT" if not (d <= -2.0 and t <= -2.0) else "REJECT")'
new_v = (
    '# THE POINT ESTIMATE DECIDES. The old rule was `d <= -2 AND t <= -2`, and at 320 games\n'
    '# (SE 3.6pt) t = -2 needs -7.2pt -- so every drop between -2 and -7pt was ADOPTED. Round 3\n'
    '# measured -5.00pt, was adopted, and the 600-game head-to-head then put it 6.17pt (t +2.36)\n'
    '# behind the round it replaced. At 600 games SE is ~2.6pt, so "d > -2" costs a ~22% false\n'
    '# stop on a true zero -- and a false stop is cheap: the champion is kept.\n'
    'print("ADOPT" if d > -2.0 else "REJECT")')
assert old_v in s, "verdict not found"
s = s.replace(old_v, new_v, 1)

old_r = ('    else\n'
         '        say "round $R REJECTED -- stopping the chain for a human read"\n'
         '        break\n'
         '    fi')
new_r = ('        MISSES=0\n'
         '    else\n'
         '        MISSES=$((MISSES+1))\n'
         '        say "round $R REJECTED ($MISSES in a row) -- champion stays $CUR"\n'
         '        # One rejection sits within noise at this SE; two in a row from the same\n'
         '        # champion means the data has stopped moving it, and more rounds are wasted GPU.\n'
         '        [ "$MISSES" -ge 2 ] && { say "two rejections -- stopping for a human read"; '
         'break; }\n'
         '    fi')
assert old_r in s, "reject block not found"
s = s.replace(old_r, new_r, 1)

# The wrapper stage waits on a gate that already finished and chose "none"; re-running the wait
# would block forever on a log line that is never written again.
s = s.replace('say "waiting for the rule-deferral gate"\n'
              'while ! grep -q "GATE_RULES_DONE" /root/after_merge.log 2>/dev/null; do sleep 120; done',
              'if [ ! -s "$STATE/wrapper.txt" ]; then\n'
              '    say "waiting for the rule-deferral gate"\n'
              '    while ! grep -q "GATE_RULES_DONE" /root/after_merge.log 2>/dev/null; '
              'do sleep 120; done\n'
              'fi', 1)

open(p + ".new", "w").write(s)
shutil.copymode(p, p + ".new")
os.replace(p + ".new", p)
print("mirror_chain.sh patched")
