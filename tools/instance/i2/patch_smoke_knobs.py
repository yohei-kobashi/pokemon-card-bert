"""The smoke run stopped at its own row floor before reaching the two trainings.

It did its job as far as it went -- collection ran, instance1 served the branch in two minutes,
the reply landed intact -- and then stopped loudly on "only 15 pairs" because MINROWS was 20.
That is the right failure mode but the wrong number: a smoke run has to reach the trainings and
the gate, since that is where the last four nights actually died.

Bigger branch budget, more branch points per game, lower floor.  Still minutes, not hours.
"""
import os

p = "/root/night_run.sh"
s = open(p).read()

old = """    OPPS=marnie_grimmsnarl,ogerpon_mono GAMES=4 GATE_GAMES=6 EPOCHS=1 \\
    BR_BUDGET=1200 BR_PLAYOUTS=6 BR_PERGAME=6 BR_WAIT=12 BR_FALLBACK=0 MINROWS=20 \\"""
new = """    OPPS=marnie_grimmsnarl,ogerpon_mono GAMES=10 GATE_GAMES=6 EPOCHS=1 \\
    BR_BUDGET=4000 BR_PLAYOUTS=6 BR_PERGAME=12 BR_WAIT=12 BR_FALLBACK=0 MINROWS=10 \\"""
assert s.count(old) == 1, "knob anchor"
s = s.replace(old, new)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
os.chmod(p, 0o755)
print("patched")
