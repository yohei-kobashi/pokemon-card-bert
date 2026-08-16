"""Size the smoke run so it can actually reach both trainings and the gate.

It died at 49 pairs against dpo_teacher's own floor of 50 -- and the FILTERED arm needs 50 after
the qmin cut, not before, so the target is not "51 pairs" but "enough that the survivors clear
the floor too".  MINROWS is what guarantees that: the threshold search keeps at least MINROWS
rows, so setting it to 100 makes the filtered arm >= 100 by construction.

Collection is ~96 games across two shards, a few minutes.  The point of a smoke is that it is
cheap, not that it is tiny.
"""
import os

p = "/root/night_run.sh"
s = open(p).read()

old = """    OPPS=marnie_grimmsnarl,ogerpon_mono GAMES=10 GATE_GAMES=6 EPOCHS=1 \\
    BR_BUDGET=4000 BR_PLAYOUTS=6 BR_PERGAME=12 BR_WAIT=12 BR_FALLBACK=0 MINROWS=10 \\"""
new = """    OPPS=marnie_grimmsnarl,ogerpon_mono GAMES=24 GATE_GAMES=6 EPOCHS=1 \\
    BR_BUDGET=10000 BR_PLAYOUTS=6 BR_PERGAME=20 BR_WAIT=12 BR_FALLBACK=0 MINROWS=100 \\"""
assert s.count(old) == 1, "knob anchor"
s = s.replace(old, new)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
os.chmod(p, 0o755)
print("patched")
