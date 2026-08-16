"""Adopt on any positive delta, not on +1.0pt.  User directive, 2026-08-14.

The +1.0pt bar has now refused ten consecutive rounds while the champion stayed at fld_r11a, and
STOP_AFTER is 08-15 12:00Z -- roughly nine more rounds exist, total.  A bar that has never once
been cleared is not protecting the champion, it is deciding that no round can ever count.

What the change does and does not buy, stated plainly so the reading of the log stays honest:
the gate is PAIRED and unbiased, so taking the point estimate has zero expected drift -- adopting
a +0.4pt arm is, on average, not a loss.  What is given up is the ratchet: a champion that got
there by luck can be replaced by one that got luckier, and with ~1-2pt of noise per round the
champion will wander within that band.  Since the rule work already moved the floor by +6.55pt
and that is baked into the baseline, wandering inside the noise band costs little and the chance
of catching a real 1-2pt gain -- invisible at the old bar -- is worth more.

MIN_GAIN is left as a variable so it can be raised again without another edit.
"""
import os

p = "/root/field_chain.sh"
s = open(p).read()

old = 'print(best if (bd is not None and bd > 1.0) else "none")'
new = ('MIN = float(os.environ.get("MIN_GAIN", "0.0"))\n'
       'print(best if (bd is not None and bd > MIN) else "none")')
assert s.count(old) == 1, "threshold anchor"
s = s.replace(old, new)

old2 = "import json, sys\narms = json.load(open(sys.argv[1])).get(\"arms\", {})"
new2 = "import json, os, sys\narms = json.load(open(sys.argv[1])).get(\"arms\", {})"
assert s.count(old2) == 1, "import anchor"
s = s.replace(old2, new2)

old3 = 'say "round $R: no challenger cleared +1.0pt -- champion stays $CUR ($MISSES in a row)"'
new3 = 'say "round $R: no challenger was positive (> ${MIN_GAIN:-0.0}pt) -- champion stays $CUR ($MISSES in a row)"'
assert s.count(old3) == 1, "message anchor"
s = s.replace(old3, new3)

# and make the knob visible at the top with the other settings
old4 = "GATE_GAMES=${GATE_GAMES:-150}"
new4 = ("GATE_GAMES=${GATE_GAMES:-150}\n"
        "# User directive 08-14: adopt on ANY positive delta. The +1.0pt bar refused ten straight\n"
        "# rounds; the gate is paired and unbiased, so the point estimate is the right thing to act\n"
        "# on when only a handful of rounds remain before STOP_AFTER.\n"
        "export MIN_GAIN=${MIN_GAIN:-0.0}")
assert s.count(old4) == 1, "knob anchor"
s = s.replace(old4, new4)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)          # new inode: the running round keeps the old file
os.chmod(p, 0o755)
print("patched: MIN_GAIN=%s" % "0.0")
print(open(p).read().count("MIN_GAIN"), "references")
