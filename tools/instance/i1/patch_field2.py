import os

p = "/root/field_chain.sh"
s = open(p).read()

# EIGHT opponents, spanning both brackets, because we have now overfit a gate in each direction:
#   * gating on alakazam_nz + marnie (the top-500's top two, which appeared in ZERO of our 19 live
#     games) produced fld_r1b: +5.20pt there, -1.87 to -4.00pt on the decks we actually meet.
#   * gating only on the five decks we meet at ~330 rating would overfit the bracket we are
#     trying to climb out of. The panel has to hold the ladder we are on AND the one above it.
# Live bracket: mega_abomasnow_sample, dudunsparce_box, dragapult, archaludon, ogerpon_mono,
# ethan_hooh (all met in the 19 games). Top bracket: marnie_grimmsnarl (35.4% of the top-500)
# and alakazam_nz.
old = 'OPPS=${OPPS:-mega_abomasnow_sample,dudunsparce_box,dragapult,archaludon,ogerpon_mono}'
new = ('OPPS=${OPPS:-marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,'
       'ogerpon_mono,mega_abomasnow_sample,ethan_hooh}')
assert s.count(old) == 1, ("opps", s.count(old))
s = s.replace(old, new)

# 150 per (arm, opponent) over 8 opponents = 1200 paired games per arm, against 500 over 2. More
# opponents at fewer games each is better on BOTH axes here: coverage widens and the aggregate SE
# falls (~1.3pt vs ~2.2pt), because the gate's statistic is the mean over cells, not any one cell.
old2 = 'GATE_GAMES=${GATE_GAMES:-250}'
new2 = 'GATE_GAMES=${GATE_GAMES:-150}'
assert s.count(old2) == 1, ("gate", s.count(old2))
s = s.replace(old2, new2)

old3 = 'COLLECT=${COLLECT:-200}                        # per opponent'
new3 = 'COLLECT=${COLLECT:-100}                        # per opponent; 8 x 100 = 800 games/round'
assert s.count(old3) == 1, ("collect", s.count(old3))
s = s.replace(old3, new3)

t = p + ".new"
open(t, "w").write(s)
os.chmod(t, 0o755)
os.replace(t, p)
print("field_chain: 8 opponents across both brackets, 150/cell gate, 100/cell collect")
