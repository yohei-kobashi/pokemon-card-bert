import os
p = "/root/ptcg/repo_sb/tools/dusk_ogerpon_audit.py"
s = open(p).read()
# "our turns per game" was engine turns halved, while every other number here is an ORDINAL
# rank among our own decisions. Two units in one table invites exactly the misreading this
# audit exists to prevent.
old = '''    print("  our turns per game (mean) %.1f   |  games where our Dragapult ex was KO'd %d"
          % (sum(last_turn.values()) / max(1, len(last_turn)) / 2.0, sum(1 for v in pult_lost.values() if v)))'''
new = '''    _ours = [max((rank_of.get((g, t), 0) for (gg, t) in dev if gg == g), default=0)
             for g in set(gg for gg, _ in dev)]
    print("  our turns per game (mean) %.1f   |  games where our Dragapult ex was KO'd %d"
          % (sum(_ours) / max(1, len(_ours)) - 1, sum(1 for v in pult_lost.values() if v)))'''
assert s.count(old) == 1, "units anchor"
open(p + ".n", "w").write(s.replace(old, new))
os.replace(p + ".n", p)
print("units fixed")
