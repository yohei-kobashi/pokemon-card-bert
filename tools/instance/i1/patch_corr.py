"""Does a faster opening actually WIN, or does it merely correlate with a good draw?

A reward term for setup speed is only worth its weight if the thing it rewards causes the thing
we want.  Two questions, and they are not the same:

  * correlation -- among our own games, do the ones that opened fast win more?  Cheap, but
    confounded: a hand that gives three Dreepy also gives Trainers and energy, so this measures
    the draw as much as the play.
  * intervention -- when we CHANGE the policy to open faster, does the win rate move?  That is
    the 8-opponent gate now running, and it is the one that settles it.

This adds the first.  A weak or negative correlation would kill the premise outright and save a
training round; a strong one still would not prove causation, but it bounds how much a perfect
setup policy could possibly be worth, which is the number needed to size the reward term.
"""
import os

p = "/root/ptcg/repo_sb/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = "    first_pd = {}                   # game -> first turn Phantom Dive was actually used\n"
new = ("    first_pd = {}                   # game -> first turn Phantom Dive was actually used\n"
       "    game_won = {}                   # game -> 1/0, joined to `dev` by the game index\n")
assert s.count(old) == 1, "state anchor"
s = s.replace(old, new)

old = """        wins += 1 if r == mine else 0"""
new = """        wins += 1 if r == mine else 0
        game_won[g] = 1 if r == mine else 0"""
assert s.count(old) == 1, "loop anchor"
s = s.replace(old, new)

old = '''    print("\\n-- ToHand searches'''
new = '''    print("\\n-- does a faster opening win?  (our turn 1 and 2 boards vs the game result) --")
    for _lbl, _key, _ord in (("Dreepy on our turn 1", "dreepy", 2),
                             ("Drakloak on our turn 2", "drakloak", 3),
                             ("bodies on our turn 1", "bodies", 2)):
        buckets = collections.defaultdict(list)
        for (g, tt), v in dev.items():
            if rank_of.get((g, tt)) == _ord and g in game_won:
                buckets[v[_key]].append(game_won[g])
        if not buckets:
            continue
        print("  %-24s %s" % (_lbl, "  ".join(
            "%s:%.0f%%(n=%d)" % (k, 100.0 * sum(v) / len(v), len(v))
            for k, v in sorted(buckets.items()))))
    # A single split is easier to read than the full table and is the number a reward term
    # would be trying to buy.
    fast = [game_won[g] for (g, tt), v in dev.items()
            if rank_of.get((g, tt)) == 2 and g in game_won and v["dreepy"] >= 2]
    slow = [game_won[g] for (g, tt), v in dev.items()
            if rank_of.get((g, tt)) == 2 and g in game_won and v["dreepy"] < 2]
    if fast and slow:
        import math as _m
        pf, ps = sum(fast) / len(fast), sum(slow) / len(slow)
        se = _m.sqrt(pf * (1 - pf) / len(fast) + ps * (1 - ps) / len(slow))
        print("  >=2 Dreepy on turn 1: %.1f%% (n=%d)   <2: %.1f%% (n=%d)   diff %+.1f +- %.1f pt"
              % (100 * pf, len(fast), 100 * ps, len(slow), 100 * (pf - ps), 100 * se))

    print("\\n-- ToHand searches'''
assert s.count(old) == 1, "report anchor"
s = s.replace(old, new)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched")
