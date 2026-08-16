"""When could we first have attacked, and when did we?

The gate says the first Phantom Dive lands on our turn 7.1 and in 53% of games.  Mechanically the
floor is our turn 3 -- Dreepy on 1, Drakloak on 2, Dragapult ex on 3, with energy carried up
through the evolutions -- so 7.1 is four turns of something.  Two very different somethings:

  * the body was never PAYABLE (2 energy including {R} and {P} on one Dragapult), which is an
    energy problem and points at Crispin and the attach decisions, or
  * it was payable and we did something else, which is a decision problem the plan can address.

`_can_pd` is dusk_plan's own test for "this body can pay Phantom Dive", so the readiness turn is
measured with the same predicate the rules use rather than a second implementation of it.
"""
import os

p = "/root/ptcg/repo_sb/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = "    game_won = {}                   # game -> 1/0, joined to `dev` by the game index\n"
new = ("    game_won = {}                   # game -> 1/0, joined to `dev` by the game index\n"
       "    first_ready = {}                # game -> first turn a body could PAY Phantom Dive\n"
       "    first_pult = {}                 # game -> first turn Dragapult ex was in play\n")
assert s.count(old) == 1, "state anchor"
s = s.replace(old, new)

old = """                if any(isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE
                       for i, o in enumerate(opts) if i in picked):
                    first_pd.setdefault(cur[0], turn)"""
new = """                if any(isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE
                       for i, o in enumerate(opts) if i in picked):
                    first_pd.setdefault(cur[0], turn)
                if PULT in ids:
                    first_pult.setdefault(cur[0], turn)
                if any(_plan._can_pd(x) for x in ma + mb if isinstance(x, dict)):
                    first_ready.setdefault(cur[0], turn)"""
assert s.count(old) == 1, "pd anchor"
s = s.replace(old, new)

old = '''    print("\\n-- does a faster opening win?'''
new = '''    def _ord(d):
        v = [rank_of.get((g, t)) for g, t in d.items() if rank_of.get((g, t))]
        return (sum(v) / len(v) - 1, len(v)) if v else (0.0, 0)
    print("\\n-- when could we have attacked, and when did we? (our-turn ordinals) --")
    for _lbl, _d in (("Dragapult ex first in play", first_pult),
                     ("a body could first PAY Phantom Dive", first_ready),
                     ("Phantom Dive actually used", first_pd)):
        m, n = _ord(_d)
        print("  %-38s turn %5.2f   in %3d of %d games" % (_lbl, m, n, a.games))
    both = [(rank_of.get((g, first_pd[g])), rank_of.get((g, first_ready[g])))
            for g in first_pd if g in first_ready
            and rank_of.get((g, first_pd[g])) and rank_of.get((g, first_ready[g]))]
    if both:
        gap = [x - y for x, y in both]
        print("  -> of the games that DID attack, the wait between payable and firing was"
              " %.2f turns (n=%d)" % (sum(gap) / len(gap), len(gap)))
    never = [g for g in game_won if g not in first_ready]
    print("  -> %d of %d games never had a payable Phantom Dive at all"
          % (len(never), len(game_won)))

    print("\\n-- does a faster opening win?'''
assert s.count(old) == 1, "report anchor"
s = s.replace(old, new)

# the audit imports dusk_plan already for the lock_early check
assert "import dusk_plan as _plan" in s, "plan import missing"

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched")
