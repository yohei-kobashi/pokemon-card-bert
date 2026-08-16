"""Make "take the second turn" expressible, so the choice can be tested instead of assumed.

MEASURED: of 300 games across two opponents, we were asked to choose 150 times and answered
"first" 150 times. Not a tendency -- a constant. No rule covers the decision, so this is the
pilot's default rather than anyone's judgement.

The guides treat it as a genuine trade-off with a metagame answer:

    "先攻だと早期の「ファントムダイブ」で相手を壊滅させやすく、
     後攻だと最初の番から「むずむずかふん」できます"

Both halves are checkable against what this deck actually does. The first-turn case is worth
less to us than to the guides' readers: our first Phantom Dive lands on our turn 7-8 and in only
53% of games, so "early Phantom Dive" is not a benefit we are collecting. The second-turn case is
worth something we can name: Budew's item lock one turn sooner, on the turn the opponent is
setting up -- and against ogerpon_mono, whose four Crushing Hammers are why 25% of our games
never reach a payable attacker, locking their items on turn one is the most direct answer we have.

Phrased as a prohibition (do not take the first turn) so it composes with the existing wrap, and
gated OFF by default: this is a hypothesis to be gated, not a belief to be shipped.
"""
import os

p = "/root/ptcg/repo_sb/tools/dusk_plan.py"
s = open(p).read()

old = '''    "search_bottom": ("do not search out a card whose pre-evolution is not in play", 1.5),'''
new = '''    "search_bottom": ("do not search out a card whose pre-evolution is not in play", 1.5),
    "go_second": ("do not take the first turn -- take the second", 1.0),'''
assert s.count(old) == 1, "RULES anchor"
s = s.replace(old, new)

old2 = "_TO_HAND_CTX = 7"
new2 = """_TO_HAND_CTX = 7
_IS_FIRST_CTX = 41           # SelectContext::IsFirst as the observation reports it (enum - 1)"""
assert s.count(old2) == 1, "ctx anchor"
s = s.replace(old2, new2)

old3 = """    # --- deck searches that put the card in HAND ------------------------------------------"""
new3 = '''    # --- the coin-flip winner's choice ------------------------------------------------------
    # A yes/no menu: "yes" takes the first turn. Measured 150 of 150 times we answered yes, with
    # no rule involved. Forbidding "yes" is how the alternative becomes testable at all.
    if sel.get("context") == _IS_FIRST_CTX and len(opts) >= 2:
        yes = {i for i, _o, t in texts() if t == "yes"}
        if yes and len(yes) < len(opts):
            out["go_second"] = (set(range(len(opts))) - yes, set(range(len(opts))))

    # --- deck searches that put the card in HAND ------------------------------------------'''
assert s.count(old3) == 1, "insert anchor"
s = s.replace(old3, new3)
open(p + ".new", "w").write(s)
os.replace(p + ".new", p)

q = "/root/ptcg/repo_sb/lm/plan_filter.py"
t = open(q).read()
old4 = '''    "search_bottom",     # do not search out a card whose pre-evolution is not in play'''
new4 = '''    "search_bottom",     # do not search out a card whose pre-evolution is not in play
    "go_second",         # do not take the first turn'''
assert t.count(old4) == 1, "PROHIBITIONS anchor"
open(q + ".new", "w").write(t.replace(old4, new4))
os.replace(q + ".new", q)
print("go_second added (repo_sb only; the live tree is untouched)")
