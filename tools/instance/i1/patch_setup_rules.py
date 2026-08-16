"""Aim the setup rules at the human template instead of at one symptom.

The template, from the Japanese guides and consistent across them:

    turn 1   bench Dreepy x3 (two at minimum) and Duskull x1-2, using Buddy-Buddy Poffin
    turn 2   evolve to Drakloak and draw with Recon Directive; three Drakloak is the ideal
    energy   "extremely tight -- attach as early and as many as you can"

Measured against it, we open on 1.45 Dreepy and 0.39 Drakloak.  `search_bottom` alone (forbid
the top of a line whose bottom is not in play) moved the first-turn ToHand searches from 5% to
24% and the board to 1.67 Dreepy -- real, but a prohibition can only push the pick off Dragapult
ex, and the freed picks went to Trainers and Munkidori rather than to the line.

So add the positive half.  `setup_search` NOMINATES the step the board is actually missing, and
`bench_line` stops treating Budew as interchangeable with Dreepy while the line is short.

  setup_search  in a deck->hand search, while the line is under the template, take the missing
                step: Dreepy until two line bodies exist, then Drakloak, then Duskull
  bench_line    while fewer than two Dreepy are in play, only Dreepy conforms -- the old set was
                (Dreepy, Duskull, Budew) under a name that says Dreepy, so a Budew scored as a
                perfect opening

Both read only our own board and the menu; neither looks at the opponent.
"""
import os

p = "/root/ptcg/repo_sb/tools/dusk_plan.py"
s = open(p).read()

old = '''    "search_bottom": ("do not search out a card whose pre-evolution is not in play", 1.5),'''
new = '''    "search_bottom": ("do not search out a card whose pre-evolution is not in play", 1.5),
    "setup_search": ("search out the step the line is missing, while it is still short", 2.0),'''
assert s.count(old) == 1, "RULES anchor"
s = s.replace(old, new)

old = "_PREREQ = {DRAKLOAK: DREEPY, PULT: DRAKLOAK, DUSCLOPS: DUSKULL, DUSKNOIR: DUSCLOPS}"
new = """_PREREQ = {DRAKLOAK: DREEPY, PULT: DRAKLOAK, DUSCLOPS: DUSKULL, DUSKNOIR: DUSCLOPS}
# The engine's turn counter is shared by both seats, so six of its turns is roughly our first
# three -- which is the window the template describes and the window where the measured defect
# lives (with a Drakloak up, the same menus already read 30-40%).
_SETUP_TURNS = 6
_LINE_TARGET = 2          # Dreepy-or-Drakloak bodies before the search moves on"""
assert s.count(old) == 1, "prereq anchor"
s = s.replace(old, new)

old = """        if bad and ready and len(bad) < len(opts):
            out["search_bottom"] = (set(range(len(opts))) - bad, set(range(len(opts))))
"""
new = """        if bad and ready and len(bad) < len(opts):
            out["search_bottom"] = (set(range(len(opts))) - bad, set(range(len(opts))))

        # ... and the positive half: name the step the board is missing. Ordered, because the
        # template is ordered -- a second Drakloak is worth less than a first Dreepy, and
        # Duskull only matters once the Dragapult line can actually run.
        if turn <= _SETUP_TURNS:
            want = None
            if my_ids.count(DREEPY) + my_ids.count(DRAKLOAK) < _LINE_TARGET:
                want = DREEPY
            elif DRAKLOAK not in my_ids and PULT not in my_ids:
                want = DRAKLOAK
            elif DUSKULL not in my_ids:
                want = DUSKULL
            if want is not None:
                got = {i for i, o in enumerate(opts)
                       if isinstance(o, dict) and _card_at(o, obs) == want}
                if got and len(got) < len(opts):
                    out["setup_search"] = (got, set(range(len(opts))))
"""
assert s.count(old) == 1, "search anchor"
s = s.replace(old, new)

# bench_line: stop scoring a Budew as a perfect opening while the line is short
old = """        b = {i for i, o, _ in play
             if isinstance(o, dict) and _card_of(obs, yi, o) in (DREEPY, DUSKULL, BUDEW)}"""
new = """        # The rule is named "put Dreepy on the bench" and used to accept Duskull or Budew for
        # it. That is right once the line is up and wrong before: the template wants three
        # Dreepy first, and this deck runs a single Budew whose lock is worth a turn only while
        # something else is developing. Below the target, only Dreepy conforms.
        _want_early = ((DREEPY,) if my_ids.count(DREEPY) + my_ids.count(DRAKLOAK) < _LINE_TARGET
                       else (DREEPY, DUSKULL, BUDEW))
        b = {i for i, o, _ in play
             if isinstance(o, dict) and _card_of(obs, yi, o) in _want_early}"""
assert s.count(old) == 1, "bench anchor"
s = s.replace(old, new)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched dusk_plan (setup_search + bench_line)")
