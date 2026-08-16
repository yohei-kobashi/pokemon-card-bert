"""Add `search_bottom`: do not search out a card whose pre-evolution is not in play.

MEASURED, 60 games vs ogerpon_mono, our own turns, per menu of context ToHand (Ultra Ball x4,
Poke Pad x4, Night Stretcher x2 -- the deck-to-HAND searches; Buddy-Buddy Poffin is a different
context and is already fine at 94%):

    our turn   menus with a Dreepy on offer   took it        board
    1                     42                    2   ( 5%)    no Drakloak
    2                      9                    2   (22%)    no Drakloak
    4                      9                    0   ( 0%)    no Drakloak
    2                     10                    3   (30%)    Drakloak in play
    6+                    30                   12   (40%)    Drakloak in play

    taken instead, no Drakloak:  Dragapult ex x45, Dusknoir x37, Munkidori x13
    taken instead, w/ Drakloak:  Crispin x24, Dragapult ex x23, Dusknoir x14

Dragapult ex cannot be played until a Drakloak exists, and Dusknoir until a Dusclops does.  On
our first turn we pull them out of the deck anyway, where they sit in hand while the board stays
at 1.45 Dreepy (the human template is 3) and the first Phantom Dive lands on our turn 9 -- in
only 13 of 60 games.  Once a Drakloak IS up the same menus read 30-40% and Crispin becomes the
top pick, so this is an opening defect, not missing knowledge.

Phrased as a PROHIBITION, so it intersects rather than nominates ([[plan-rule-audit-and-wrapper-
bugs]]: unioning a prohibition re-admits exactly what a positive rule just excluded).  It fires
only when a line card whose own prerequisite IS satisfied is also on the menu -- forbidding the
top of the line when there is nothing better to take would just push the pick onto a Trainer.

Opponent-independent by construction: it reads our own board and the menu, never theirs.
"""
import os

# ---------------------------------------------------------------- dusk_plan.py
p = "/root/ptcg/repo_sb/tools/dusk_plan.py"
s = open(p).read()

old = '''    "stadium_replace": ("do not play a Stadium onto an identical Stadium", 1.0),'''
new = '''    "stadium_replace": ("do not play a Stadium onto an identical Stadium", 1.0),
    "search_bottom": ("do not search out a card whose pre-evolution is not in play", 1.5),'''
assert s.count(old) == 1, "RULES anchor"
s = s.replace(old, new)

old = "ITCHY_POLLEN, PHANTOM_DIVE, JET_HEADBUTT = 323, 154, 153"
new = """ITCHY_POLLEN, PHANTOM_DIVE, JET_HEADBUTT = 323, 154, 153
# SelectContext::ToHand.  ToJson.h emits `(int)state.selectContext - 1`, so the enum's 8 arrives
# as 7 -- the same shift that makes the damage-counter menus (13, 14) line up with DamageCounter
# and DamageCounterAny, which is how this was checked rather than assumed.
_TO_HAND_CTX = 7
_PREREQ = {}          # filled below, once the line ids exist"""
assert s.count(old) == 1, "ctx anchor"
s = s.replace(old, new)

old = "_BASICS = (DREEPY, DUSKULL, BUDEW, MUNKIDORI, FEZ, MEOWTH)"
new = """_BASICS = (DREEPY, DUSKULL, BUDEW, MUNKIDORI, FEZ, MEOWTH)
# what each evolution needs UNDER it before it can ever be played
_PREREQ = {DRAKLOAK: DREEPY, PULT: DRAKLOAK, DUSCLOPS: DUSKULL, DUSKNOIR: DUSCLOPS}"""
assert s.count(old) == 1, "_BASICS anchor"
s = s.replace(old, new)

old = "    from lm.actions import encode_option\n"
new = "    from lm.actions import encode_option, _card_at\n"
assert s.count(old) == 1, "import anchor"
s = s.replace(old, new)

old = "    # --- benching / evolving ---------------------------------------------------------------"
new = '''    # --- deck searches that put the card in HAND ------------------------------------------
    # See the RULES entry. Poffin (context ToBench) already takes a Dreepy 94% of the time; this
    # is the other search path, and on our first turn with no Drakloak in play it takes the
    # Dreepy 5% of the time and Dragapult ex or Dusknoir instead.
    if sel.get("context") == _TO_HAND_CTX:
        bad, ready = set(), set()
        for i, o in enumerate(opts):
            if not isinstance(o, dict):
                continue
            cid = _card_at(o, obs)
            pre = _PREREQ.get(cid)
            if pre is not None and pre not in my_ids:
                bad.add(i)                       # unplayable until the step below it exists
            elif cid in _PREREQ or cid in (DREEPY, DUSKULL):
                ready.add(i)                     # a line card we could actually use
        # `ready` is the guard: forbidding the top of the line when nothing better is on the
        # menu would only push the pick onto a Trainer, which this rule has no opinion about.
        if bad and ready and len(bad) < len(opts):
            out["search_bottom"] = (set(range(len(opts))) - bad, set(range(len(opts))))

    # --- benching / evolving ---------------------------------------------------------------'''
assert s.count(old) == 1, "insert anchor"
s = s.replace(old, new)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)

# ---------------------------------------------------------------- lm/plan_filter.py
q = "/root/ptcg/repo_sb/lm/plan_filter.py"
t = open(q).read()
old = '''    "clops_hold",        # do not fire Dusclops' Cursed Blast while Dusknoir is in hand'''
new = '''    "clops_hold",        # do not fire Dusclops' Cursed Blast while Dusknoir is in hand
    "search_bottom",     # do not search out a card whose pre-evolution is not in play'''
assert t.count(old) == 1, "PROHIBITIONS anchor"
t = t.replace(old, new)
open(q + ".new", "w").write(t)
os.replace(q + ".new", q)

print("patched dusk_plan.py + plan_filter.py in repo_sb")
