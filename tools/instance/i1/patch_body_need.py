"""Need-aware Pokemon search: while short of BODIES, a benchable Basic outranks everything.

`_card_need` already carries the right idea and the wrong magnitude:

    elif card.cardType == CardType.POKEMON and card.basic:
        if len(field) <= 2:
            s += 40

+40 cannot cross an explicit tier gap. `card_roles` tiers a deck's win-condition EVOLUTION as
`win` (900) and the Basic that evolves into it as `line` (600), so every "search your deck for a
Pokemon" fetched an UNPLAYABLE Stage 1 while the bench was empty. The audit finds that inversion
in 52 of 63 decks, all 52 of which run a Pokemon-search card
([[honchkrow-profile-was-net-negative]]).

So when we are genuinely short of bodies the boost must DOMINATE the tier table rather than nudge
inside it. 2000 sits above every _TIER_VALUE and below the two special returns in
decide_acquire (energy-attach target 10000, recover-a-Pokemon-from-discard 5000), so it changes
ranking only where it should.

ENGINE_BODY_NEED=0 disables it, so both arms run from one repo.
"""
import os

ENG = os.path.join(os.getcwd(), "agents/engine_v2.py")
s = open(ENG).read()

CONST_ANCHOR = '_USE_ROLES = True    # A/B switch for explicit per-deck card_roles\n'
CONST_NEW = CONST_ANCHOR + '''
# Short of BODIES, a benchable Basic must outrank the win-condition evolution it feeds --
# see BasePolicy._card_need. Above every _TIER_VALUE, below decide_acquire's 5000/10000
# special cases. ENGINE_BODY_NEED=0 turns it off for A/B.
_BODY_NEED = 2000 if os.environ.get("ENGINE_BODY_NEED", "1") != "0" else 0
'''

OLD = """        elif card.cardType == CardType.POKEMON and card.basic:
            # (3) a thin board loses to a single KO -- a body beats a spell
            if len(field) <= 2:
                s += 40
        return s
"""
NEW = """        elif card.cardType == CardType.POKEMON and card.basic:
            # (3) a thin board loses to a single KO -- a body beats a spell.
            #
            # The +40 below could never cross an EXPLICIT tier gap, and that is the common
            # case: card_roles tiers the win-condition evolution `win` (900) and the Basic
            # that evolves into it `line` (600), an inversion present in 52 of 63 decks (all
            # 52 run a Pokemon search). So "search for a Pokemon" fetched an unplayable
            # Stage 1 while the bench sat empty -- rockets_honchkrow lost 39.4% of its games
            # to bench-out that way. While we are actually short of bodies the need has to
            # DOMINATE the tier table; once the bench is developed the tiers take over again,
            # because fetching the win condition is then the right move.
            if len(ctx.me.bench) < self.bench_target:
                s += _BODY_NEED
            elif len(field) <= 2:
                s += 40
        return s
"""

if "_BODY_NEED" in s:
    print("already patched")
else:
    assert s.count(CONST_ANCHOR) == 1, "const anchor not unique"
    assert s.count(OLD) == 1, "body anchor not unique"
    s = s.replace(CONST_ANCHOR, CONST_NEW).replace(OLD, NEW)
    if "\nimport os" not in s.split("class ")[0]:
        print("WARNING: os may not be imported at module level -- check")
    open(ENG, "w").write(s)
    print("patched", ENG)
