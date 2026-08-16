"""Name every select context we are actually asked about -- and check the one Poffin should raise.

Traced through the engine source: Buddy-Buddy Poffin is `.effectDeckToBenchAndShuffle(2)`, which
is `EffectType::ToBench`, which CreateCard.h maps to `SelectContext::ToBench` = 6.  Dreepy is
60 HP so it always qualifies for Poffin's "70 HP or less".  Poffin is played ~100 times in 60
games -- yet context 6 did not appear once in the previous table.

Either the enum indices are not what the observation reports, or the pilot is never asked what
Poffin fetches.  Those have completely different consequences, so print the full context census
with names rather than guessing.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = '''    print("\\n-- menus where a Dreepy was selectable, BY CONTEXT --")'''
new = '''    CTXNAME = ("None Main SetupActivePokemon SetupBenchPokemon Switch ToActive ToBench ToField "
               "ToHand Discard ToDeck ToDeckBottom ToPrize NotMove DamageCounter DamageCounterAny "
               "Damage RemoveDamageCounter Heal EvolvesFrom EvolvesTo Devolve AttachFrom AttachTo "
               "DetachFrom Look EffectTarget DiscardEnergyCard DiscardToolCard SwitchEnergyCard "
               "DiscardCardOrAttachedCard DiscardEnergy ToHandEnergy ToDeckEnergy SwitchEnergy "
               "SkillOrder Attack DisableAttack Evolve DrawCount DamageCounterCount "
               "RemoveDamageCounterCount IsFirst Mulligan Activate FirstEffect MoreDevolve "
               "CoinHead AffectSpecialCondition RecoverSpecialCondition").split()
    _nm_of = lambda c: (CTXNAME[c] if isinstance(c, int) and 0 <= c < len(CTXNAME) else str(c))
    print("\\n-- every select context we were asked about (census) --")
    print("  %-4s %-24s %8s %8s %7s" % ("ctx", "name", "menus", "dreepy", "took"))
    for _c, _n in ctx_all.most_common(25):
        print("  %-4s %-24s %8d %8d %7d"
              % (_c, _nm_of(_c), _n, ctx_able[_c], ctx_took[_c]))
    if not ctx_all.get(6):
        print("  !! context 6 (ToBench) NEVER appeared -- Buddy-Buddy Poffin resolves its own")
        print("     search without ever asking the pilot which basics to fetch.")

    print("\\n-- menus where a Dreepy was selectable, BY CONTEXT --")'''
assert s.count(old) == 1, "report anchor"
s = s.replace(old, new)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
