"""Is the opening slow because the deck cannot, or because the pilot does not?

The human template -- bench 3 Dreepy plus a Duskull on turn 1 -- is not aspirational for this
list: it runs Buddy-Buddy Poffin x4, which puts two 70-HP-or-less basics from the deck straight
onto the bench, plus Ultra Ball x4 and Poke Pad x4.  So measure the setup cards the same way as
the disruption ones: per TURN, opportunity in the denominator, keyed to OUR turn ordinal.

If Poffin is played on nearly every turn it is holdable, the opening is deck-limited and there is
nothing for a pilot to fix.  If it sits in hand, this is the dead-card class again
([[dead-trainer-buckets-audit]], [[dead-cards-audit]]) and it is the largest single lever left.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = """            # --- how fast the board actually comes up ----------------------------------"""
new = """            # --- the setup engine: offered vs played, per turn --------------------------
            for _cid, _nm in ((1086, "poffin"), (1121, "ultra"), (1227, "lillie"),
                              (1198, "crispin"), (1152, "pad")):
                _s = [i for i, t in enumerate(texts) if ("c%d" % _cid) in t]
                if _s:
                    T["su_%s_able" % _nm].add(key)
                    if picked & set(_s):
                        T["su_%s_play" % _nm].add(key)

            # --- how fast the board actually comes up ----------------------------------"""
assert s.count(old) == 1, "watch anchor"
s = s.replace(old, new)

old2 = '''    ig = len(T["itchy_turns"])'''
new2 = '''    print("\\n-- the setup engine, per turn (able = it was on the menu that turn) --")
    print("  %-9s %6s %7s %6s   %s" % ("card", "able", "played", "rate", "of which on OUR turn 1 / 2"))
    for _nm, _lbl in (("poffin", "poffin"), ("ultra", "ultra ball"), ("lillie", "lillie"),
                      ("crispin", "crispin"), ("pad", "poke pad")):
        able, play = T["su_%s_able" % _nm], T["su_%s_play" % _nm]
        t1a = sum(1 for k in able if rank_of.get(k) == 1)
        t1p = sum(1 for k in play if rank_of.get(k) == 1)
        t2a = sum(1 for k in able if rank_of.get(k) == 2)
        t2p = sum(1 for k in play if rank_of.get(k) == 2)
        print("  %-9s %6d %7d %5.0f%%   t1 %d/%d   t2 %d/%d"
              % (_lbl, len(able), len(play), 100.0 * len(play) / max(1, len(able)),
                 t1p, t1a, t2p, t2a))

    ig = len(T["itchy_turns"])'''
assert s.count(old2) == 1, "report anchor"
s = s.replace(old2, new2)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
