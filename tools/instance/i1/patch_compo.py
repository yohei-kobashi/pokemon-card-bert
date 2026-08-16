"""What is actually occupying the board on our first turns?

Turn 1 shows 3.40 bodies but only 1.45 Dreepy and 0.57 Duskull -- so ~1.4 slots are held by
something else.  This list carries four basics that are not part of either evolution line
(Budew, Fezandipiti ex, Meowth ex, Munkidori), and every one of them competes for the same bench
space and the same Poffin.  Name them, because "we bench 3.4 bodies" and "we bench 3.4 of the
RIGHT bodies" are different findings with opposite conclusions.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = """                for _cid, _nm in ((DREEPY, "dreepy"), (DRAKLOAK, "drakloak"), (PULT, "pult"),
                                  (DUSKULL, "duskull"), (DUSCLOPS, "dusclops"),
                                  (DUSKNOIR, "dusknoir"), (BUDEW, "budew")):"""
new = """                for _cid, _nm in ((DREEPY, "dreepy"), (DRAKLOAK, "drakloak"), (PULT, "pult"),
                                  (DUSKULL, "duskull"), (DUSCLOPS, "dusclops"),
                                  (DUSKNOIR, "dusknoir"), (BUDEW, "budew"),
                                  (FEZ, "fez"), (MEOWTH, "meowth"), (MUNKIDORI, "munki")):"""
assert s.count(old) == 1, "species anchor"
s = s.replace(old, new)

old2 = "DREEPY, DRAKLOAK = 119, 120"
new2 = "DREEPY, DRAKLOAK = 119, 120\nFEZ, MEOWTH, MUNKIDORI = 140, 1071, 112"
assert s.count(old2) == 1, "const anchor"
s = s.replace(old2, new2)

old3 = '''        print("  %-6d %7.2f %8.2f %6.2f %8.2f %8.2f %8.2f %7.2f %8d"
              % (i, m("dreepy"), m("drakloak"), m("pult"), m("duskull"), m("dusclops"),
                 m("bodies"), m("energy"), len(rows)))'''
new3 = '''        print("  %-6d %7.2f %8.2f %6.2f %8.2f %8.2f %8.2f %7.2f %8d"
              % (i, m("dreepy"), m("drakloak"), m("pult"), m("duskull"), m("dusclops"),
                 m("bodies"), m("energy"), len(rows)))
        if i <= 3:
            print("         off-line basics: budew %.2f  fez %.2f  meowth %.2f  munkidori %.2f"
                  " -> %.2f of %.2f bodies"
                  % (m("budew"), m("fez"), m("meowth"), m("munki"),
                     m("budew") + m("fez") + m("meowth") + m("munki"), m("bodies")))'''
assert s.count(old3) == 1, "report anchor"
s = s.replace(old3, new3)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
