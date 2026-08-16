"""Measure our OPENING against the human template.

The Japanese guides give an explicit target for this deck, not a vibe:
  * turn 1: bench Dreepy x3 and Duskull x1-2
  * turn 2: use Budew's Itchy Pollen if you can -- and the way you get Budew in front is to
    RETREAT by discarding energy (or with Latias's zero-cost retreat, which our list lacks)
  * "energy is extremely tight, attach as early and as many as you can"
  * use Drakloak's Recon Directive as often as possible; three Drakloak is the ideal

So the question "are we simply slower than ogerpon" has a checkable form: what IS on our board at
the end of turns 1, 2, 3, and how does it compare to 3 Dreepy + 1-2 Duskull?

Board state is taken as the MAXIMUM seen within a turn, keyed (game, turn) -- a board only grows
during a turn, and sampling per menu would weight long turns.  Nothing here touches dusk_plan.py
or plan_filter.py, so rules_fp is unchanged.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = "ITCHY_POLLEN = 323"
new = """DREEPY, DRAKLOAK = 119, 120
DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
ITCHY_POLLEN = 323"""
assert s.count(old) == 1, "const anchor"
s = s.replace(old, new, 1)

old2 = "    last_hp = {}\n"
new2 = "    last_hp = {}\n    dev = {}                        # (game, turn) -> max board seen that turn\n    first_pd = {}                   # game -> first turn Phantom Dive was actually used\n"
assert s.count(old2) == 1, "state anchor"
s = s.replace(old2, new2)

old3 = """            # --- Budew's lock, and what the plan's gate does to it ---------------------"""
new3 = """            # --- how fast the board actually comes up ----------------------------------
            if isinstance(turn, int):
                d = dev.setdefault((cur[0], turn), collections.Counter())
                ids = [(x or {}).get("id") for x in ma + mb]
                for _cid, _nm in ((DREEPY, "dreepy"), (DRAKLOAK, "drakloak"), (PULT, "pult"),
                                  (DUSKULL, "duskull"), (DUSCLOPS, "dusclops"),
                                  (DUSKNOIR, "dusknoir"), (BUDEW, "budew")):
                    d[_nm] = max(d[_nm], ids.count(_cid))
                d["energy"] = max(d["energy"], sum(_energy(x) for x in ma + mb))
                d["bodies"] = max(d["bodies"], len(ma) + len(mb))
                if ma and (ma[0] or {}).get("id") == BUDEW:
                    d["budew_active"] = 1
                if any(isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE
                       for i, o in enumerate(opts) if i in picked):
                    first_pd.setdefault(cur[0], turn)

            # --- Budew's lock, and what the plan's gate does to it ---------------------"""
assert s.count(old3) == 1, "dev anchor"
s = s.replace(old3, new3)

old4 = """    ig = len(T["itchy_turns"])"""
new4 = '''    print("\\n-- the opening, against the human template (t1: Dreepy x3 + Duskull x1-2) --")
    print("  %-5s %7s %8s %6s %8s %8s %8s %7s %8s"
          % ("turn", "dreepy", "drakloak", "pult", "duskull", "dusclops", "bodies", "energy", "games"))
    for t in range(1, 7):
        rows = [v for (g, tt), v in dev.items() if tt == t]
        if not rows:
            continue
        m = lambda k: sum(r[k] for r in rows) / len(rows)
        print("  %-5d %7.2f %8.2f %6.2f %8.2f %8.2f %8.2f %7.2f %8d"
              % (t, m("dreepy"), m("drakloak"), m("pult"), m("duskull"), m("dusclops"),
                 m("bodies"), m("energy"), len(rows)))
    b2 = sum(1 for (g, tt), v in dev.items() if tt <= 2 and v["budew_active"])
    print("  games with Budew ACTIVE by turn 2: %d of %d" % (b2, a.games))
    if first_pd:
        import statistics as _s2
        print("  first Phantom Dive: turn %.1f mean / %d median, in %d of %d games"
              % (_s2.mean(first_pd.values()), _s2.median(first_pd.values()),
                 len(first_pd), a.games))
    else:
        print("  first Phantom Dive: NEVER used in any of the %d games" % a.games)

    ig = len(T["itchy_turns"])'''
assert s.count(old4) == 1, "report anchor"
s = s.replace(old4, new4)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
