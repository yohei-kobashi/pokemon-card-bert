"""Re-key the opening table by OUR turn ordinal instead of the engine's turn counter.

The first table had exactly 30 games at t1 and 30 at t2 (the seat split) but 46-47 at t3-t5, so
the engine's `turn` is not a clean per-player index and "turn 11" could mean our 6th turn or our
11th.  Rank the turns we actually acted on, per game, and the row label becomes unambiguous:
row N is OUR Nth turn.  The human template is stated in our-turns ("bench 3 Dreepy on turn 1"),
so this is the only version comparable to it.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = '''    print("\\n-- the opening, against the human template (t1: Dreepy x3 + Duskull x1-2) --")
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
'''
new = '''    by_game = collections.defaultdict(dict)
    for (g, tt), v in dev.items():
        by_game[g][tt] = v
    ordinal = collections.defaultdict(list)     # our Nth turn -> [board, ...]
    rank_of = {}                                # (game, engine turn) -> our Nth turn
    for g, d in by_game.items():
        for i, tt in enumerate(sorted(d), 1):
            ordinal[i].append(d[tt])
            rank_of[(g, tt)] = i
    print("\\n-- the opening, by OUR turn (human template: turn 1 = Dreepy x3 + Duskull x1-2) --")
    print("  %-6s %7s %8s %6s %8s %8s %8s %7s %8s"
          % ("ours", "dreepy", "drakloak", "pult", "duskull", "dusclops", "bodies", "energy", "games"))
    for i in range(1, 8):
        rows = ordinal.get(i) or []
        if not rows:
            continue
        m = lambda k: sum(r[k] for r in rows) / len(rows)
        print("  %-6d %7.2f %8.2f %6.2f %8.2f %8.2f %8.2f %7.2f %8d"
              % (i, m("dreepy"), m("drakloak"), m("pult"), m("duskull"), m("dusclops"),
                 m("bodies"), m("energy"), len(rows)))
    b2 = sum(1 for g, d in by_game.items()
             if any(v["budew_active"] for tt, v in d.items() if rank_of[(g, tt)] <= 2))
    print("  games with Budew ACTIVE by OUR turn 2: %d of %d" % (b2, a.games))
    if first_pd:
        import statistics as _s2
        fr = [rank_of.get((g, tt), tt) for g, tt in first_pd.items()]
        print("  first Phantom Dive: OUR turn %.1f mean / %d median, in %d of %d games"
              % (_s2.mean(fr), _s2.median(fr), len(fr), a.games))
    else:
        print("  first Phantom Dive: NEVER used in any of the %d games" % a.games)
'''
assert s.count(old) == 1, "report anchor"
s = s.replace(old, new)
t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
