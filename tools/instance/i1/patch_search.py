"""Dreepy is refused 24-30% of the time it is on a menu -- but which menu?

Benching from HAND is nearly clean: bench_line is live on 1.6 turns a game and obeyed 94%.  Yet
Buddy-Buddy Poffin is played on 77% of our first turns and the Dreepy count is still 1.45, so the
loss is not in the bench decision.  Poffin does not put cards in hand -- it pulls two basics from
the DECK -- so its choice shows up as a search/sub-select menu, never as `play:c119`.

Measure that menu directly: when a Dreepy is among the searchable options, do we take it, and
when we do not, what did we take instead?  Blind sub-selection has bitten this engine before
([[engine-blind-selection-bugs]], [[subselect-fallback-audit]]), and it is invisible to every
rule in dusk_plan.py, none of which has a SEARCH scope.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = """            # --- the setup engine: offered vs played, per turn --------------------------"""
new = """            # --- search menus: is Dreepy being passed over in the DECK, not the hand? ---
            _dsr = [i for i, t in enumerate(texts)
                    if ("c%d" % DREEPY) in t and not t.startswith("play")]
            if _dsr:
                T["dreepy_search_able"].add(key)
                if picked & set(_dsr):
                    T["dreepy_search_took"].add(key)
                else:
                    for _i in picked:
                        if isinstance(_i, int) and 0 <= _i < len(texts):
                            search_instead[texts[_i].split(":")[-1]] += 1

            # --- the setup engine: offered vs played, per turn --------------------------"""
assert s.count(old) == 1, "watch anchor"
s = s.replace(old, new)

old2 = "    attacks_used = collections.Counter()\n"
new2 = "    attacks_used = collections.Counter()\n    search_instead = collections.Counter()\n"
assert s.count(old2) == 1, "state anchor"
s = s.replace(old2, new2)

old3 = '''    print("\\n-- bench_line: was the line offered, and did we take it? --")'''
new3 = '''    _sa, _st2 = T["dreepy_search_able"], T["dreepy_search_took"]
    print("\\n-- SEARCH menus (Poffin / Ultra Ball / Pad pull from the DECK, not the hand) --")
    print("  turns a Dreepy was among the searchable options  %5d   (%.2f per game)"
          % (len(_sa), len(_sa) / max(1, a.games)))
    print("  ... and we took it                               %5d   %4.0f%%"
          % (len(_st2), 100.0 * len(_st2) / max(1, len(_sa))))
    print("  when we did not, what we took instead (top 12):")
    for _cid, _n in search_instead.most_common(12):
        print("      %-10s %d" % (_cid, _n))

    print("\\n-- bench_line: was the line offered, and did we take it? --")'''
assert s.count(old3) == 1, "report anchor"
s = s.replace(old3, new3)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
