"""Measure the one setup decision no rule has ever touched: first or second.

Context 41 (SelectContext::IsFirst, shifted) fires 30 times in 60 games -- once whenever we win
the coin flip -- and dusk_plan has no rule for it.  The guides treat it as a real trade-off with
a metagame answer, not a default:

    "先攻だと早期の「ファントムダイブ」で相手を壊滅させやすく、
     後攻だと最初の番から「むずむずかふん」できます"

Going first buys a turn of development toward the attack; going second buys Budew's item lock a
turn earlier, on the turn the opponent is setting up.  Both are plausible for this list, which is
exactly why it is worth measuring rather than assuming.

Also counts the other high-frequency contexts that no rule covers (Crispin's attach-or-hand split,
retreat discards, draw-count choices), because their frequency is the argument for looking at
them next.
"""
import os

p = "/root/ptcg/repo_sb/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = "    game_won = {}                   # game -> 1/0, joined to `dev` by the game index\n"
new = ("    game_won = {}                   # game -> 1/0, joined to `dev` by the game index\n"
       "    first_choice = {}               # game -> what we answered when asked to choose\n"
       "    unruled = collections.Counter()  # contexts with no plan rule, by name\n")
assert s.count(old) == 1, "state anchor"
s = s.replace(old, new)

old2 = """            _ctx = sel.get("context")"""
new2 = """            _ctx = sel.get("context")
            # 41 == SelectContext::IsFirst (the JSON reports enum-1). The menu is yes/no; the
            # engine asks the coin-flip winner whether to take the first turn.
            if _ctx == 41 and cur[0] not in first_choice:
                _txt = [t for i, t in enumerate(texts) if i in picked]
                first_choice[cur[0]] = (_txt[0] if _txt else "?")
            if _ctx in (22, 30, 33, 38):
                unruled[{22: "AttachTo", 30: "DiscardEnergy",
                         33: "SwitchEnergy(Crispin)", 38: "DrawCount"}[_ctx]] += 1"""
assert s.count(old2) == 1, "watch anchor"
s = s.replace(old2, new2)

old3 = '''    print("\\n-- when could we have attacked'''
new3 = '''    print("\\n-- first or second, the decision no rule covers --")
    if first_choice:
        import collections as _c
        by = _c.defaultdict(list)
        for g, ans in first_choice.items():
            if g in game_won:
                by[ans].append(game_won[g])
        print("  asked in %d of %d games" % (len(first_choice), a.games))
        for ans, v in sorted(by.items()):
            print("    answered %-5s %3d games, won %3d (%5.1f%%)"
                  % (ans, len(v), sum(v), 100.0 * sum(v) / len(v)))
        rest = [game_won[g] for g in game_won if g not in first_choice]
        if rest:
            print("    (never asked  %3d games, won %3d (%5.1f%%) -- the flip went to them)"
                  % (len(rest), sum(rest), 100.0 * sum(rest) / len(rest)))
    else:
        print("  NEVER ASKED in %d games" % a.games)
    print("  other contexts with no plan rule, per %d games: %s"
          % (a.games, dict(unruled.most_common())))

    print("\\n-- when could we have attacked'''
assert s.count(old3) == 1, "report anchor"
s = s.replace(old3, new3)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched")
