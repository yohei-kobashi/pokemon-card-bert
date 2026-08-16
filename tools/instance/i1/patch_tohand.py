"""Name the contexts correctly (the JSON reports enum-1) and split the ToHand search by turn.

ToJson.h:206 emits `(int)state.selectContext - 1`, so an observed 5 is SelectContext::ToBench and
an observed 7 is ToHand.  That shift is independently confirmed: this audit has always used
(13, 14) for the damage-counter menus and those readings were validated against CalcDamage --
under the shift they are DamageCounter and DamageCounterAny exactly.

With the names right, the earlier table reads:

  observed 5  ToBench   (Buddy-Buddy Poffin)   69 Dreepy-bearing menus, taken 94%   <- fine
  observed 7  ToHand    (Ultra Ball / Poke Pad / Night Stretcher)  217, taken 19%   <- here

In four ToHand searches out of five we pass over a Dreepy and take Dragapult ex (68) or Dusknoir
(51) instead -- the top of a line whose bottom is not on the board yet.  But "wrong" depends on
WHEN: fetching Dragapult ex on turn 6 with two Drakloak up is correct play.  Split by our turn
ordinal, and also record whether a Drakloak existed at that moment, because that is the condition
a rule would actually key on -- and it needs no read on the opponent at all.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

# 1. correct the census names
old = "    _nm_of = lambda c: (CTXNAME[c] if isinstance(c, int) and 0 <= c < len(CTXNAME) else str(c))"
new = ("    # ToJson.h emits selectContext-1, so shift back before naming\n"
       "    _nm_of = lambda c: (CTXNAME[c + 1] if isinstance(c, int) and -1 <= c < len(CTXNAME) - 1\n"
       "                        else str(c))")
assert s.count(old) == 1, "name anchor"
s = s.replace(old, new)
s = s.replace('''    if not ctx_all.get(6):
        print("  !! context 6 (ToBench) NEVER appeared -- Buddy-Buddy Poffin resolves its own")
        print("     search without ever asking the pilot which basics to fetch.")
''', "")

# 2. record the ToHand searches with enough context to design a rule
old2 = """            _ctx = sel.get("context")"""
new2 = """            _ctx = sel.get("context")
            # observed 7 == SelectContext::ToHand: Ultra Ball, Poke Pad, Night Stretcher.
            if _ctx == 7:
                _dre7 = [i for i, t in enumerate(texts) if ("c%d" % DREEPY) in t]
                if _dre7:
                    _have_dk = DRAKLOAK in [(x or {}).get("id") for x in ma + mb]
                    _took = bool(picked & set(_dre7))
                    tohand[(key, "able")] = (_have_dk, _took)
                    for _i in picked:
                        if isinstance(_i, int) and 0 <= _i < len(texts):
                            tohand_pick[(_have_dk, texts[_i].split("@")[0])] += 1"""
assert s.count(old2) == 1, "watch anchor"
s = s.replace(old2, new2)

old3 = "    ctx_instead = collections.Counter(); ctx_all = collections.Counter()\n"
new3 = ("    ctx_instead = collections.Counter(); ctx_all = collections.Counter()\n"
        "    tohand = {}; tohand_pick = collections.Counter()\n")
assert s.count(old3) == 1, "state anchor"
s = s.replace(old3, new3)

old4 = '''    print("\\n-- menus where a Dreepy was selectable, BY CONTEXT --")'''
new4 = '''    print("\\n-- ToHand searches (Ultra Ball / Poke Pad / Night Stretcher) with a Dreepy on offer --")
    print("  %-8s %8s %8s %7s   %s" % ("our turn", "menus", "took", "rate", "have a Drakloak already?"))
    for _lbl, _want in (("no Drakloak yet", False), ("Drakloak in play", True)):
        for _t in (1, 2, 3, 4, 5, "6+"):
            rows = [(hd, tk) for (k, _), (hd, tk) in tohand.items()
                    if hd == _want and ((rank_of.get(k, 99) - 1) == _t
                                        or (_t == "6+" and rank_of.get(k, 99) - 1 >= 6))]
            if not rows:
                continue
            took = sum(1 for _h, t in rows if t)
            print("  %-8s %8d %8d %6.0f%%   %s"
                  % (_t, len(rows), took, 100.0 * took / len(rows), _lbl))
    print("  what we took instead, by whether a Drakloak was already in play (top 8 each):")
    for _want in (False, True):
        top = [("%s x%d" % (k[1], v)) for k, v in tohand_pick.most_common() if k[0] == _want][:8]
        print("    %-16s %s" % ("no Drakloak:" if not _want else "have Drakloak:", ", ".join(top)))

    print("\\n-- menus where a Dreepy was selectable, BY CONTEXT --")'''
assert s.count(old4) == 1, "report anchor"
s = s.replace(old4, new4)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
