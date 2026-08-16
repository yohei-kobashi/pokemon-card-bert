import os

p = "/root/ptcg/repo/tools/dusk_plan.py"
s = open(p).read()

# --- a reverse evolution index, built once ---------------------------------------------------
helper = '''
_EVOLVABLE = None


def _evolvable_names():
    """Names that SOMETHING evolves from -- i.e. bodies still developing into a real attacker.

    Cursed Blast is worth spending on Ralts before it is Gardevoir ex and on Abra before it is
    Alakazam; it is not worth spending on a body that is already everything it will ever be.
    `evolvesFrom` carries the pre-evolution NAME, so the set of "can still grow" cards is just
    the image of that field over the whole card DB.
    """
    global _EVOLVABLE
    if _EVOLVABLE is None:
        from agents._engine import _CARDS as _CD
        _EVOLVABLE = {getattr(c, "evolvesFrom", None) for c in _CD.values()
                      if getattr(c, "evolvesFrom", None)}
    return _EVOLVABLE

'''
anchor = "def _field_id(obs, yi, o):"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + "\n" + anchor, 1)

# --- replace the clops_hold block -------------------------------------------------------------
old_start = "    hand_ids = [h.get(\"id\") if isinstance(h, dict) else h for h in (me.get(\"hand\") or [])]\n"
old_end = "        out[\"clops_hold\"] = (set(range(len(opts))) - clops_fire, set(range(len(opts))))\n"
i = s.index(old_start)
j = s.index(old_end) + len(old_end)

NEW = '''    hand_ids = [h.get("id") if isinstance(h, dict) else h for h in (me.get("hand") or [])]
    if _CLOPS_HOLD:
        # WHEN IS CURSED BLAST WORTH A PRIZE? Every published guide for this deck says the same
        # three things, and the shipped rule encoded only the first:
        #   * hold Dusclops while Dusknoir is in hand -- same prize, 50 damage instead of 130
        #   * Dusknoir is the LATE FINISHER: 200 + 130 = 330 takes a 320-330 HP ex in one turn,
        #     conceding 1 prize to take 2
        #   * the other legitimate use is removing a BENCH body that is still growing into a
        #     threat (Ralts before Gardevoir, Abra before Alakazam) -- and the guides condition
        #     that on the prize race being level or in our favour, because it trades 1 for 1
        # Anything else is a free prize. Hareruya describes the whole game as roughly three
        # Phantom Dives and ONE Cursed Blast; this rule is what makes it one.
        #
        # Our list is more constrained than the reference lists, which is why the bar is set
        # here rather than left to taste: those run Counter Catcher / Iono / Roxanne, so the
        # conceded prize turns on a comeback. Ours runs none of them, and both Unfair Stamp and
        # Fezandipiti ex require a knock-out on the OPPONENT's turn, so a self-KO on our own turn
        # switches nothing on. For us the prize is pure cost.
        #
        # lethal_now returns early and alone, so a blast that WINS is never reached by this rule.
        _blast = {}                     # option index -> counters it would place
        for _bi, _bo, _bt in texts():
            if _bt.startswith("ability:c%d" % DUSCLOPS):
                _blast[_bi] = 5
            elif _bt.startswith("ability:c%d" % DUSKNOIR):
                _blast[_bi] = 13
        if _blast:
            from agents._engine import _CARDS as _CD6
            _opp_pz = len(opp.get("prize") or [])
            _my_pz = len(me.get("prize") or [])
            _oact = (opp.get("active") or [None])[0]
            _obench = [b for b in (opp.get("bench") or []) if isinstance(b, dict)]
            # Phantom Dive has to actually be reachable this turn for the combo test to mean
            # anything: either it is on this menu, or Dragapult is Active and paid for.
            _pd_ready = any(isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE
                            for o in opts)
            if not _pd_ready:
                _ma = (me.get("active") or [None])[0]
                _pd_ready = (isinstance(_ma, dict) and _ma.get("id") == PULT and _can_pd(_ma))

            def _worth(counters):
                dmg = counters * 10
                # A prize handed over when they need one is the game, whatever it buys.
                if _opp_pz <= 1:
                    return False
                for _t in ([_oact] if isinstance(_oact, dict) else []) + _obench:
                    _hp = _t.get("hp") or 0
                    _c = _CD6.get(_t.get("id"))
                    if _hp <= 0:
                        continue
                    # (1) it kills something worth two prizes: 1 conceded for 2 taken
                    if dmg >= _hp and (_prizes_for(_c) or 1) >= 2:
                        return True
                    # (2) it kills a still-developing BENCH body, at level or better prizes
                    if (dmg >= _hp and _t is not _oact and _my_pz <= _opp_pz
                            and getattr(_c, "name", None) in _evolvable_names()):
                        return True
                    # (3) it is the half that lets Phantom Dive finish the job THIS turn --
                    #     200 to the Active, or the six counters to a bench body
                    if _pd_ready and dmg < _hp:
                        _reach = PD_DMG if _t is _oact else PD_COUNTERS
                        if _hp - dmg <= _reach:
                            return True
                return False

            _bad = set()
            for _bi, _cnt in _blast.items():
                # the upgrade guard, unchanged and first: firing the 5 while the 13 is in hand
                # spends the body for under half its value at identical cost
                if _cnt == 5 and DUSKNOIR in hand_ids and DUSCLOPS in my_ids:
                    _bad.add(_bi)
                elif not _worth(_cnt):
                    _bad.add(_bi)
            if _bad and len(_bad) < len(opts):
                out["clops_hold"] = (set(range(len(opts))) - _bad, set(range(len(opts))))
'''

s = s[:i] + NEW + s[j:]
t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("clops_hold extended: upgrade guard + 2-prize kill / developing-bench kill / PD combo,")
print("                     and never while the opponent is on 1 prize")
