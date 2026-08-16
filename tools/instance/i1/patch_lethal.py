import os

p = "/root/ptcg/repo/tools/dusk_plan.py"
s = open(p).read()

start_mark = "    if _NEW_EXCL:\n        from agents._engine import _CARDS as _CD3\n"
end_mark = '                    return {"lethal_now": (kill, set(range(len(opts))))}\n'
i = s.index(start_mark)
j = s.index(end_mark) + len(end_mark)

NEW = '''    if _NEW_EXCL:
        from agents._engine import _CARDS as _CD3
        # EVERY way this deck can take its last prizes this turn, not just Phantom Dive.
        #
        # The first version credited bench knock-outs only to Phantom Dive's six counters. That
        # is one of four sources: Dusclops' Cursed Blast puts 5 counters on ANY one of the
        # opponent's Pokemon, Dusknoir's puts 13, and Munkidori's Adrena-Brain moves up to 3 from
        # one of ours to one of theirs. All three are ABILITIES -- they do not end the turn, so
        # they chain with each other and with the attack, and a line like "Dusknoir's 130 kills a
        # benched Kadabra, then attack the Active" was invisible to a rule that only read the
        # attack menu.
        #
        # Two costs the search has to respect:
        #   * Cursed Blast KNOCKS OUT THE USER, handing the opponent a prize. A line that brings
        #     them to zero loses the game before ours resolves, so it is refused outright.
        #   * An attack ends the turn. When a winning line needs an ability first, only the
        #     ABILITY is nominated; the attack half is nominated on the next menu, once the
        #     counters have actually landed and the board says so.
        _need = len(me.get("prize") or [])
        _opp_prizes = len(opp.get("prize") or [])
        _oa = (opp.get("active") or [None])[0]
        if _need and isinstance(_oa, dict):
            _def = _CD3.get(_oa.get("id"))
            # targets: index 0 is the Active, 1.. are the bench, as (hp, prizes it yields)
            _tgt = [((_oa.get("hp") or 0), _prizes_for(_def))]
            for _b in (opp.get("bench") or []):
                if isinstance(_b, dict):
                    _tgt.append(((_b.get("hp") or 0), _prizes_for(_CD3.get(_b.get("id")))))

            # counter packets offered on THIS menu: (option index, counters, prizes conceded)
            _packs = []
            for _pi, _po, _pt in texts():
                if not (_pt.startswith("abl") or _pt.startswith("ability")):
                    continue
                _who = _field_id(obs, yi, _po)
                if _who == DUSKNOIR:
                    _packs.append((_pi, 13, _prizes_for(_CD3.get(DUSKNOIR)) or 1))
                elif _who == DUSCLOPS:
                    _packs.append((_pi, 5, _prizes_for(_CD3.get(DUSCLOPS)) or 1))
                elif _who == MUNKIDORI:
                    # Adrena-Brain MOVES counters, so it can only supply what is already on one
                    # of ours -- an undamaged board makes this ability worth zero here.
                    _mv = 0
                    for _q in mine:
                        if isinstance(_q, dict):
                            _mv = max(_mv, min(3, ((_q.get("maxHp") or 0)
                                                   - (_q.get("hp") or 0)) // 10))
                    if _mv:
                        _packs.append((_pi, _mv, 0))

            _myact = (me.get("active") or [None])[0]
            _att = _CD3.get((_myact or {}).get("id")) if isinstance(_myact, dict) else None
            _atks = []          # (option index, damage on the Active, spreads 6 to the bench)
            for _ai, _ao in enumerate(opts):
                if not isinstance(_ao, dict) or not _ao.get("attackId"):
                    continue
                _d = _attack_damage(_ao["attackId"])
                if not _d:
                    continue
                if _att is not None and _def is not None:
                    if getattr(_def, "weakness", None) == getattr(_att, "energyType", None):
                        _d *= 2
                    elif getattr(_def, "resistance", None) == getattr(_att, "energyType", None):
                        _d -= 30
                _atks.append((_ai, _d, _ao["attackId"] == PHANTOM_DIVE))

            def _spread(res, cap):
                """Best prizes from `cap` counters spread over residual bench bodies. Exact:
                at most five bodies, so brute force over subsets costs nothing."""
                import itertools as _it
                items = [(((h + 9) // 10), pz) for h, pz in res if h > 0]
                best = 0
                for _r in range(1, len(items) + 1):
                    for _c in _it.combinations(items, _r):
                        if sum(x for x, _v in _c) <= cap:
                            best = max(best, sum(v for _x, v in _c))
                return best

            # Search: which abilities to fire, where each lands, and which attack to follow with.
            # <= 3 packets x 6 targets x <= 5 attacks is a few thousand states.
            import itertools as _it2
            _win_abl, _win_atk = set(), set()
            for _use in range(1 << len(_packs)):
                _chosen = [_packs[k] for k in range(len(_packs)) if _use >> k & 1]
                _conceded = sum(c for _i, _n, c in _chosen)
                # a self-KO that empties their prize count wins the game for them, not us
                if _conceded and _conceded >= _opp_prizes:
                    continue
                for _where in _it2.product(range(len(_tgt)), repeat=len(_chosen)):
                    _hp = [h for h, _pz in _tgt]
                    for (_oi, _cnt, _c), _w in zip(_chosen, _where):
                        _hp[_w] -= _cnt * 10
                    _got = sum(_tgt[t][1] for t in range(len(_tgt))
                               if _tgt[t][0] > 0 >= _hp[t])
                    if _got >= _need:
                        # abilities alone close it -- nominate them and never spend the turn
                        _win_abl.update(_oi for _oi, _n, _c in _chosen)
                        continue
                    for _ai, _d, _is_pd in _atks:
                        _tot = _got
                        if _hp[0] > 0 and _d >= _hp[0]:
                            _tot += _tgt[0][1]
                        if _is_pd:
                            _tot += _spread([(_hp[t], _tgt[t][1])
                                             for t in range(1, len(_tgt))
                                             if _tgt[t][0] > 0 < _hp[t]], 6)
                        if _tot >= _need:
                            if _chosen:
                                _win_abl.update(_oi for _oi, _n, _c in _chosen)
                            else:
                                _win_atk.add(_ai)
            # An ability-first line is preferred whenever one exists: it does not end the turn,
            # so taking it keeps every attack still on the table for the next menu.
            kill = _win_abl or _win_atk
            if kill and len(kill) < len(opts):
                return {"lethal_now": (kill, set(range(len(opts))))}
'''

s = s[:i] + NEW + s[j:]
t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("lethal_now: abilities (Dusclops 5 / Dusknoir 13 / Munkidori 3) + Phantom Dive + attack,")
print("            with a self-KO guard and ability-before-attack nomination")
