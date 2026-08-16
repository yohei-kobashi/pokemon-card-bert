import os

p = "/root/ptcg/repo/tools/dusk_plan.py"
s = open(p).read()

# 1. A board wipe is a SECOND win condition, independent of the prize count: knock out every
#    Pokemon the opponent has in play and there is nothing to promote, so the game ends whatever
#    the prizes say. The first version counted prizes only, so "Phantom Dive kills the Active and
#    its six counters clear a one-body bench" was invisible whenever we still needed 3 prizes.
old = '''            def _spread(res, cap):
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
'''
new = '''            def _spread(res, cap):
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

            def _clears(res, cap):
                """Can `cap` counters finish EVERY surviving bench body? A different question
                from _spread, which maximises prizes and will happily leave a body standing."""
                return sum(((h + 9) // 10) for h in res if h > 0) <= cap

            # Our own board must survive the plan too: Cursed Blast knocks the user out, and
            # emptying our OWN bench loses the game just as surely as emptying theirs wins it.
            _my_bodies = sum(1 for _q in mine if isinstance(_q, dict))
'''
assert s.count(old) == 1, ("spread", s.count(old))
s = s.replace(old, new)

# 2. the self-KO count must not empty our own board
old2 = '''                if _conceded and _conceded >= _opp_prizes:
                    continue'''
new2 = '''                if _conceded and _conceded >= _opp_prizes:
                    continue
                _selfko = sum(1 for _i, _n, c in _chosen if c)
                if _my_bodies - _selfko < 1:
                    continue           # the last Cursed Blast would empty our own board'''
assert s.count(old2) == 1, ("selfko", s.count(old2))
s = s.replace(old2, new2)

# 3. evaluate BOTH win conditions -- prizes taken, and nothing of theirs left in play
old3 = '''                    _got = sum(_tgt[t][1] for t in range(len(_tgt))
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
                                _win_atk.add(_ai)'''
new3 = '''                    _got = sum(_tgt[t][1] for t in range(len(_tgt))
                               if _tgt[t][0] > 0 >= _hp[t])
                    _alive = [t for t in range(len(_tgt)) if _hp[t] > 0]
                    if _got >= _need or not _alive:
                        # abilities alone close it, on prizes or by clearing their board --
                        # nominate them and never spend the turn
                        _win_abl.update(_oi for _oi, _n, _c in _chosen)
                        continue
                    for _ai, _d, _is_pd in _atks:
                        _tot = _got
                        _act_dead = _hp[0] <= 0 or _d >= _hp[0] > 0
                        if _hp[0] > 0 and _d >= _hp[0]:
                            _tot += _tgt[0][1]
                        _bench_res = [_hp[t] for t in range(1, len(_tgt))]
                        if _is_pd:
                            _tot += _spread([(_hp[t], _tgt[t][1])
                                             for t in range(1, len(_tgt))
                                             if _tgt[t][0] > 0 < _hp[t]], 6)
                        # WIN CONDITION 2: they have nothing left to promote. The attack must
                        # kill the Active, and every surviving bench body must fall to the six
                        # counters -- which is _clears, not _spread: maximising prizes can leave
                        # exactly the one body that keeps them alive.
                        _wipe = _act_dead and (_clears(_bench_res, 6) if _is_pd
                                               else not any(h > 0 for h in _bench_res))
                        if _tot >= _need or _wipe:
                            if _chosen:
                                _win_abl.update(_oi for _oi, _n, _c in _chosen)
                            else:
                                _win_atk.add(_ai)'''
assert s.count(old3) == 1, ("win", s.count(old3))
s = s.replace(old3, new3)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("lethal_now: board-wipe win added; self-KO no longer allowed to empty our own board")
