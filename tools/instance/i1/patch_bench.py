"""Is the thin Dreepy count a CHOICE or an AVAILABILITY problem?

1.45 Dreepy on our first turn against a human target of 2-3 has two possible causes with opposite
fixes.  If a Dreepy play was on the menu and we benched a Munkidori instead, that is a pilot
defect and `bench_line` is not being obeyed.  If Dreepy was never on the menu, no rule change can
help and the shortfall is draw, not decision.

`bench_line` is asked for its own verdict rather than re-derived: opportunities() returns
(conformant, scope), so "offered and refused" is exactly `picked & scope` without `picked & good`.

Worth knowing while reading the result: bench_line's conformant set is (DREEPY, DUSKULL, BUDEW) --
the rule's name says Dreepy but it scores a Budew as equally correct -- and WRAP_RULES carries
only prohibitions plus lethal_now, so bench_line never restricts the pilot at play time.  It
shapes training only.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = """            # --- the setup engine: offered vs played, per turn --------------------------"""
new = """            # --- bench_line: offered, and taken? ----------------------------------------
            try:
                _o2 = _plan.opportunities(obs)
            except Exception:
                _o2 = {}
            _bl = _o2.get("bench_line")
            if _bl:
                _good, _scope = _bl
                T["bench_turns"].add(key)
                if picked & set(_good):
                    T["bench_right"].add(key)
                elif picked & set(_scope):
                    T["bench_wrong"].add(key)
            # and the same question for Dreepy specifically, since bench_line counts a Budew
            # as conformant and the human template does not.
            _dre = [i for i, o in enumerate(opts)
                    if isinstance(o, dict) and texts[i].startswith("play")
                    and ("c%d" % DREEPY) in texts[i]]
            if _dre:
                T["dreepy_offered"].add(key)
                if picked & set(_dre):
                    T["dreepy_benched"].add(key)

            # --- the setup engine: offered vs played, per turn --------------------------"""
assert s.count(old) == 1, "watch anchor"
s = s.replace(old, new)

old2 = '''    print("\\n-- the setup engine, per turn (able = it was on the menu that turn) --")'''
new2 = '''    print("\\n-- bench_line: was the line offered, and did we take it? --")
    for _nm, _lbl in (("bench_turns", "turns bench_line was LIVE"),
                      ("bench_right", "... we benched Dreepy/Duskull/Budew"),
                      ("bench_wrong", "... we benched an off-line basic instead"),
                      ("dreepy_offered", "turns a DREEPY play was on the menu"),
                      ("dreepy_benched", "... and we benched it")):
        ks = T[_nm]
        print("  %-42s %5d   (our t1 %d, t2 %d, t3 %d)"
              % (_lbl, len(ks), sum(1 for k in ks if rank_of.get(k) == 2),
                 sum(1 for k in ks if rank_of.get(k) == 3),
                 sum(1 for k in ks if rank_of.get(k) == 4)))
    if len(T["dreepy_offered"]):
        print("  -> Dreepy taken %.0f%% of the turns it was offered"
              % (100.0 * len(T["dreepy_benched"]) / len(T["dreepy_offered"])))

    print("\\n-- the setup engine, per turn (able = it was on the menu that turn) --")'''
assert s.count(old2) == 1, "report anchor"
s = s.replace(old2, new2)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
