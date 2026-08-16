"""Separate SEARCH menus from every other menu a Dreepy can appear on.

The previous cut -- "a Dreepy is in the option text and the option is not a play" -- was too
coarse: what we picked "instead" was mostly c120/c121 at BENCH/ACTIVE, i.e. evolving that very
Dreepy into Drakloak or Dragapult, which is correct play, not a missed search.  Reading a 49%
rate off that would have been the third per-menu-style error in this investigation.

Group by the select context id instead.  The audit already relies on context (13, 14) being the
damage-counter menus, so the field is trustworthy; printing the breakdown says which contexts are
deck searches and lets the Dreepy rate be read inside each one separately.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = """            _dsr = [i for i, t in enumerate(texts)
                    if ("c%d" % DREEPY) in t and not t.startswith("play")]
            if _dsr:
                T["dreepy_search_able"].add(key)
                if picked & set(_dsr):
                    T["dreepy_search_took"].add(key)
                else:
                    for _i in picked:
                        if isinstance(_i, int) and 0 <= _i < len(texts):
                            search_instead[texts[_i].split(":")[-1]] += 1"""
new = """            _ctx = sel.get("context")
            _dsr = [i for i, t in enumerate(texts)
                    if ("c%d" % DREEPY) in t and not t.startswith("play")]
            if _dsr:
                ctx_able[_ctx] += 1
                if picked & set(_dsr):
                    ctx_took[_ctx] += 1
                else:
                    for _i in picked:
                        if isinstance(_i, int) and 0 <= _i < len(texts):
                            ctx_instead[(_ctx, texts[_i].split("@")[0])] += 1
            # every menu, so the context ids can be named by what they usually contain
            ctx_all[_ctx] += 1"""
assert s.count(old) == 1, "watch anchor"
s = s.replace(old, new)

old2 = "    search_instead = collections.Counter()\n"
new2 = ("    search_instead = collections.Counter()\n"
        "    ctx_able = collections.Counter(); ctx_took = collections.Counter()\n"
        "    ctx_instead = collections.Counter(); ctx_all = collections.Counter()\n")
assert s.count(old2) == 1, "state anchor"
s = s.replace(old2, new2)

old3 = '''    _sa, _st2 = T["dreepy_search_able"], T["dreepy_search_took"]'''
new3 = '''    print("\\n-- menus where a Dreepy was selectable, BY CONTEXT --")
    print("  %-8s %8s %8s %7s %6s   %s" % ("ctx", "menus", "dreepy", "took", "rate", "top pick instead"))
    for _c, _n in ctx_able.most_common(10):
        _inst = [("%s x%d" % (k[1], v)) for k, v in ctx_instead.most_common() if k[0] == _c][:3]
        print("  %-8s %8d %8d %7d %5.0f%%   %s"
              % (_c, ctx_all[_c], _n, ctx_took[_c],
                 100.0 * ctx_took[_c] / max(1, _n), ", ".join(_inst)))

    _sa, _st2 = T["dreepy_search_able"], T["dreepy_search_took"]'''
assert s.count(old3) == 1, "report anchor"
s = s.replace(old3, new3)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
