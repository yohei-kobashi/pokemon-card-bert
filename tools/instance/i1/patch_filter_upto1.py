"""plan_filter discards every "choose up to 1" menu, which is nearly every deck search.

    search_bottom fired 20 times in 8 games; 17 of those menus were minCount=0, maxCount=1
    and the non-strict path returned the FULL menu to the model.

The guard reads `if not (lo == 1 and hi == 1): return lm_agent(obs)` and its stated reason --
"rewriting min/maxCount would change what is being asked" -- is right for multi-pick (hi > 1)
and wrong for "up to 1": restricting the option list leaves min and max untouched, and declining
is still available.  So the rule fires, the filter throws it away, and the arm measures nothing.
This is why the first search_bottom A/B came back byte-identical on turns 1-4.

Behind SB_UPTO1 (default OFF, the shipped behaviour) because the fix does not only enable the
new rule -- it also lets the five EXISTING prohibitions act on menus they have never acted on.
That is a separate change with its own risk, so it gets its own arm rather than riding along.
"""
import os

p = "/root/ptcg/repo_sb/lm/plan_filter.py"
s = open(p).read()

old = """        lo = sel.get("minCount", 1) or 0
        hi = sel.get("maxCount", 1) or 1"""
new = """        lo = sel.get("minCount", 1) or 0
        hi = sel.get("maxCount", 1) or 1
        # "up to 1" (lo 0, hi 1) is the shape of nearly every deck search, and the exactly-one
        # test below rejected all of them -- so a prohibition could fire on a search and change
        # nothing. Restricting the OPTIONS of an up-to-1 menu does not touch min/max and does
        # not remove the right to decline, so the objection does not apply to it.
        _upto1 = os.environ.get("SB_UPTO1", "") not in ("", "0")"""
assert s.count(old) == 1, "lohi anchor"
s = s.replace(old, new)

old2 = """        if not (lo == 1 and hi == 1):
            return lm_agent(obs)
        keep = sorted(w)"""
new2 = """        if not (lo == 1 and hi == 1) and not (_upto1 and lo <= 1 and hi == 1):
            return lm_agent(obs)
        keep = sorted(w)"""
assert s.count(old2) == 1, "guard anchor"
s = s.replace(old2, new2)

old3 = """        pick = lm_agent(sub)
        if not pick or pick[0] >= len(keep):
            return [keep[0]]
        return [keep[pick[0]]]"""
new3 = """        pick = lm_agent(sub)
        if not pick:
            # On an up-to-1 menu an empty pick is a legal DECLINE, not a failure to answer;
            # forcing keep[0] there would invent a choice the model did not make.
            return pick if lo == 0 else [keep[0]]
        if pick[0] >= len(keep):
            return [keep[0]]
        return [keep[pick[0]]]"""
assert s.count(old3) == 1, "pick anchor"
s = s.replace(old3, new3)

if "\nimport os\n" not in s:
    s = s.replace('"""\n\n\n# Rules phrased', '"""\nimport os\n\n\n# Rules phrased', 1)
assert "\nimport os\n" in s, "import anchor"

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched")
