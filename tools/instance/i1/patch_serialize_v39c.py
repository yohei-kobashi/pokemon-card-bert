"""Two fixes to the v39 patch, both caught by rendering a real state before/after.

1. `deck_mode="roles"` skipped the remaining-library subtraction, because that branch was gated
   on `mode == "remaining"` alone -- so roles rendered the original 60 (c891x3) where v37
   rendered what is actually still gettable (c891x2). Role grouping is meant to sit ON TOP of
   the remaining library, not replace it: the whole reason `remaining` exists is that a fixed
   60-card list is memorisable as a fingerprint and 70-85% of it already appears elsewhere in
   the state.

2. `identify="op"` was never wired (a heredoc escaping error), so ID ME still rendered.
"""
import os

P = os.path.join(os.getcwd(), "lm/serialize.py")
s = open(P).read()

OLD = '    if mode == "remaining" and obs is not None:'
NEW = '    if mode in ("remaining", "roles") and obs is not None:'
if OLD in s:
    s = s.replace(OLD, NEW)
    print("fixed: roles now subtracts the known cards too")
else:
    print("remaining-gate already fixed")

A = "{_identify(obs, yi, deck_name)}"
B = '{_identify(obs, yi, None if identify == "op" else deck_name)}'
if A in s:
    assert s.count(A) == 1, "identify anchor not unique"
    s = s.replace(A, B)
    print("fixed: identify='op' drops ID ME")
else:
    print("identify already wired")

open(P, "w").write(s)
