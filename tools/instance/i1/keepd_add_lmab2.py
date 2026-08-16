#!/usr/bin/env python3
"""Add lmab2 to keepd's watch list, same pattern as lmab (temp file + os.replace)."""
import os
import sys

P = "/root/keepd.sh"
src = open(P).read()
if "lmab2" in src:
    print("already patched")
    sys.exit(0)

old_for = "for W in hole_adopt restart_at_boundary lmab; do"
new_for = "for W in hole_adopt restart_at_boundary lmab lmab2; do"
old_case = "            lmab) grep -aq LMAB_DONE /root/lmab.log 2>/dev/null && continue ;;"
new_case = old_case + (
    "\n            lmab2) grep -aq LMAB2_DONE /root/lmab2.log 2>/dev/null && continue ;;")

for a, b in ((old_for, new_for), (old_case, new_case)):
    if a not in src:
        print("ANCHOR MISSING, refusing:\n%r" % a)
        sys.exit(1)
    src = src.replace(a, b, 1)

tmp = P + ".new"
open(tmp, "w").write(src)
os.replace(tmp, P)
print("patched: keepd now watches lmab2")
