#!/usr/bin/env python3
"""Move every STOP_AFTER from 2026-08-15T12:00Z to 2026-08-16T03:00Z (= 8/16 12:00 JST).

Three files carry the same literal and all three have to move together, because they gate each
other: field_chain stops itself, field_keep declines to restart it, and keepd stops supervising
everything. Missing one leaves the loop alive with nobody watching it, or watched by a
supervisor that refuses to act.

Written to a temp file and os.replace'd: keepd and field_keep are RUNNING, and a running bash
script is read by byte offset, so an in-place edit corrupts the next command they read. The new
inode leaves the live processes on the old content until they are restarted.
"""
import os
import sys

OLD = "2026-08-15T12:00:00Z"
NEW = "2026-08-16T03:00:00Z"
FILES = ["/root/field_chain.sh", "/root/field_keep.sh", "/root/keepd.sh"]

bad = [p for p in FILES if not os.path.exists(p)]
if bad:
    print("MISSING: %s -- refusing to patch a partial set" % bad)
    sys.exit(1)

plan = []
for p in FILES:
    src = open(p).read()
    n = src.count(OLD)
    if n == 0 and NEW in src:
        print("%-24s already at %s" % (os.path.basename(p), NEW))
        continue
    if n == 0:
        print("%-24s does NOT contain %s -- refusing" % (os.path.basename(p), OLD))
        sys.exit(1)
    plan.append((p, src, n))

for p, src, n in plan:
    tmp = p + ".new"
    open(tmp, "w").write(src.replace(OLD, NEW))
    os.replace(tmp, p)
    print("%-24s %d occurrence(s) -> %s" % (os.path.basename(p), n, NEW))

for p in FILES:
    src = open(p).read()
    assert OLD not in src, p
    assert NEW in src, p
print("all three now stop at %s (= 2026-08-16 12:00 JST)" % NEW)
