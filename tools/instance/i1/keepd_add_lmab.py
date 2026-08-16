#!/usr/bin/env python3
"""Put lmab under keepd's supervision.

Written to a temp file and os.replace'd rather than edited in place: keepd is running, and a
running bash script is read by BYTE OFFSET, so editing the file underneath it corrupts the
next command it reads. os.replace gives the new content a new inode; the live process keeps
reading the old one until it is restarted, which is exactly the behaviour wanted here."""
import os
import sys

P = "/root/keepd.sh"
src = open(P).read()

if "lmab" in src:
    print("already patched")
    sys.exit(0)

old_for = 'for W in hole_adopt restart_at_boundary; do'
new_for = 'for W in hole_adopt restart_at_boundary lmab; do'
old_case = ('            restart_at_boundary) grep -aq RESTART_DONE /root/restart.log '
            '2>/dev/null && continue ;;')
new_case = old_case + (
    '\n            lmab) grep -aq LMAB_DONE /root/lmab.log 2>/dev/null && continue ;;')

for a, b in ((old_for, new_for), (old_case, new_case)):
    if a not in src:
        print("ANCHOR MISSING, refusing to patch:\n%r" % a)
        sys.exit(1)
    src = src.replace(a, b, 1)

tmp = P + ".new"
open(tmp, "w").write(src)
os.replace(tmp, P)
print("patched: keepd now watches lmab (guarded by LMAB_DONE)")
