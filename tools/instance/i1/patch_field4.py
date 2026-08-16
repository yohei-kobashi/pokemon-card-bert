import os

p = "/root/field_chain.sh"
s = open(p).read()

# Resume must test INTEGRITY, not size. Killing the chain mid-collection left a 201 KB truncated
# fld_tr1.jsonl.gz; the next start saw a non-empty file, skipped collection, and died in the
# branch on "Compressed file ended before the end-of-stream marker". `gzip -t` verifies the CRC
# and the end-of-stream marker, which is exactly the property "someone killed the writer"
# destroys and "the file is non-empty" does not. Same bug class as the 0-byte traces that lost
# instance2 a whole pass on 08-11 -- fixed there, not here, until now.
ok = '''ok_gz() {   # a gzip that fails its CRC / end-of-stream check is not a resumable artifact
    [ -s "$1" ] && gzip -t "$1" 2>/dev/null
}

'''
anchor = 'gpu_wait() {'
assert s.count(anchor) == 1
s = s.replace(anchor, ok + anchor, 1)

old = '    if [ ! -s "$TR" ]; then'
new = '''    if ! ok_gz "$TR"; then
        [ -e "$TR" ] && { say "discarding a truncated $TR"; rm -f "$TR" /root/fld_log$R.jsonl.gz; }'''
assert s.count(old) == 1, ("tr", s.count(old))
s = s.replace(old, new)

old2 = '    [ -s "$TR" ] || { say "STOP: trace $TR is empty -- collection produced nothing"; exit 1; }'
new2 = '    ok_gz "$TR" || { say "STOP: $TR is missing or truncated after collection"; exit 1; }'
assert s.count(old2) == 1, ("trcheck", s.count(old2))
s = s.replace(old2, new2)

old3 = '    if [ ! -s "$PAIRS" ]; then'
new3 = '''    if ! ok_gz "$PAIRS"; then
        [ -e "$PAIRS" ] && { say "discarding a truncated $PAIRS"; rm -f "$PAIRS"; }'''
assert s.count(old3) == 1, ("pairs", s.count(old3))
s = s.replace(old3, new3)

t = p + ".new"
open(t, "w").write(s)
os.chmod(t, 0o755)
os.replace(t, p)
print("field_chain: resume guards now verify gzip integrity")
