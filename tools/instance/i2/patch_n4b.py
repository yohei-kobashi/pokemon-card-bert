import os

p = "/root/night4b.sh"
s = open(p).read()

# (1) The pairs file arrives as a transfer from instance1, and `[ -s ]` goes true on its FIRST
#     byte.  Reading it there cost a night: NP came back 1287 of 1626 and the qmin python died
#     on the truncated tail, leaving QMIN empty.  Wait for a size that stops changing AND a gzip
#     that passes its own CRC before anything reads this file.
old = '''for _ in $(seq 1 100); do [ -s "$PAIRS" ] && break; sleep 60; done'''
new = '''for _ in $(seq 1 100); do [ -s "$PAIRS" ] && break; sleep 60; done
    for _ in $(seq 1 60); do
        a=$(stat -c %s "$PAIRS" 2>/dev/null || echo 0); sleep 10
        b=$(stat -c %s "$PAIRS" 2>/dev/null || echo 0)
        [ "$a" = "$b" ] && [ "$a" != 0 ] && gzip -t "$PAIRS" 2>/dev/null && break
    done'''
assert s.count(old) == 1, "pairs-wait anchor"
s = s.replace(old, new)

# (2) An empty QMIN reached the trainer as `--qmin ""` and argparse killed the arm.  Fail here,
#     where the log says why, instead of two minutes later inside python.
old2 = 'say "qmin chosen: $QMIN"'
new2 = ('[ -n "$QMIN" ] || { say "STOP: qmin selection produced nothing '
        '(is $PAIRS a complete gzip?)"; exit 1; }\n' + old2)
assert s.count(old2) == 1, "qmin anchor"
s = s.replace(old2, new2)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
os.chmod(p, 0o755)
print("patched")
