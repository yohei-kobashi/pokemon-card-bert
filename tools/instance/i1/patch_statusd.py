"""Fix the digest's cell counter, which reported both a wrong count and a wrong denominator.

Two bugs, and the second one cost real time: `grep -c` exits non-zero on zero matches, so
`n=$(grep -ac ... || echo 0)` produced the two-line string "0\n0"; and the denominator was
hard-coded to 16 for every gate, so pf4b -- which has 2 arms x 4 opponents = 8 cells per shard --
read "9/16" when it had actually FINISHED, and I called it stalled.

A monitor that has to be right about denominators is a monitor with a second thing to get wrong.
Report the count and how long since the log last moved: staleness is the thing worth alerting on,
and it needs no knowledge of how big the job was supposed to be.
"""
import os

p = "/root/statusd.sh"
s = open(p).read()

old = """# cells done / cells expected for a sharded gate log
cells() {   # $1 log, $2 arms, $3 opponents
    local n; n=$(grep -ac " vs " "$1" 2>/dev/null || echo 0)
    echo "$n/$(( $2 * $3 ))"
}"""
new = """# how much a gate log has produced, and whether it is still moving
cells() {   # $1 log
    local n age
    n=$(grep -ac " vs " "$1" 2>/dev/null); n=${n:-0}
    age=$(( ($(date -u +%s) - $(stat -c %Y "$1" 2>/dev/null || echo 0)) / 60 ))
    if [ "$age" -gt 25 ]; then
        echo "$n cells, STALE ${age}m"
    else
        echo "$n cells, ${age}m ago"
    fi
}"""
assert s.count(old) == 1, "cells anchor"
s = s.replace(old, new)

old2 = '        [ -f /root/hole_$i.log ] && say "    hole gate s$i $(cells /root/hole_$i.log 4 4)"'
new2 = '''        [ -f /root/hole_$i.log ] && {
            C=$(cells /root/hole_$i.log)
            say "    hole gate s$i $C"
            case "$C" in *STALE*) ALERTS="$ALERTS hole-s$i-stale";; esac
        }'''
assert s.count(old2) == 1, "hole anchor"
s = s.replace(old2, new2)

old3 = '''            [ -f /root/pf4b_$i.log ] && echo "pf4b s$i $(grep -ac " vs " /root/pf4b_$i.log)/16"'''
new3 = '''            for J in /root/pf4b_$i.log /root/gate_night6.log; do
                [ -f "$J" ] && echo "$(basename $J) $(grep -ac " vs " "$J") cells $(( ($(date -u +%s) - $(stat -c %Y "$J")) / 60 ))m"
            done'''
assert s.count(old3) == 1, "pf4b anchor"
s = s.replace(old3, new3)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
os.chmod(p, 0o755)
print("patched")
