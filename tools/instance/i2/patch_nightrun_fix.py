"""Fix the preflight's own silent exit -- the exact failure class it exists to catch.

The header check eval'd the script's assignment lines in the CURRENT shell.  Those lines include
`PAIRS=/root/pairs_$TAG.jsonl.gz`, and TAG was in the exclusion list, so under `set -u` the
expansion of an unset TAG killed the shell -- with `2>/dev/null` swallowing the message and the
run reporting rc=0 having never reached the smoke.  A preflight that can exit silently is worth
nothing, so: evaluate in a subshell that cannot kill the parent, keep its stderr, and end with an
explicit marker so "it stopped early" can never again look like "it passed".
"""
import os

p = "/root/night_run.sh"
s = open(p).read()

old = """# Paths named in the script's header block, evaluated the way the script will evaluate them.
eval "$(sed -n '1,/^say /p' "$SCRIPT" | grep -E '^[A-Za-z_]+=' | grep -vE '^(GAMES|GATE_GAMES|TAG|LOG|OPPS|DEADLINE)=')" 2>/dev/null
for v in REPO REF PREV VOCAB SO; do
    val=${!v:-}
    [ -n "$val" ] || continue
    [ -e "$val" ] || bad "$v=$val does not exist"
    ok "$v -> $val"
done"""
new = """# Paths named in the script's header block.  In a SUBSHELL: these are the script's own
# assignments and one of them referencing an unset variable must not be able to end the check.
MISSING=$(bash -c '
    set +u
    eval "$(sed -n "1,/^say /p" "$0" | grep -E "^[A-Za-z_]+=")" 2>/dev/null
    for v in REPO REF PREV VOCAB SO; do
        val=$(eval echo "\\${$v}")
        [ -n "$val" ] || continue
        [ -e "$val" ] && echo "ok $v -> $val" || echo "MISSING $v -> $val"
    done' "$SCRIPT")
echo "$MISSING" | sed 's/^ok /  ok    /'
echo "$MISSING" | grep -q '^MISSING' && bad "$(echo "$MISSING" | grep '^MISSING')"
echo "  ok    every path in the header exists\""""
assert s.count(old) == 1, "header-check anchor"
s = s.replace(old, new)

# and a marker that makes an early exit visible
old2 = 'if [ "${SKIP_SMOKE:-0}" = 1 ]; then'
new2 = 'echo "== preflight complete =="\nif [ "${SKIP_SMOKE:-0}" = 1 ]; then'
assert s.count(old2) == 1, "marker anchor"
s = s.replace(old2, new2)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
os.chmod(p, 0o755)
print("patched")
