import os

p = "/root/deck_lora2.sh"
s = open(p).read()

old = 'if [ ! -s "$PAIRS" ]; then\n    echo "$TAG|$DECK|$BUDGET|$PLAYOUTS|$PER_GAME" > /root/branch_request2'
new = (
    "# Resume on RECORD COUNT, not file size. A gzip holding ZERO records is ~53 bytes, so -s is\n"
    "# true for it: pass 3 skipped the branch step entirely for four decks because pass 2 had\n"
    "# shipped them empty pairs files, then died at the count check below with 'pairs: 0' and no\n"
    "# hint of why. Exactly the trap the trace guard above was written for, one file downstream.\n"
    'if [ "$(zcat "$PAIRS" 2>/dev/null | head -1 | wc -l)" -eq 0 ]; then\n'
    '    rm -f "$PAIRS"\n'
    '    echo "$TAG|$DECK|$BUDGET|$PLAYOUTS|$PER_GAME" > /root/branch_request2'
)
assert s.count(old) == 1, ("branch guard", s.count(old))
s = s.replace(old, new)

old2 = ('grep -aq "PROBE OK" "$STATE/probe_$TAG.log" || { say "PROBE FAILED"; '
        'tail -4 "$STATE/probe_$TAG.log"; exit 1; }')
new2 = (
    "# Gate on ACCURACY, not on the 0.15 loss threshold dpo_teacher prints its verdict from.\n"
    "# slowking r1 overfit 2464 pairs to 94.1% train accuracy at loss 0.1644 and was killed for\n"
    "# missing 0.15 by 0.014 -- the probe exists to catch a trainer that CANNOT learn, and 94%\n"
    "# is not that. Loss floors move with pair difficulty per deck; accuracy does not.\n"
    'PACC=$(grep -a "^FINAL train loss" "$STATE/probe_$TAG.log" '
    '| sed -n "s/.*acc \\([0-9.]*\\)%.*/\\1/p" | head -1)\n'
    'if ! awk -v a="${PACC:-0}" "BEGIN{exit !(a+0 >= 85)}"; then\n'
    '    say "PROBE FAILED (train acc ${PACC:-?}% < 85)"; tail -4 "$STATE/probe_$TAG.log"; exit 1\n'
    "fi\n"
    'say "probe OK (train acc ${PACC}%)"'
)
assert s.count(old2) == 1, ("probe gate", s.count(old2))
s = s.replace(old2, new2)

t = p + ".new"
open(t, "w").write(s)
os.chmod(t, 0o755)
os.replace(t, p)
print("deck_lora2.sh patched: pairs guard by record count, probe gated on accuracy")
