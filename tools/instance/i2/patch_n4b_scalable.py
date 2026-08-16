"""Make night4b.sh runnable at 1/50 scale so a smoke run can exercise the real code path.

Every knob becomes an env override with today's value as the default, so the real invocation is
byte-identical in behaviour.  The point is that the smoke run must be the SAME SCRIPT -- a
separate toy copy drifts, and then it certifies a pipeline nobody runs.
"""
import os

p = "/root/night4b.sh"
s = open(p).read()
subs = [
    # identity + namespace: TAG drives pairs_/traces_/lora_/gate_/train_ names, so overriding it
    # gives the smoke run a private namespace that cannot touch the real artifacts.
    ("TAG=night4b",
     "TAG=${TAG:-night4b}"),
    ("LOG=/root/night4b.log",
     "LOG=${LOG:-/root/night4b.log}"),
    ("OPPS=marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh",
     "OPPS=${OPPS:-marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh}"),
    ('DEADLINE=$(date -u -d "+7 hours" +%s)',
     'DEADLINE=$(date -u -d "+${HOURS:-7} hours" +%s)'),
    # the branch request instance1 answers
    ('echo "$TAG|dragapult_dusknoir|12000|24|15" > /root/branch_request2',
     'echo "$TAG|dragapult_dusknoir|${BR_BUDGET:-12000}|${BR_PLAYOUTS:-24}|${BR_PERGAME:-15}" > /root/branch_request2'),
    ('for _ in $(seq 1 100); do [ -s "$PAIRS" ] && break; sleep 60; done',
     'for _ in $(seq 1 ${BR_WAIT:-100}); do [ -s "$PAIRS" ] && break; sleep 60; done'),
    # BR_FALLBACK=0 makes a dead i1->i2 link FATAL instead of silently degrading to a local
    # branch -- which is the whole point of testing the link in the smoke run.
    ('    if [ ! -s "$PAIRS" ]; then\n        say "FALLBACK: local branch',
     '    if [ ! -s "$PAIRS" ]; then\n'
     '        [ "${BR_FALLBACK:-1}" = 1 ] || { say "STOP: no pairs from instance1 in ${BR_WAIT:-100} min'
     ' (fallback disabled)"; exit 1; }\n'
     '        say "FALLBACK: local branch'),
    # row floors scale with the run
    ('[ "$NP" -ge 500 ] || { say "STOP: only $NP pairs"; exit 1; }',
     '[ "$NP" -ge "${MINROWS:-500}" ] || { say "STOP: only $NP pairs"; exit 1; }'),
    ('QMIN=$(python3 - "$PAIRS" <<\'PY\'',
     'QMIN=$(python3 - "$PAIRS" "${MINROWS:-500}" <<\'PY\''),
    ("    if keep >= max(500, int(0.30 * n)):",
     "    if keep >= max(int(sys.argv[2]), int(0.30 * n)):"),
    ('--out "$OUT" --epochs 3 --beta 0.1',
     '--out "$OUT" --epochs "${EPOCHS:-3}" --beta 0.1'),
]
for old, new in subs:
    assert s.count(old) == 1, "anchor %r matched %d times" % (old[:60], s.count(old))
    s = s.replace(old, new)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)          # new inode: the run in flight keeps reading the old one
os.chmod(p, 0o755)
print("patched %d knobs" % len(subs))
