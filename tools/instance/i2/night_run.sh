#!/usr/bin/env bash
# The only way an instance2 overnight job should be started:  preflight -> smoke -> real.
#
# WHY THIS EXISTS.  Four nights in a row were lost to failures that died in the first minutes and
# then let the GPU idle until morning:
#   * vast rewrote authorized_keys, so the i1->i2 link was dead and the branch never came back
#   * a run was launched with `> /dev/null`, so its only progress log went to the void
#   * the pairs file was read while it was still being scp'd -- truncated gzip, empty qmin
#   * that empty qmin reached argparse as `--qmin ''` and killed the arm at second zero
# None of them needed a long run to expose.  Every one of them would have shown up in fifteen
# minutes at 1/50 scale, which is exactly what this does: it runs THE SAME SCRIPT, with every
# knob turned down, in a private TAG namespace, and refuses to launch the real one unless the
# short version reached its own DONE marker.
#
#   usage:  bash /root/night_run.sh /root/night4b.sh
#           SKIP_SMOKE=1 bash /root/night_run.sh /root/night4b.sh    # only when re-launching a
#                                                                    # run that already smoked
set -u
SCRIPT=${1:-}
[ -n "$SCRIPT" ] && [ -r "$SCRIPT" ] || { echo "usage: bash /root/night_run.sh /root/<job>.sh"; exit 2; }
BASE=$(basename "$SCRIPT" .sh)
SMOKE_MIN=${SMOKE_MIN:-45}
REPO=/root/ptcg/repo
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; exit 1; }

echo "== preflight: $SCRIPT =="

bash -n "$SCRIPT" || bad "the script does not parse"
ok "syntax"

# A smoke run that writes to the real TAG would overwrite the artifacts it is meant to protect.
grep -q 'TAG=${TAG:-' "$SCRIPT" || bad "not tag-parameterised (need TAG=\${TAG:-...}); the smoke run would clobber the real artifacts"
grep -q 'LOG=${LOG:-' "$SCRIPT" || bad "not log-parameterised (need LOG=\${LOG:-...})"
ok "tag/log parameterised, so the smoke run gets its own namespace"

pgrep -f "[b]ash $SCRIPT" >/dev/null && bad "$BASE is already running: $(pgrep -f "[b]ash $SCRIPT" | tr '\n' ' ')"
ok "no copy of this job already running"

MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$MIB" -le 2000 ] || bad "GPU busy (${MIB} MiB): $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"
ok "GPU free (${MIB} MiB used)"

AVAIL=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "$AVAIL" -ge 40 ] || bad "only ${AVAIL}G free on /root; a run writes two ~170 MB adapters plus traces"
ok "disk ${AVAIL}G free"

# The i1->i2 link, checked in one second instead of the 100 minutes the branch poll would take.
# branchd2 on instance1 touches this file on every poll (every 120 s); if the ssh key was
# rewritten under us, the mtime stops moving and nothing else here would notice.
HB=/root/.branchd2_alive
if [ -f "$HB" ]; then
    AGE=$(( $(date -u +%s) - $(stat -c %Y "$HB") ))
    [ "$AGE" -lt 400 ] || bad "instance1's brancher last polled ${AGE}s ago -- the i1->i2 link is down, so the branch step would hang"
    ok "instance1 brancher alive (${AGE}s ago)"
else
    printf '  warn  no %s yet -- the smoke run will test the link the slow way\n' "$HB"
fi

[ -s /root/branch_request2 ] && bad "a branch request is already queued: $(cat /root/branch_request2) -- the smoke run would collide with it"
ok "no branch request in flight"

# Paths named in the script's header block.  In a SUBSHELL: these are the script's own
# assignments and one of them referencing an unset variable must not be able to end the check.
MISSING=$(bash -c '
    set +u
    eval "$(sed -n "1,/^say /p" "$0" | grep -E "^[A-Za-z_]+=")" 2>/dev/null
    for v in REPO REF PREV VOCAB SO; do
        val=$(eval echo "\${$v}")
        [ -n "$val" ] || continue
        [ -e "$val" ] && echo "ok $v -> $val" || echo "MISSING $v -> $val"
    done' "$SCRIPT")
echo "$MISSING" | sed 's/^ok /  ok    /'
echo "$MISSING" | grep -q '^MISSING' && bad "$(echo "$MISSING" | grep '^MISSING')"
echo "  ok    every path in the header exists"

python3 - <<'PY' || bad "the training stack does not import"
import torch, peft, transformers
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("  ok    torch %s / peft %s / transformers %s / %s"
      % (torch.__version__, peft.__version__, transformers.__version__, torch.cuda.get_device_name(0)))
PY

echo "== preflight complete =="
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
    echo "== smoke SKIPPED (SKIP_SMOKE=1) =="
else
    echo "== smoke: the same script at 1/50 scale, ${SMOKE_MIN} min cap =="
    SM="/root/pairs_smoke* /root/traces_smoke* /root/lmlog_smoke* /root/collect_smoke* \
        /root/train_smoke* /root/gate_smoke* /root/branch_smoke* /root/smoke.log"
    # shellcheck disable=SC2086
    ls -d $SM /root/out/lora_smoke_* 2>/dev/null | sed 's/^/  stale, removing: /'
    # shellcheck disable=SC2086
    rm -rf $SM /root/out/lora_smoke_*

    # Small enough to finish in minutes, structurally identical to the real run: two collection
    # shards, a real branch request served by instance1, BOTH trainings, and the gate with its
    # summary block -- which is where a key-name typo would otherwise wait until morning.
    START=$(date -u +%s)
    TAG=smoke LOG=/root/smoke.log HOURS=1 \
    OPPS=marnie_grimmsnarl,ogerpon_mono GAMES=24 GATE_GAMES=6 EPOCHS=1 \
    BR_BUDGET=10000 BR_PLAYOUTS=6 BR_PERGAME=20 BR_WAIT=12 BR_FALLBACK=0 MINROWS=100 \
        timeout "${SMOKE_MIN}m" bash "$SCRIPT"
    RC=$?
    MIN=$(( ($(date -u +%s) - START) / 60 ))

    if [ $RC -ne 0 ] || ! grep -q DONE /root/smoke.log 2>/dev/null; then
        echo
        echo "== SMOKE FAILED after ${MIN} min (rc $RC) -- the real run is NOT starting =="
        tail -25 /root/smoke.log 2>/dev/null
        for f in /root/train_smoke_filt.log /root/train_smoke_base.log /root/gate_smoke.log \
                 /root/collect_smoke.s0.log; do
            [ -s "$f" ] && { echo "--- $f ---"; tail -8 "$f"; }
        done
        exit 1
    fi
    echo "  ok    smoke reached DONE in ${MIN} min"
    rm -rf /root/out/lora_smoke_*      # two adapters, ~340 MB, of no further use
fi

# SMOKE_ONLY exercises the harness itself without committing the night to a real run.
[ "${SMOKE_ONLY:-0}" = 1 ] && { echo "== SMOKE_ONLY: harness validated, real run not started =="; exit 0; }

echo "== launching the real run =="
# Never `> /dev/null`: one night's progress log was lost that way and completion had to be
# reconstructed from adapter timestamps.
setsid nohup bash "$SCRIPT" > "/root/${BASE}.stderr" 2>&1 < /dev/null &
sleep 20
pgrep -f "[b]ash $SCRIPT" >/dev/null || { echo "  FAIL  it exited within 20s:"; tail -20 "/root/${BASE}.stderr"; exit 1; }
echo "  ok    pid $(pgrep -f "[b]ash $SCRIPT" | head -1); watch: tail -f /root/${BASE}.log"
