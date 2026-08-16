#!/usr/bin/env bash
# Adopt night6's winner and start night7 from it, so instance2 does not idle after ~16:15 UTC.
#
# night6 finishes around 16:15 UTC and morning is six hours later; that gap has already been lost
# twice this week. The adoption rule is the one the user fixed today -- any positive delta wins --
# so nothing here needs a human at 01:00 JST.
#
#   read   /root/gate_night6.json
#   pick   the better of arms a / b if its delta beats the filtered baseline at all
#   run    night7 from that adapter (or from dpo_r8 unchanged if neither arm was positive)
#
# Going again from an UNCHANGED policy is still worth the GPU: the round re-collects against the
# same field with different seeds, so it is another independent sample of the same question, not
# a repeat of the same experiment.
set -u
LOG=/root/night_chain.log
say() { echo "[chain $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
grep -aq CHAIN_DONE "$LOG" 2>/dev/null && { say "already done"; exit 0; }

say "waiting for night6"
for _ in $(seq 1 480); do
    grep -aq NIGHT6_DONE /root/night6.log 2>/dev/null && break
    sleep 60
done
grep -aq NIGHT6_DONE /root/night6.log 2>/dev/null || { say "STOP: night6 never finished"; exit 1; }

NEXT=$(python3 - <<'PY'
import json, sys
try:
    a = json.load(open("/root/gate_night6.json"))["arms"]
except Exception as e:
    print("/root/out/dpo_r8"); print("could not read the gate: %s" % e, file=sys.stderr); raise SystemExit
best, bd = None, 0.0
for k in ("a", "b"):
    d = (a.get(k) or {}).get("delta_vs_baseline")
    print("%s %+0.2f" % (k, d if d is not None else float("nan")), file=sys.stderr)
    if d is not None and d > bd:
        best, bd = k, d
print("/root/out/lora_night6_%s" % best if best else "/root/out/dpo_r8")
print("picked %s (%+.2f)" % (best or "neither -- staying on dpo_r8", bd), file=sys.stderr)
PY
2>>"$LOG")
say "night7 starts from $NEXT"

if [ ! -d "$NEXT" ]; then
    say "WARN: $NEXT is not there -- falling back to dpo_r8"
    NEXT=/root/out/dpo_r8
fi

# Same script, new tag: night6.sh already takes PREV/TAG/LOG from the environment, and night_run
# re-runs its own preflight and smoke before letting the real one start.
cd /root
PREV="$NEXT" TAG=night7 LOG=/root/night7.log \
    setsid --fork nohup bash /root/night_run.sh /root/night6.sh \
    > /root/night7_run.log 2>&1 < /dev/null
sleep 30
say "night7 launched: $(pgrep -cf '[n]ight_run.sh') runner up"
say "CHAIN_DONE"
