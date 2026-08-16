#!/usr/bin/env bash
# Push the pilot's rule files to both instances, VERIFY they match, and name the processes that
# are still running the old code.
#
# Why this exists. Editing dusk_plan.py changes what the pilot is, so it has to reach three
# machines and then be re-loaded. Doing that by hand has two silent failure modes, and both cost
# a run today:
#   * a copy that did not land -- there was no md5 check, only the assumption rsync worked
#   * a copy that landed under a RUNNING process. Python reads a module once, at import, so a
#     chain started before the edit keeps piloting with the old rules while every file on disk
#     says otherwise. Nothing errors; the numbers are just measured on the wrong pilot.
#
# What this does NOT sync, deliberately: models/adapters.json. The registry is per-machine STATE
# (instance1 points dusknoir at its champion, instance2 points its sparring copy at whatever the
# current pass was gated against). Copying it across would silently re-point one machine's
# opponents mid-run.
set -u
cd "$(dirname "$0")/.."
I1="-i $HOME/.ssh/id_ed25519_vast -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 20142"
I1H=root@ssh5.vast.ai
I2="-i $HOME/.ssh/id_ed25519_vast -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 19839"
I2H=root@175.155.64.145
REPO=/root/ptcg/repo
FILES="tools/dusk_plan.py lm/plan_filter.py tools/rl_config.py lm/rerank_scorer.py"

for f in $FILES; do
    [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done
python3 - <<'PY' || exit 1
import ast, sys
for f in ("tools/dusk_plan.py", "lm/plan_filter.py", "tools/rl_config.py",
          "lm/rerank_scorer.py"):
    try:
        ast.parse(open(f).read())
    except SyntaxError as e:
        sys.exit("%s does not parse: %s" % (f, e))
print("all rule files parse")
PY

echo "--- pushing ---"
for f in $FILES; do
    d=$(dirname "$f")
    rsync -az -e "ssh $I1" "$f" "$I1H:$REPO/$d/" || exit 1
    rsync -az -e "ssh $I2" "$f" "$I2H:$REPO/$d/" || exit 1
done

echo "--- verifying (all three must agree) ---"
LOCAL=$(md5sum $FILES | awk '{print $1}' | md5sum | cut -d' ' -f1)
R1=$(ssh $I1 $I1H "cd $REPO && md5sum $FILES | awk '{print \$1}' | md5sum | cut -d' ' -f1" 2>/dev/null)
R2=$(ssh $I2 $I2H "cd $REPO && md5sum $FILES | awk '{print \$1}' | md5sum | cut -d' ' -f1" 2>/dev/null)
echo "  local $LOCAL"
echo "  i1    $R1"
echo "  i2    $R2"
if [ "$LOCAL" != "$R1" ] || [ "$LOCAL" != "$R2" ]; then
    echo "MISMATCH -- do not trust any measurement taken now"; exit 1
fi
echo "  OK, identical"

# A process that started before the newest rule file is piloting with the old rules in memory.
echo "--- processes older than the edit (these need a restart to pick it up) ---"
NEWEST=$(date -u -r tools/dusk_plan.py +%s)
for pair in "$I1|$I1H|i1" "$I2|$I2H|i2"; do
    OPT=${pair%%|*}; rest=${pair#*|}; HOST=${rest%%|*}; NAME=${rest##*|}
    ssh $OPT $HOST "ps -eo pid,lstart,cmd | grep -E 'field_chain|deck_lora2|pass3|gate_protagonist|lm_mirror_log|dpo_branch' | grep -v grep" 2>/dev/null \
      | while read -r pid rest2; do
            started=$(date -u -d "$(echo "$rest2" | cut -d' ' -f1-5)" +%s 2>/dev/null) || continue
            [ "$started" -lt "$NEWEST" ] && echo "  $NAME STALE pid $pid  $(echo "$rest2" | cut -c26-100)"
        done
done
echo "(nothing listed above = every running process already has the new rules)"
