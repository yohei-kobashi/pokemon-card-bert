#!/bin/bash
# Start dagger_loop3 overnight, on whichever checkpoint the v40 chain leaves standing.
#
# The two branches need DIFFERENT PROMPT FORMATS, which is the whole reason this is a script and
# not a one-liner. rerank_v40 was trained with menu_dedup=True (one menu entry per act);
# rerank_loop2 predates it. Screening or collecting for a model under the other format measures
# the mismatch, so rl_config.PROMPT_FMT and the pools are switched together or not at all.
#
# If no verdict can be read -- the chain died, the screen produced nothing -- the fallback is
# rerank_loop2, the checkpoint with the best screen on record (median 41.0%, WORSE 20). Falling
# back to the UNMEASURED model would be the wrong way round.
set -u
REPO=/root/ptcg/repo
LOG=/root/autostart_l3.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[auto $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "############ waiting for the v40 chain ############"
while pgrep -f "[c]hain_v40.sh" > /dev/null; do sleep 120; done
sleep 30
say "v40 chain has exited"

VERDICT=$(grep -oE "VERDICT: (v40 WINS|keep rerank_loop2)" /root/chain_v40.log | tail -1)
say "verdict line: ${VERDICT:-<none found>}"

if echo "$VERDICT" | grep -q "v40 WINS"; then
  MODEL=/root/out/rerank_v40
  BASE=$REPO/data/rerank/v40_base.jsonl.gz
  VALUED=$REPO/data/rerank/v40_attach_q1.jsonl.gz,$REPO/data/rerank/v40_attach_q2.jsonl.gz
  WANT_DEDUP=True
  say "promoting v40: menu_dedup stays ON"
else
  MODEL=/root/out/rerank_loop2
  BASE=$REPO/data/rerank/v39_0731.rerank.jsonl.gz
  VALUED=$REPO/data/rerank/attach_q1c.jsonl.gz,$REPO/data/rerank/attach_q2.jsonl.gz
  WANT_DEDUP=False
  say "reverting to rerank_loop2: menu_dedup must go OFF, pools revert to the old menu"
fi

[ -f "$MODEL/model.safetensors" ] || { say "STOP: $MODEL has no model.safetensors"; exit 1; }
for f in $(echo "$BASE,$VALUED" | tr ',' ' '); do
  [ -s "$f" ] || { say "STOP: missing pool $f"; exit 1; }
done

python3 - "$WANT_DEDUP" <<'PY' || { say "STOP: could not set PROMPT_FMT"; exit 1; }
import re, sys
want = sys.argv[1]
p = "tools/rl_config.py"
s = open(p).read()
new = re.sub(r"menu_dedup=(True|False)", "menu_dedup=%s" % want, s)
if "menu_dedup=%s" % want not in new:
    raise SystemExit("menu_dedup not found in %s" % p)
open(p, "w").write(new)
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
from tools import rl_config
print("[fmt] PROMPT_FMT is now %s" % rl_config.PROMPT_FMT)
assert str(rl_config.PROMPT_FMT["menu_dedup"]) == want, "the file says one thing and the import another"
PY

# The pools and the renderer must agree, so check rather than trust: render the menu of a pool
# record the way the live agent will and require it to come back unchanged.
python3 - "$BASE" <<'PY' || { say "STOP: the pool's menu does not match what the agent will render"; exit 1; }
import gzip, json, sys
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib"); sys.path.insert(0, "tools")
from menu_dedup_pool import rewrite
from tools import rl_config
want = rl_config.PROMPT_FMT["menu_dedup"]
n = same = 0
for line in gzip.open(sys.argv[1], "rt"):
    s = json.loads(line).get("state") or ""
    if " :: " not in s:
        continue
    n += 1
    same += (rewrite(s)[0] == s)          # already deduped?
    if n >= 2000:
        break
frac = same / max(1, n)
print("[fmt] %d pool prompts, %.1f%% already one-entry-per-act; PROMPT_FMT wants menu_dedup=%s"
      % (n, 100 * frac, want))
ok = (frac > 0.95) if want else (frac < 0.5)
raise SystemExit(0 if ok else 1)
PY

say "starting dagger_loop3 | MODEL=$MODEL"
export MODEL BASE VALUED
KIND=rerank3 RATIO=0.05 VALUED_FRAC=0.05 DEADLINE_H=20 MARGIN=0.5 \
  setsid nohup bash tools/dagger_loop3.sh >/dev/null 2>&1 < /dev/null &
sleep 10
pgrep -f "[d]agger_loop3.sh" > /dev/null && say "loop3 is running" || say "STOP: loop3 did not start"
