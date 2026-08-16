"""Validate the ACTUAL submission bundle: import its main.py (bootstrap + bundled
llama_cpp + relative paths) and play a real game vs an engine opponent."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
BUND = "/root/subm_crustle"
REPO = "/root/ptcg/repo"
os.chdir(BUND)
sys.path.insert(0, BUND)                       # bundle's cg/lm/agents/llama_cpp win
sys.path.insert(0, os.path.join(REPO, "tools"))  # arena only

import main                                     # runs bootstrap -> main.agent, main._scorer
import arena
from lm.agent import make_lm_agent

odeck = [int(x) for x in open(os.path.join(REPO, "decks", "alakazam.csv")) if x.strip()]
otun = json.load(open(os.path.join(REPO, "agents", "tuning.json")))
opp = make_lm_agent(odeck, otun.get("alakazam", {}), model=None)
pdeck = main._deck

sc = getattr(main, "_scorer", None)
print("scorer_loaded=%s model=%s" % (sc is not None, os.path.exists(os.path.join(BUND, "model.gguf"))), flush=True)

per = []
CAP = 15


class _Done(Exception):
    pass


def timed(obs):
    n0, s0 = (sc.n_decisions, sc.spent) if sc else (0, 0)
    r = main.agent(obs)
    if sc and sc.n_decisions > n0:
        per.append(sc.spent - s0)
        print("  decision %d: %.2fs (chose %s)" % (len(per), per[-1], r), flush=True)
        if len(per) >= CAP:
            raise _Done()
    return r


print("playing bundle crustle(LM) vs alakazam(engine)...", flush=True)
t0 = time.time()
r = arena.play(timed, opp, pdeck, odeck, max_steps=4000)
wall = time.time() - t0
print("RESULT: %s" % {0: "crustle(LM) WINS", 1: "alakazam wins", None: "draw/timeout"}[r], flush=True)
if sc:
    print("decisions=%d total_infer=%.1fs wall=%.1fs" % (sc.n_decisions, sc.spent, wall), flush=True)
    if per:
        per.sort()
        print("per-decision mean=%.2fs median=%.2fs p90=%.2fs max=%.2fs" % (
            sum(per) / len(per), per[len(per) // 2], per[int(len(per) * 0.9)], per[-1]), flush=True)
print("BUNDLE_OK", flush=True)
