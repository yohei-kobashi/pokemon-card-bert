import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = "/root/ptcg/repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cg-lib"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import arena
from lm.agent import make_lm_agent
from lm.scorer import LlamaScorer

GGUF = "/root/sftv2.Q4_K_M.gguf"


def load_deck(name):
    with open(os.path.join(REPO, "decks", name + ".csv")) as f:
        return [int(x) for x in f if x.strip()]


tuning = json.load(open(os.path.join(REPO, "agents", "tuning.json")))
PILOT, OPP = "crustle", "alakazam"
pdeck, odeck = load_deck(PILOT), load_deck(OPP)
pprof = tuning.get(PILOT, {})
oprof = tuning.get(OPP, {})

print("loading scorer...", flush=True)
scorer = LlamaScorer(GGUF, n_threads=4, time_budget=10**9)   # no fallback -> pure speed
lm_agent = make_lm_agent(pdeck, pprof, model=scorer)
opp_agent = make_lm_agent(odeck, oprof, model=None)          # engine opponent

# per-decision timing wrapper
times = []


def timed(obs):
    n0, s0 = scorer.n_decisions, scorer.spent
    r = lm_agent(obs)
    if scorer.n_decisions > n0:
        times.append(scorer.spent - s0)
    return r


print("playing 1 game (crustle LM vs alakazam engine)...", flush=True)
t0 = time.time()
r = arena.play(timed, opp_agent, pdeck, odeck, max_steps=4000)
wall = time.time() - t0
res = {0: "crustle(LM) WINS", 1: "alakazam(engine) wins", None: "draw/timeout"}[r]
print("RESULT:", res, flush=True)
print("decisions=%d  total_infer=%.1fs  wall=%.1fs" % (scorer.n_decisions, scorer.spent, wall), flush=True)
if times:
    times.sort()
    print("per-decision: mean=%.2fs  median=%.2fs  p90=%.2fs  max=%.2fs" % (
        sum(times) / len(times), times[len(times) // 2],
        times[int(len(times) * 0.9)], times[-1]), flush=True)
print("ALLDONE", flush=True)
