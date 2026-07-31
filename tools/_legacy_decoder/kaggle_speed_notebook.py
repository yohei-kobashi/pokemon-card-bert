"""Kaggle NOTEBOOK speed measurement for the crustle LLM agent (run ON Kaggle CPU).

PURPOSE: measure the PURE per-decision + per-game inference speed on the real Kaggle
CPU (AMD EPYC, 4 vCPU) with NO time-bank fallback and NO forfeit risk -- the authoritative
number the local vast box can only approximate. Complements the submission (which is
instrumented but masked by the safety fallback).

SETUP (in a Kaggle Notebook, CPU accelerator, Internet ON):
  1. Upload `crustle_lm_submission.tar.gz` as a Kaggle Dataset (e.g. "crustle-lm-agent").
  2. Attach it to the notebook (Add Input). It mounts read-only under /kaggle/input/<slug>/.
  3. Paste this file into a cell (or `%run` it) and run.

It extracts the bundle to a writable dir, installs llama-cpp-python, then plays ONE full
crustle(LM) vs alakazam(engine) game with the scorer's time budget lifted, printing every
decision's latency and the per-game total against the 600s bank.
"""
import glob
import os
import subprocess
import sys
import tarfile
import time

# ---- locate the uploaded bundle tarball -------------------------------------------
CANDS = glob.glob("/kaggle/input/*/crustle_lm_submission.tar.gz") + \
        glob.glob("/kaggle/input/*/*.tar.gz")
if not CANDS:
    sys.exit("upload crustle_lm_submission.tar.gz as a Dataset and attach it first")
TARBALL = CANDS[0]
WORK = "/kaggle/working/crustle_bundle"
os.makedirs(WORK, exist_ok=True)
if not os.path.exists(os.path.join(WORK, "main.py")):
    print("extracting", TARBALL, flush=True)
    with tarfile.open(TARBALL) as t:
        t.extractall(WORK)

# ---- deps: use Kaggle's internet to pip-install llama-cpp-python -------------------
# (the bundle also ships a llama_cpp/, but a fresh pip build matches the notebook env)
try:
    import llama_cpp  # noqa: F401
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "llama-cpp-python==0.3.34"], check=True)

# cap threads to the 4 vCPU the competition runtime gives
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, WORK)
import json

from lm.agent import make_lm_agent
from lm.scorer import LlamaScorer

# arena isn't in the bundle -> drive the bundled cg game loop directly
from cg.game import battle_start, battle_select, battle_finish


def load_deck(name):
    p = os.path.join(WORK, "decks", name + ".csv")
    if not os.path.exists(p):
        p = os.path.join(WORK, "deck.csv")
    with open(p) as f:
        return [int(x) for x in f if x.strip()]


tuning = json.load(open(os.path.join(WORK, "agents", "tuning.json")))
pdeck = load_deck("crustle")
odeck = load_deck("alakazam") if os.path.exists(os.path.join(WORK, "decks", "alakazam.csv")) else pdeck

GGUF = os.path.join(WORK, "model.gguf")
print("loading scorer (llama.cpp, 4 threads)...", flush=True)
t0 = time.time()
scorer = LlamaScorer(GGUF, n_threads=4, time_budget=10**9)   # NO fallback -> pure speed
print("scorer ready in %.1fs" % (time.time() - t0), flush=True)

pilot = make_lm_agent(pdeck, tuning.get("crustle", {}), model=scorer)
opp = make_lm_agent(odeck, tuning.get("alakazam", {}), model=None)

# ---- play one full game, timing every LLM decision --------------------------------
per = []
obs, _sd = battle_start(pdeck, odeck)
agents = (pilot, opp)
game_t0 = time.time()
try:
    for _ in range(4000):
        cur = obs.get("current")
        if cur is None or cur.get("result", -1) != -1:
            break
        if obs.get("select") is None:
            break
        yi = cur["yourIndex"]
        if yi == 0:
            n0, s0 = scorer.n_decisions, scorer.spent
            choice = agents[0](obs)
            if scorer.n_decisions > n0:
                dt = scorer.spent - s0
                per.append(dt)
                print("  decision %3d: %.2fs  (bank spent %.0f/600s)" % (
                    len(per), dt, scorer.spent), flush=True)
        else:
            choice = agents[1](obs)
        obs = battle_select(choice)
finally:
    battle_finish()

wall = time.time() - game_t0
res = obs.get("current", {}).get("result", -1)
print("\n==== RESULT ====", flush=True)
print("winner:", {0: "crustle(LM)", 1: "alakazam(engine)", -1: "unfinished"}.get(res, res), flush=True)
print("LLM decisions this game: %d" % len(per), flush=True)
print("total LLM inference: %.1fs  (bank limit 600s -> %s)" % (
    scorer.spent, "OK" if scorer.spent < 600 else "OVER BUDGET"), flush=True)
print("wall clock: %.1fs" % wall, flush=True)
if per:
    per_sorted = sorted(per)
    print("per-decision: mean=%.2fs median=%.2fs p90=%.2fs max=%.2fs" % (
        sum(per) / len(per), per_sorted[len(per) // 2],
        per_sorted[int(len(per) * 0.9)], per_sorted[-1]), flush=True)
