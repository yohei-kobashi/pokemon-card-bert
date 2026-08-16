"""Play real games through the EXTRACTED TARBALL, exactly as Kaggle will.

Everything checked so far ran against the staged directory or against library code imported
from the repo. This imports the tarball's own main.py, from its own extraction directory, with
the repo removed from sys.path -- so a file that only exists in the repo cannot rescue it.

Checks:
  * main.py imports and builds the ONNX scorer (not the silent engine_v2 fallback)
  * every returned move is legal (arena would raise otherwise) and games run to completion
  * the ID segment survives in the deployed prompt
  * per-game bank spend vs the 600 s forfeit line, measured rather than projected
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

TAR = sys.argv[1]
DECK = sys.argv[2]
OPP = sys.argv[3]
GAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 4
WORK = sys.argv[5] if len(sys.argv) > 5 else "/root/smoke_extract"

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
with tarfile.open(TAR) as tf:
    tf.extractall(WORK)
print("extracted %s -> %s" % (TAR, WORK), flush=True)

RUNNER = r'''
import json, os, sys, time
WORK, DECK, OPP, GAMES, REPO = sys.argv[1:6]
GAMES = int(GAMES)
# The bundle FIRST, and the repo only for the arena/opponent harness that Kaggle provides
# itself. main.py and everything it imports must resolve inside WORK.
sys.path.insert(0, WORK)
import main as bundle                                   # noqa: E402

scorer = getattr(bundle, "_scorer", None)
print("scorer built: %s" % (type(scorer).__name__ if scorer else "NONE -> engine_v2 fallback"))
assert scorer is not None, "bundle fell back to engine_v2: the model never loaded"

seen = {"id": 0, "deck": 0, "n": 0}
import lm.serialize as _ser                              # the BUNDLE's copy
_orig = _ser.serialize_stateless
def spy(obs, **kw):
    s = _orig(obs, **kw)
    seen["n"] += 1
    seen["id"] += (" ID ME d_" in s)
    seen["deck"] += s.startswith("DECK[")
    return s
_ser.serialize_stateless = spy
import lm.agent as _ag
_ag.serialize_stateless = spy                            # already bound by make_lm_agent

for p in (REPO, os.path.join(REPO, "cg-lib"), os.path.join(REPO, "tools")):
    sys.path.append(p)
import arena                                             # noqa: E402  (harness only)

def _read(name):
    with open(os.path.join(WORK, "decks", name + ".csv")) as f:
        return [int(x) for x in f if x.strip()]

dl, ol = _read(DECK), _read(OPP)
# The opponent is engine_v2, NOT battle_log.load_agent's per-deck module: the bundle ships its
# own `agents` package (engine_v2 only) and it is first on sys.path, so `agents.alakazam` does
# not resolve -- correctly, since Kaggle supplies the opponent and it is not part of the
# submission. A smoke test asks "does the artifact run, stay legal, and fit the bank", not
# "what is the win rate"; the win rate is measured separately through eval_rerank.
_tun = json.load(open(os.path.join(WORK, "agents", "tuning.json")))
from lm.agent import make_lm_agent as _mk                # noqa: E402  (the BUNDLE's copy)
oa = _mk(ol, _tun.get(OPP, {}), model=None)
w = 0
rows = []
for g in range(GAMES):
    if hasattr(scorer, "reset_bank"):
        scorer.reset_bank()
    t0 = time.time()
    mine = g % 2
    r = arena.play(bundle.agent, oa, dl, ol) if mine == 0 else arena.play(oa, bundle.agent, ol, dl)
    w += (r == mine)
    rows.append((time.time() - t0, scorer.spent, scorer.n_decisions))
    print("  game %d: %s  wall %.0fs  bank %.0fs  decisions %d"
          % (g + 1, "WIN" if r == mine else "loss", rows[-1][0], rows[-1][1], rows[-1][2]),
          flush=True)
print("RESULT %d/%d wins" % (w, GAMES))
print("BANK  max %.0fs of 600s  mean %.0fs   decisions max %d"
      % (max(x[1] for x in rows), sum(x[1] for x in rows) / len(rows),
         max(x[2] for x in rows)))
print("PROMPT  ID segment %d/%d   DECK[ head %d/%d"
      % (seen["id"], seen["n"], seen["deck"], seen["n"]))
assert seen["n"] and seen["id"] == seen["n"], "ID segment missing from deployed prompts"
print("SMOKE OK")
'''

runner = os.path.join(WORK, "_runner.py")
open(runner, "w").write(RUNNER)
t0 = time.time()
r = subprocess.run([sys.executable, runner, WORK, DECK, OPP, str(GAMES), "/root/ptcg/repo"],
                   text=True, cwd="/")
print("elapsed %.0fs, rc=%d" % (time.time() - t0, r.returncode))
sys.exit(r.returncode)
