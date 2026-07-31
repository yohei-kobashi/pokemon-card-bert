"""Assemble an engine_v2 submission bundle (mega_lucario ships this way, NOT policies.py).

The competition archive needs a self-contained main.py at its root, but engine_v2 imports
from agents/_engine.py. So concatenate:

    header + agents/_engine.py + agents/engine_v2.py (with the `from agents._engine import`
    block stripped, since those names are already in scope) + load_deck + a wrapper that
    builds `make_policy(DECK, PROFILE)` and exposes `agent(obs_dict)`.

Then reuse library.build_submission's layout (deck.csv + cg/) by staging over it.

    PYTHONPATH=cg-lib python tools/build_engine_v2_submission.py <agent_name> <deck> [--tag v30]
"""
import argparse, json, os, re, shutil, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

P = argparse.ArgumentParser()
P.add_argument("policy", help="policy/l2 key, e.g. mega_lucario")
P.add_argument("deck", help="deck name in decks/, e.g. mega_lucario_hilda")
P.add_argument("--tag", required=True, help="submission tag, e.g. mega_lucario_v30")
P.add_argument("--note", default="", help="comment appended to the wrapper")
A = P.parse_args()

eng = open(os.path.join(ROOT, "agents", "_engine.py")).read()
v2 = open(os.path.join(ROOT, "agents", "engine_v2.py")).read()

# strip the `from agents._engine import ( ... )` block -- those names are already defined
v2_stripped, n = re.subn(r"from agents\._engine import \([^)]*\)\n", "", v2)
if n != 1:
    sys.exit(f"expected exactly 1 _engine import block in engine_v2.py, found {n}")
if "from agents._engine import" in v2_stripped:
    sys.exit("a single-line `from agents._engine import` survived -- fix the regex")

tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
prof = dict(tuning.get(A.deck) or tuning.get(A.policy) or {})
prof.setdefault("policy", A.policy)
prof.setdefault("l2", A.policy)

LOAD_DECK = '''

def load_deck(name):
    """Same loader as the legacy engine (local decks/ or bundled deck.csv)."""
    candidates = [
        os.path.join("decks", name + ".csv"),
        name + ".csv", "deck.csv", "/kaggle_simulations/agent/deck.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return [int(line) for line in f if line.strip()]
    raise FileNotFoundError("deck not found: " + ", ".join(candidates))

'''

wrapper = (f"\n# --- deck agent: {A.policy}{(' + ' + A.note) if A.note else ''} ---\n"
           f"DECK = load_deck({A.deck!r})\n"
           f"PROFILE = {prof!r}\n"
           f"_POLICY = make_policy(DECK, PROFILE)\n\n"
           "def agent(obs_dict: dict) -> list[int]:\n"
           "    return _POLICY.act(obs_dict)\n")

main_py = ("# --- auto-assembled engine_v2 submission (_engine + engine_v2 + wrapper) ---\n"
           + eng + "\n\n" + v2_stripped + LOAD_DECK + wrapper)

# stage: reuse build_submission for deck.csv + cg/, then overwrite main.py
import library
res = library.build_submission(A.policy, A.deck, _v2_stage=True)
stage = res["dir"] if isinstance(res, dict) and "dir" in res else None
if stage is None:
    cands = [d for d in os.listdir(os.path.join(ROOT, "submissions")) if d == f"{A.policy}-{A.deck}"]
    stage = os.path.join(ROOT, "submissions", cands[0])

final_dir = os.path.join(ROOT, "submissions", f"{A.tag}-{A.deck}")
if os.path.exists(final_dir):
    shutil.rmtree(final_dir)
shutil.copytree(stage, final_dir)
shutil.rmtree(os.path.join(final_dir, "__pycache__"), ignore_errors=True)
open(os.path.join(final_dir, "main.py"), "w").write(main_py)

tar = os.path.join(ROOT, "submissions", f"{A.tag}-{A.deck}.tar.gz")
if os.path.exists(tar):
    os.remove(tar)
subprocess.check_call(["tar", "-czf", tar, "-C", final_dir] + sorted(os.listdir(final_dir)))
print(f"main.py lines : {main_py.count(chr(10))}")
print(f"deck          : {A.deck}  PROFILE={prof}")
print(f"staged        : {final_dir}")
print(f"tar           : {tar}  ({os.path.getsize(tar)/1e6:.2f} MB)")
