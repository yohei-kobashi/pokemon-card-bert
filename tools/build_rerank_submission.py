"""Assemble a Kaggle submission for the cross-encoder reranker piloting one deck.

Adapted from the Qwen/llama.cpp bundle (submissions/CRUSTLE_LM_SUBMIT.md). Two swaps:
  model.gguf   -> model.onnx + tokenizer.json + vocab_remap.npy
  llama_cpp/   -> onnxruntime/ + tokenizers/   (the only runtimes the scorer needs)

The 197.65625 MiB cap applies to the **.tar.gz**, so sizes are reported compressed --
measuring the staged tree over-counts by ~20% and once nearly caused a false abort.

Runtime deps are copied from THIS machine's site-packages, so build it where the glibc /
Python minor version matches Kaggle (the vast box does; that is how the llama_cpp bundle
was made). onnxruntime ships large test/tooling trees that the scorer never imports; they
are stripped (~90 MiB -> ~17.5 MiB).

The prompt format is baked into main.py from tools/rl_config -- ONE source of truth shared with
build_rerank.py, rather than flags to re-type per build. Getting it wrong is silent: the model
just receives inputs it never saw and the win rate drops with no error anywhere. That has already
happened twice -- eval_rerank.py did not pass deck_name, so every historical reranker number was
measured without `ID ME`; and a bundle shipped without the ID segment entirely.

    PYTHONPATH=cg-lib python tools/build_rerank_submission.py crustle_stall \
        --onnx /root/onnx/pruned/model_wonly_int8.onnx \
        --tokenizer /root/out/rerank_gte_mp --remap /root/onnx/pruned/model/vocab_remap.npy \
        --pfmt current --tag rr_v39_1
"""
import argparse
import os
import shutil
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

TAR_CAP = 207257600           # 197.65625 MiB, measured on the .tar.gz
# Embedded into main.py in DEPENDENCY ORDER (each module is exec'd when its turn comes, so a
# module must not appear before one it imports at top level):
#   vocab -> cg.api | actions -> vocab | serialize -> vocab, actions | agent -> serialize, actions
# `identify` and `roles` are imported LAZILY from inside serialize, but they are embedded here
# anyway -- lazily-imported and swallowed is exactly how the ID segment went missing before.
LM_EMBED = ("vocab", "actions", "identify", "roles", "serialize", "rerank_scorer", "agent")
AGENT_FILES = ("engine_v2.py", "_engine.py", "tuning.json")
# onnxruntime's wheel is ~90 MiB, nearly all of it tooling the InferenceSession never loads
ORT_STRIP = ("test", "tests", "tools", "transformers", "quantization", "datasets",
             "__pycache__", "capi/libonnxruntime_providers_cuda*", "*.onnx")

MAIN_TEMPLATE = '''"""Kaggle submission: {deck} piloted by the {tag} cross-encoder reranker.

SELF-CONTAINED: every lm/ module is embedded in this file and installed into sys.modules, so the
bundle cannot lose one silently. That failure mode is not hypothetical -- lm/serialize imports
`identify` and `roles` LAZILY, inside the functions that render the prompt, and those call sites
swallow exceptions by design, which is how an earlier bundle shipped every prompt without its
`ID ME` segment and still ran.

Layout at run time (/kaggle_simulations/agent/):
    main.py  model.onnx  tokenizer.json  vocab_remap.npy  agents/  cg/  decks/{deck}.csv

Fallback is LAYERED, and the last tier does not touch lm at all:
    1. reranker scorer                   argmax over legal candidates
    2. engine_v2 through lm.agent        scorer unavailable
    3. engine_v2 through make_policy     lm itself unavailable
The previous version imported lm.agent at module scope AND routed its "fallback" through the
same module, so any lm import failure left no fallback at all -- just ERROR.
"""
import os

# Cap CPU threading BEFORE any native lib loads: Kaggle gives 4 vCPU and an unpinned
# OpenMP/BLAS pool oversubscribes and thrashes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "{threads}")

import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "cg-lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json

DECK_NAME = {deck!r}
# Baked from tools/rl_config.PROMPT_FMT at BUILD time. The prompt format is part of the model and
# lives in four renderers; passing it as loose command-line flags is how train and deploy drift
# apart without anything failing.
PROMPT_FMT = {pfmt!r}

_LM_ORDER = {lm_order!r}
_LM_SRC = {lm_src!r}


def _install_lm():
    """Exec the embedded sources as REAL modules, not a flattened namespace.

    lm/ modules reference each other module-qualified (`vocab.ctx_name`, `_id.render`), so
    concatenating them the way build_engine_v2_submission concatenates _engine + engine_v2 would
    leave those names unresolvable. Registering ModuleType objects keeps every import inside lm/
    working exactly as it does from disk.
    """
    pkg = types.ModuleType("lm")
    pkg.__file__ = os.path.join(HERE, "lm", "__init__.py")
    pkg.__path__ = [os.path.join(HERE, "lm")]
    pkg.__package__ = "lm"
    sys.modules["lm"] = pkg
    exec(compile(_LM_SRC["__init__"], pkg.__file__, "exec"), pkg.__dict__)
    for _name in _LM_ORDER:
        _full = "lm." + _name
        _m = types.ModuleType(_full)
        _m.__file__ = os.path.join(HERE, "lm", _name + ".py")
        _m.__package__ = "lm"
        sys.modules[_full] = _m
        exec(compile(_LM_SRC[_name], _m.__file__, "exec"), _m.__dict__)
        setattr(pkg, _name, _m)


def _load_deck(name):
    for p in (os.path.join(HERE, "decks", name + ".csv"),
              os.path.join(HERE, "deck.csv"),
              "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(p):
            with open(p) as f:
                return [int(x) for x in f if x.strip()]
    raise FileNotFoundError("deck not found for " + name)


_deck = _load_deck(DECK_NAME)
try:
    _profile = json.load(open(os.path.join(HERE, "agents", "tuning.json"))).get(DECK_NAME, dict())
except Exception:
    _profile = dict()


def _probe_natives():
    """Import the WHOLE native stack in a child process before trusting it in this one.

    onnxruntime and tokenizers are both compiled extensions built against one CPython ABI; on a
    different interpreter they can SEGFAULT rather than raise, and a segfault is not catchable by
    `except Exception` -- the agent dies and Kaggle records ERROR. The earlier probe covered
    onnxruntime only and left tokenizers unguarded. This buys CONTAINMENT, not diagnosis: the
    child's output never reaches the replay, whose `observation.logs` is [] even for known-good
    submissions.
    """
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import onnxruntime, tokenizers; "
            "print(onnxruntime.__version__, tokenizers.__version__)" % (HERE,))
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True,
                           timeout=180, env=dict(os.environ, PYTHONPATH=HERE))
    except Exception:
        return False
    return r.returncode == 0


_agent = None
TIER = "none"
try:
    _install_lm()
    from lm.agent import make_lm_agent
    _lm_ok = True
except Exception:
    _lm_ok = False

if _lm_ok:
    try:
        if not _probe_natives():
            raise RuntimeError("native probe failed in a child process")
        from lm.rerank_scorer import OnnxRerankerScorer
        _scorer = OnnxRerankerScorer(
            os.path.join(HERE, "model.onnx"), HERE,
            max_len={max_len}, threads={threads}, time_budget={time_budget},
            remap=os.path.join(HERE, "vocab_remap.npy"))
        _agent = make_lm_agent(_deck, _profile, model=_scorer, deck_name=DECK_NAME, **PROMPT_FMT)
        TIER = "reranker"
    except Exception:
        try:
            _agent = make_lm_agent(_deck, _profile, model=None)
            TIER = "engine_via_lm"
        except Exception:
            _agent = None

if _agent is None:
    from agents.engine_v2 import make_policy
    _agent = make_policy(_deck, _profile).act
    TIER = "engine_direct"


def agent(obs_dict: dict) -> list:
    # The FIRST call of every episode carries select=None and must return the 60-card DECK, not
    # an action -- see the competition's own sample_submission/main.py. engine_v2 implements it
    # at agents/engine_v2.py (`if obs.select is None: return self.deck`), lm/agent.py does not,
    # and that alone ERRORed three LM submissions before it was found.
    if obs_dict.get("select") is None:
        return list(_deck)
    return _agent(obs_dict)
'''


def _mb(n):
    return f"{n / 1024 / 1024:.1f} MiB"


def _copy_runtime(pkg, dest, strip=()):
    import importlib
    mod = importlib.import_module(pkg)
    src = os.path.dirname(mod.__file__)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*strip) if strip else None)
    return src


_SELFCHECK = r'''
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [HERE] + [p for p in sys.path if p not in ("", ".")]
import main as M                            # noqa: E402  (runs the real import path)

# 1. The MODEL must be driving. Every tier below `reranker` still plays legal games and still
#    scores -- in the engine_v2 band -- and that score would be recorded as the LM's live
#    rating. A silent degrade is worse than a build failure.
print("SELFCHECK tier %s" % M.TIER)
assert M.TIER == "reranker", "scorer did not load; this bundle would play as engine_v2"

# 2. The first observation of every Kaggle episode has select=None and must return the 60-card
#    deck. The local harness starts from battle_start() and NEVER issues this call, so passing
#    games proves nothing about it -- which is why three LM submissions came back ERROR.
first = {"current": None, "logs": [], "remainingOverageTime": 600.0,
         "search_begin_input": None, "select": None, "step": 1}
d = M.agent(first)
assert isinstance(d, list) and len(d) == 60 and all(isinstance(x, int) for x in d), \
    "deck-selection call returned %r len=%s" % (type(d).__name__, len(d) if hasattr(d, "__len__") else "?")

# 3. The embedded lm must be the version this prompt format needs, and EVERY decklist must be
#    present: lm/identify builds its posterior over all of tuning.json, so a bundle carrying one
#    list would not fail -- it would confidently name that deck every turn, in a segment the
#    model saw on 100% of training prompts.
from lm import identify, serialize, roles    # noqa: E402
fleet = identify._fleet()
want = sum(1 for v in json.load(open(os.path.join(HERE, "agents", "tuning.json"))).values()
           if isinstance(v, dict) and v.get("archetype"))
print("SELFCHECK fleet %d/%d decks | fmt %s" % (len(fleet), want, M.PROMPT_FMT))
assert len(fleet) == want and want > 1, "decklists missing from the bundle"
assert M.PROMPT_FMT.get("deck_mode") in serialize.DECK_MODES, "unknown deck_mode"
if M.PROMPT_FMT.get("deck_mode") == "roles":
    assert roles.for_deck(M.DECK_NAME), \
        "no prompt roles for %s: DECK[] would collapse to one 'oth' group" % M.DECK_NAME
print("SELFCHECK OK")
'''


def selfcheck(stage):
    """Import the STAGED tree in a clean interpreter and prove the prompt can be built.

    Everything in lm/serialize._identify is wrapped in ``except Exception: return ""``, by
    design -- identification is an aid, not a dependency. That makes a missing file in the
    bundle invisible: no crash, no size change, just a prompt missing a segment the model was
    trained on 100% of the time. Only running the staged tree in isolation catches it (this
    check was written after `import library` did exactly that)."""
    import subprocess
    path = os.path.join(stage, "_selfcheck.py")
    with open(path, "w") as f:
        f.write(_SELFCHECK)
    r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                       cwd="/", env={"PATH": os.environ.get("PATH", "")})
    os.remove(path)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("bundle selfcheck FAILED -- do not submit this tarball")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer", required=True, help="dir holding tokenizer.json")
    ap.add_argument("--remap", default="", help="vocab_remap.npy (vocab-pruned models)")
    # ONE knob, resolved from tools/rl_config, instead of three flags that must be remembered.
    # The prompt format is part of the model and is rendered in four places; every past
    # train/deploy divergence came from passing it by hand.
    ap.add_argument("--pfmt", default="current", choices=("current", "v37"),
                    help="current = rl_config.PROMPT_FMT (what build_rerank just used); "
                         "v37 = rl_config.PROMPT_FMT_V37, for models trained before the rebuild")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--time-budget", type=float, default=480.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions"))
    args = ap.parse_args()

    import library
    if args.deck not in set(library.list_decks()):
        raise SystemExit(f"unknown deck: {args.deck!r}")

    from tools import rl_config
    pfmt = dict(rl_config.PROMPT_FMT if args.pfmt == "current" else rl_config.PROMPT_FMT_V37)
    print(f"prompt format ({args.pfmt}): {pfmt}", flush=True)

    stage = os.path.join(args.out, args.tag)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "agents"))
    os.makedirs(os.path.join(stage, "decks"))

    # Embed lm/ INTO main.py rather than copying the package. A missing or stale module in a
    # copied lm/ is invisible: serialize imports identify and roles lazily and swallows the
    # failure, so the bundle keeps running and just drops a prompt segment.
    lm_src = {"__init__": open(os.path.join(ROOT, "lm", "__init__.py")).read()}
    for name in LM_EMBED:
        lm_src[name] = open(os.path.join(ROOT, "lm", name + ".py")).read()
    print(f"lm modules embedded: {len(lm_src)} "
          f"({sum(len(v) for v in lm_src.values()) / 1024:.0f} KiB)", flush=True)

    with open(os.path.join(stage, "main.py"), "w") as f:
        f.write(MAIN_TEMPLATE.format(deck=args.deck, tag=args.tag, pfmt=pfmt,
                                     lm_order=LM_EMBED, lm_src=lm_src,
                                     threads=args.threads, max_len=args.max_len,
                                     time_budget=args.time_budget))
    for fn in AGENT_FILES:
        shutil.copy(os.path.join(ROOT, "agents", fn), os.path.join(stage, "agents", fn))
    open(os.path.join(stage, "agents", "__init__.py"), "w").close()
    # ALL decklists, not just ours: lm/identify builds its posterior over every deck in
    # tuning.json, so a bundle carrying one list would not fail -- it would confidently name
    # the only deck it knows, on every turn, in a segment the model relies on. 276 KB.
    n_csv = 0
    for name in sorted(library.list_decks()):
        src = library.deck_path(name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(stage, "decks", name + ".csv"))
            n_csv += 1
    deck_csv = library.deck_path(args.deck)
    shutil.copy(deck_csv, os.path.join(stage, "deck.csv"))
    print(f"decklists bundled: {n_csv}", flush=True)
    shutil.copytree(library._cg_source(), os.path.join(stage, "cg"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    shutil.copy(args.onnx, os.path.join(stage, "model.onnx"))
    shutil.copy(os.path.join(args.tokenizer, "tokenizer.json"),
                os.path.join(stage, "tokenizer.json"))
    if args.remap:
        shutil.copy(args.remap, os.path.join(stage, "vocab_remap.npy"))

    ort_src = _copy_runtime("onnxruntime", os.path.join(stage, "onnxruntime"), ORT_STRIP)
    tk_src = _copy_runtime("tokenizers", os.path.join(stage, "tokenizers"),
                           ("__pycache__", "*.pyc"))
    print(f"runtimes bundled from:\n  {ort_src}\n  {tk_src}", flush=True)

    selfcheck(stage)          # before tarring: an unshippable tree must not become a tarball
    # selfcheck imports main.py, which leaves __pycache__ behind; it is tiny but it is bytecode
    # compiled by THIS interpreter, and shipping it invites a stale-cache mismatch on Kaggle's.
    _pyc = [os.path.join(d, "__pycache__") for d, subs, _ in os.walk(stage)
            if "__pycache__" in subs]
    for _d in _pyc:                       # collect first: rmtree during os.walk breaks the walk
        shutil.rmtree(_d, ignore_errors=True)

    tar_path = os.path.join(args.out, args.tag + ".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tf:
        for entry in sorted(os.listdir(stage)):
            tf.add(os.path.join(stage, entry), arcname=entry)

    # per-component compressed cost, so an over-budget build says WHAT to cut
    print("\ncompressed contribution:")
    for entry in sorted(os.listdir(stage)):
        tmp = tar_path + ".probe"
        with tarfile.open(tmp, "w:gz") as tf:
            tf.add(os.path.join(stage, entry), arcname=entry)
        print(f"  {entry:20s} {_mb(os.path.getsize(tmp)):>12s}")
        os.remove(tmp)

    n = os.path.getsize(tar_path)
    print(f"\n-> {tar_path}\n   {_mb(n)} / cap {_mb(TAR_CAP)}  "
          f"({100 * n / TAR_CAP:.0f}%)  {'OK' if n < TAR_CAP else 'OVER BUDGET'}")
    if n >= TAR_CAP:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
