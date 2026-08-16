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
LM_EMBED = ("vocab", "costs", "actions", "damage", "hidden", "identify", "roles",
            "action_token", "serialize", "rerank_scorer", "plan_filter", "agent")
# Embedded as TOP-LEVEL modules, not under lm/. tools/dusk_plan.py holds the hand-authored plan
# rules, and lm/plan_filter.py resolves it by bare name at call time precisely so it can be
# installed here without the research tree coming with it. Only pulled in when --wrap asks for
# it: a bundle with no wrapper must not carry 705 lines of rules it never calls.
PLAN_EMBED = ("dusk_plan",)
# costs/damage/hidden/action_token were added to lm/ after this list was written and never added
# to it. Only `costs` failed loudly (serialize imports it at module level, and the selfcheck's
# tier assertion caught that). The other three are imported LAZILY inside the functions that
# render the prompt -- hidden -> hidden_facts, damage -> board_facts, action_token -> menu_dedup,
# all three ON in the current format -- so their absence silently deletes prompt segments the
# model was trained on 100% of the time, which is the exact failure this embedding scheme exists
# to prevent. Keep the tuple in DEPENDENCY ORDER: each module is exec'd when its turn comes.
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

# Cap CPU threading BEFORE any native lib loads: the grader gives 2 vCPU and an unpinned
# OpenMP/BLAS pool oversubscribes and thrashes. This file previously said 4 -- an
# unsourced number repeated from tools/bench_rerank_onnx.py, which is where every speed
# projection in the reranker plan came from. Benching at 4 threads on a box with spare
# cores measures a machine that does not exist and reports roughly half the real cost.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "{threads}")

# tools/dusk_plan.py gates the BEHAVIOUR of six rules on these two flags (_NEW_EXCL and
# _CLOPS_HOLD, read at import). RULES still lists all sixteen names either way, so a wrapper
# naming a gated rule does NOT raise -- the rule simply returns an empty set and silently
# constrains nothing. Every training and gating run exports them; submission 55445834 did not,
# and the audit of its 19 live games shows the cost exactly: lethal_now, clops_hold,
# judge_timing, spare_ex_bench, retreat_energy and stadium_replace fired ZERO times, so two of
# the five rules the shipped wrapper spec names were inert on the ladder. Set BEFORE dusk_plan
# is imported, which is why this sits with the thread caps rather than next to the wrapper.
for _v in ("DUSK_NEW_RULES", "DUSK_CLOPS_HOLD", "DUSK_FRONT_DIVE", "DUSK_BOSS_LETHAL",
           "DUSK_TIPS", "DUSK_SPIKE"):
    os.environ.setdefault(_v, "1")

import sys
import types

def _find_here():
    """The bundle's own directory, without assuming __file__ exists.

    Kaggle EXECs main.py instead of importing it, so __file__ is never bound. The old one-liner
    sat above every try/except in this file, so that NameError killed the module at load:
    `agent` was never defined, the three-tier fallback below never ran, and the harness recorded
    action=null at step 1 with the 600 s bank untouched. That is the signature of ALL SEVEN LM
    submissions to date -- including the ones whose notes say the deck-selection call was
    already fixed. It was. It never got the chance to run. The engine_v2 bundles score because
    they resolve their paths from a candidate LIST and never touch __file__.

    Probed by CONTENT, not by guessing one path: a directory is only accepted if it holds the
    two files this bundle cannot run without.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    cands = ["/kaggle_simulations/agent", os.getcwd()]
    if sys.argv and sys.argv[0]:
        cands.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    cands += [p for p in sys.path if p]
    for _d in cands:
        try:
            if (os.path.exists(os.path.join(_d, "main.py"))
                    and os.path.exists(os.path.join(_d, "deck.csv"))):
                return os.path.abspath(_d)
        except Exception:
            continue
    return os.getcwd()


HERE = _find_here()
for _p in (HERE, os.path.join(HERE, "cg-lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _install_local_pkg(name):
    """Bind `name` to THIS bundle's copy, whatever else is on sys.path.

    Kaggle's runtime ships kaggle_environments/envs/lux_ai_s3/agents.py and its directory
    precedes ours in the grader's sys.path, so a plain `import agents` resolves to THAT file --
    which then dies on its own relative import ("attempted relative import with no known parent
    package") and takes us down with it. Every fallback tier routes through
    `from agents.engine_v2 import make_policy` (tier 3 directly, tiers 1-2 via lm/agent), so one
    shadowed name killed all three at once and the bundle could not even reach engine_v2.

    Registering the package by FILE PATH removes the dependency on path order entirely -- the
    same reason lm/ is exec'd into sys.modules below rather than imported.
    """
    import importlib.util
    _d = os.path.join(HERE, name)
    _init = os.path.join(_d, "__init__.py")
    if not os.path.exists(_init):
        return False
    _spec = importlib.util.spec_from_file_location(name, _init, submodule_search_locations=[_d])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[name] = _mod
    _spec.loader.exec_module(_mod)
    return True


for _pkg in ("agents", "cg"):
    try:
        _install_local_pkg(_pkg)
    except Exception:
        pass          # a shadowed name is survivable; a raise here would not be

import json

DECK_NAME = {deck!r}
# Baked from tools/rl_config.PROMPT_FMT at BUILD time. The prompt format is part of the model and
# lives in four renderers; passing it as loose command-line flags is how train and deploy drift
# apart without anything failing.
PROMPT_FMT = {pfmt!r}
# Action kinds handed to engine_v2 instead of the model. Baked at build time for the same reason
# the prompt format is: a routing that was measured on one configuration and shipped on another
# is not the thing that was measured.
DEFER_KINDS = {defer!r}
# The plan-rule wrapper, e.g. "planfilter:lethal_now,spread_aim,...". Baked for the same reason
# as the two above: this one is not a tuning knob but a POLICY -- it takes whole decision families
# off the model -- and a bundle built with a different rule list is a different pilot.
WRAP = {wrap!r}

_LM_ORDER = {lm_order!r}
_LM_SRC = {lm_src!r}
_PLAN_SRC = {plan_src!r}


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


def _install_plan():
    """Install the plan modules under their BARE names, the way lm/plan_filter looks for them.

    They live in tools/ in the repo and there is no tools package here, so the import that
    resolves in the repo (`import dusk_plan`, with tools/ on sys.path) is made to resolve the
    same way by registering the source directly. Exec'd with __name__ set to the module name so
    the file's own `if __name__ == "__main__"` CLI block stays dormant.
    """
    for _name, _src in _PLAN_SRC.items():
        _m = types.ModuleType(_name)
        _m.__file__ = os.path.join(HERE, _name + ".py")
        sys.modules[_name] = _m
        exec(compile(_src, _m.__file__, "exec"), _m.__dict__)


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
        _agent = make_lm_agent(_deck, _profile, model=_scorer, deck_name=DECK_NAME,
                               defer_kinds=DEFER_KINDS, **PROMPT_FMT)
        TIER = "reranker" if not DEFER_KINDS else "reranker+defer:" + ",".join(DEFER_KINDS)
        if WRAP:
            # Deliberately its own try. A wrapper that cannot load must cost us the WRAPPER, not
            # the model: dropping to engine_v2 over it would trade a measured 62.0% pilot for a
            # measured 55% one. The build-time selfcheck asserts WRAP is in TIER, so this can
            # only ever fire on Kaggle, and the tier string then says so instead of lying.
            try:
                _install_plan()
                from lm.plan_filter import make_plan_rule
                _head, _, _rules = WRAP.partition(":")
                _agent = make_plan_rule(_agent, _rules.split(","),
                                        strict=(_head == "planrule"))
                TIER += "+" + WRAP
            except Exception:
                TIER += "+wrapfailed"
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
        # Forward it FIRST, for the side effect: lm/agent.py treats select=None as "new game"
        # and refills the scorer's time bank there. The old fast path returned _deck without
        # ever reaching the pilot, so reset_bank() was unreachable from a bundle and `spent`
        # accumulated across every episode the process served -- measured on the staged tree at
        # 49.7 -> 94.4 -> 166.1 -> 221.5 -> 249.9 s over five consecutive games. Nothing errors
        # when that budget runs out; the scorer raises, lm/agent catches it, and the rest of the
        # match is played by engine_v2 while the tier string still says "reranker".
        try:
            _agent(obs_dict)
        except Exception:
            pass
        # The return value is OURS regardless: the deck contract is 60 ints, and a fallback tier
        # or a wrapper returning something else must not be able to break it.
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
assert M.TIER.startswith("reranker"), "scorer did not load; this bundle would play as engine_v2"
# TIER is "reranker" ["+defer:<kinds>"] ["+<wrap>"], and it is the ONLY runtime evidence of what
# the bundle actually built -- every layer below the reranker still plays legal games. Parse it
# and prove it describes the constants, rather than pattern-matching a prefix.
_parts = M.TIER.split("+")
_tdefer = [p.partition(":")[2] for p in _parts if p.startswith("defer:")]
_twrap = [p for p in _parts[1:] if not p.startswith("defer:")]
assert set(M.DEFER_KINDS or ()) == set(_tdefer[0].split(",") if _tdefer else []) - {""}, \
    "TIER %r does not describe DEFER_KINDS %r" % (M.TIER, M.DEFER_KINDS)
assert _twrap == ([M.WRAP] if M.WRAP else []), \
    "TIER %r does not describe WRAP %r (wrapfailed = the plan did not load)" % (M.TIER, M.WRAP)

# The wrapper takes whole decision families off the model, so "it imported" is not enough: the
# rule NAMES have to exist in the plan that shipped. A typo'd or renamed rule raises inside
# make_plan_rule, which the runtime catches into +wrapfailed -- caught above -- but a rule that
# exists under a stale definition would not, so print the plan's full rule set for the record.
if M.WRAP:
    import dusk_plan                            # noqa: E402  installed by main.py's _install_plan
    _want = M.WRAP.partition(":")[2].split(",")
    _missing = [r for r in _want if r not in dusk_plan.RULES]
    assert not _missing, "wrap names rules the bundled plan does not have: %r" % (_missing,)
    # Name-checking RULES is NOT enough, and believing it was is what shipped 55445834 with two
    # of its five wrapper rules dead. dusk_plan reads DUSK_NEW_RULES / DUSK_CLOPS_HOLD at import
    # and, when they are unset, keeps every rule NAME while emptying six rules' good sets -- so
    # the wrapper builds, the tier string is honest, the selfcheck passes, and the rules do
    # nothing. Assert the gates themselves, in the interpreter that just imported main.py.
    for _flag in ("_NEW_EXCL", "_CLOPS_HOLD", "_FRONT_DIVE", "_BOSS_LETHAL",
                  "_TIPS", "_SPIKE"):
        assert getattr(dusk_plan, _flag, False), (
            "dusk_plan.%s is False: main.py did not set the env flag before importing the plan, "
            "so gated rules would silently constrain nothing" % _flag)
    print("SELFCHECK wrap %s | plan carries %d rules | gates _NEW_EXCL/_CLOPS_HOLD both live"
          % (M.WRAP, len(dusk_plan.RULES)))

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

# 4. THE GRADER EXECS main.py; it does not import it. Under exec there is no __file__, and every
#    check above binds one by importing -- which is why seven LM submissions passed this file and
#    then died at module load on a bare NameError, before any of the three fallback tiers could
#    run. Run the source the way the grader runs it.
src = open(os.path.join(HERE, "main.py")).read()
g = {"__name__": "__main__"}
exec(compile(src, "main.py", "exec"), g)          # noqa: S102 -- this IS the thing under test
assert "agent" in g, "exec of main.py defined no agent()"
d2 = g["agent"](first)
assert isinstance(d2, list) and len(d2) == 60 and all(isinstance(x, int) for x in d2), \
    "exec-mode deck-selection returned %r" % (type(d2).__name__,)
assert str(g.get("TIER", "")).startswith("reranker"), \
    "exec-mode fell back to tier %r" % (g.get("TIER"),)
print("SELFCHECK exec-mode OK (tier %s)" % g.get("TIER"))

# 5. The grader's sys.path holds kaggle_environments/envs/lux_ai_s3/agents.py AHEAD of the
#    bundle, so a plain `import agents` binds THAT file, which dies on its own relative import
#    and takes us with it. Every tier routes through `from agents.engine_v2 import make_policy`,
#    so the shadow killed the reranker, the engine-through-lm tier and the bare engine tier
#    together -- submission 8, immediately after the __file__ bug was fixed. Reproduce the
#    shadow and prove the bundle still binds its own copy.
import tempfile                                   # noqa: E402
_decoy = tempfile.mkdtemp()
with open(os.path.join(_decoy, "agents.py"), "w") as _f:
    _f.write("from .test_agents.python.main import agent_fn\n")   # lux_ai_s3's exact shape
sys.path.insert(0, _decoy)
for _m in [k for k in list(sys.modules) if k == "agents" or k.startswith("agents.")]:
    sys.modules.pop(_m, None)
g2 = {"__name__": "__main__"}
exec(compile(src, "main.py", "exec"), g2)
import agents as _ag                              # noqa: E402
assert os.path.abspath(getattr(_ag, "__file__", "")).startswith(os.path.abspath(HERE)), \
    "agents resolved to %r -- the bundle is shadowed" % getattr(_ag, "__file__", None)
d3 = g2["agent"](first)
assert isinstance(d3, list) and len(d3) == 60, "shadowed deck-selection returned %r" % (d3,)
assert str(g2.get("TIER", "")).startswith("reranker"), \
    "under a shadowed `agents` the bundle fell to tier %r" % (g2.get("TIER"),)
print("SELFCHECK shadow-resistant OK (tier %s)" % g2.get("TIER"))
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
    ap.add_argument("--defer", default="",
                    help="comma-separated action kinds routed to engine_v2 instead of the "
                         "model (e.g. 'attach'). Empty = pure LM.")
    ap.add_argument("--wrap", default="",
                    help="plan-rule wrapper, 'planfilter:<rule>,...' (the plan narrows the menu "
                         "and the model ranks inside) or 'planrule:<rule>,...' (the plan "
                         "decides). Empty = the model sees every menu.")
    ap.add_argument("--pfmt", default="current", choices=("current", "v37", "dusk"),
                    help="current = rl_config.PROMPT_FMT (what build_rerank just used); "
                         "v37 = rl_config.PROMPT_FMT_V37, for models trained before the rebuild")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=2, help="grader vCPUs; see main.py")
    ap.add_argument("--max-len", type=int, default=1024,
                    help="512 for DeBERTa-v3 (max_position_embeddings), which is what "
                         "HFRerankScorer clamps to -- exceed it and the deploy model "
                         "sees longer inputs than the checkpoint was ever scored on")
    ap.add_argument("--time-budget", type=float, default=480.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions"))
    ap.add_argument("--from-registry", action="store_true",
                    help="take --pfmt/--defer from this deck's models/adapters.json entry and "
                         "refuse to build if a flag contradicts it")
    args = ap.parse_args()

    import library
    if args.deck not in set(library.list_decks()):
        raise SystemExit(f"unknown deck: {args.deck!r}")

    if args.from_registry:
        # A per-deck adapter is gated with one prompt format and one defer set; shipping it with
        # another is the exact divergence this flag exists to make impossible. The ONNX file
        # itself still comes from --onnx: the registry names checkpoints, not exports.
        from lm import registry as _reg
        r = _reg.resolve(args.deck, require_exists=False)
        if r["source"] != "deck":
            raise SystemExit(f"--from-registry: no entry for {args.deck!r} in "
                             f"{_reg.registry_path()}; add one with tools/adapters.py set")
        want_defer = ",".join(r["entry"].get("defer") or [])
        want_wrap = (r["entry"].get("wrap") or "").rstrip(":")
        # the registry says "prompt"/"dusk"; this builder's flag says "current"/"dusk"
        want_pfmt = {"prompt": "current", "dusk": "dusk"}[r["fmt"]]
        for flag, given, wanted in (("--pfmt", args.pfmt, want_pfmt),
                                    ("--defer", args.defer, want_defer),
                                    ("--wrap", args.wrap, want_wrap)):
            if given not in ("", "current") and given != wanted:
                raise SystemExit(f"--from-registry: {flag}={given!r} contradicts the registry "
                                 f"({wanted!r}) for {args.deck}")
        args.pfmt, args.defer, args.wrap = want_pfmt, want_defer, want_wrap
        print(f"[reg] {args.deck} -> {r['spec']} (fmt {r['fmt']})", flush=True)

    from tools import rl_config
    pfmt = dict({"current": rl_config.PROMPT_FMT, "v37": rl_config.PROMPT_FMT_V37,
                 "dusk": rl_config.DUSK_FMT}[args.pfmt])
    defer = tuple(x for x in (s.strip() for s in args.defer.split(",")) if x)
    wrap = args.wrap.strip().rstrip(":")
    if wrap:
        head, _, rules = wrap.partition(":")
        if head not in ("planfilter", "planrule"):
            # planengine is the matched CONTROL for the wrapper experiment -- it routes the
            # rule's menus to engine_v2. Measuring with it is the point; shipping it would ship
            # the control instead of the arm.
            raise SystemExit(f"--wrap: {head!r} is not shippable "
                             f"(use planfilter:<rules> or planrule:<rules>)")
        if not [r for r in rules.split(",") if r.strip()]:
            raise SystemExit("--wrap: no rules named")
    print(f"prompt format ({args.pfmt}): {pfmt}", flush=True)
    print(f"deferred to engine_v2: {list(defer) or 'nothing (pure LM)'}", flush=True)
    print(f"plan wrapper: {wrap or 'none (the model sees every menu)'}", flush=True)

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
    plan_src = {}
    if wrap:
        for name in PLAN_EMBED:
            plan_src[name] = open(os.path.join(ROOT, "tools", name + ".py")).read()
        print(f"plan modules embedded: {len(plan_src)} "
              f"({sum(len(v) for v in plan_src.values()) / 1024:.0f} KiB)", flush=True)

    with open(os.path.join(stage, "main.py"), "w") as f:
        f.write(MAIN_TEMPLATE.format(deck=args.deck, tag=args.tag, pfmt=pfmt, defer=defer,
                                     wrap=wrap, plan_src=plan_src,
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
