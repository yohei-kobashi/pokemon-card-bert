"""Make the submission bundle crash-proof and self-diagnosing.

Evidence (Kaggle validation episode 88561153 for rr_v37_dragapult, the ONLY LM submission ever
made): both agents ERROR at step 1 having consumed **0.083 s of the 600 s overage bank**, so no
timing explanation survives -- `actTimeout` is 0 (no per-move limit) and `runTimeout` is 2000 s
while a full game costs ~144 s. Locally every check passes: 436 MiB peak RSS, 180 MiB extracted,
main.py at the tar root, numpy the only non-vendored import. The failure is therefore something
that only happens in Kaggle's runtime, and the prime suspect is the vendored
`onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-...so`: a native module built for the
wrong CPython can SEGFAULT rather than raise, and a segfault is not catchable by
`except Exception` -- which is exactly why the existing fallback did not save the agent.

Two changes:

1. Probe `import onnxruntime` in a SUBPROCESS first. A crash there kills the child, the parent
   reads a non-zero exit code, and the agent continues with the engine_v2 fallback instead of
   dying. Only if the probe succeeds is onnxruntime imported in-process.
2. Print the environment (python version, platform, probe result) to stdout. The replay records
   agent stdout in `observation.logs`, so a failed run still tells us WHY -- the previous
   submission left `logs: []` and cost a submission slot for no information.

Net effect: one submission can no longer end in ERROR. It either returns the live LM baseline or
the diagnosis.
"""
import os

P = os.path.join(os.getcwd(), "tools/build_rerank_submission.py")
s = open(P).read()

if "_probe_onnxruntime" in s:
    print("already patched")
    raise SystemExit(0)

OLD = """    from lm.rerank_scorer import OnnxRerankerScorer"""
assert s.count(OLD) == 1, "scorer import anchor"

NEW = '''    if not _probe_onnxruntime():
        raise RuntimeError("onnxruntime probe failed in a subprocess -- see the log above")
    from lm.rerank_scorer import OnnxRerankerScorer'''
s = s.replace(OLD, NEW)

# insert the probe helper just before the try block that builds the scorer
ANCHOR = "try:\n    from lm.rerank_scorer import OnnxRerankerScorer"
if ANCHOR not in s:
    ANCHOR = "try:\n    if not _probe_onnxruntime():"
assert s.count(ANCHOR) == 1, "try anchor"

HELPER = '''def _probe_onnxruntime():
    """Import onnxruntime in a CHILD process first.

    The vendored extension is built for one CPython ABI. On a different interpreter it can
    SEGFAULT instead of raising, and a segfault cannot be caught by `except Exception` -- the
    whole agent dies and Kaggle records ERROR. Probing in a child means the crash costs a
    subprocess, not the submission. Everything printed here lands in the episode replay's
    `observation.logs`, so a failure is diagnosable without another submission.
    """
    import platform
    import subprocess
    print("[env] python %s | %s | %s"
          % (sys.version.split()[0], platform.platform(), platform.machine()), flush=True)
    try:
        r = subprocess.run([sys.executable, "-c",
                            "import onnxruntime,sys;"
                            "print(onnxruntime.__version__);"
                            "sys.stdout.flush()"],
                           cwd=HERE, capture_output=True, timeout=120,
                           env=dict(os.environ, PYTHONPATH=HERE))
    except Exception as e:
        print("[env] onnxruntime probe could not run: %r" % (e,), flush=True)
        return False
    ok = (r.returncode == 0)
    print("[env] onnxruntime probe rc=%s out=%r err=%r"
          % (r.returncode, r.stdout[:200].decode("utf8", "replace"),
             r.stderr[-400:].decode("utf8", "replace")), flush=True)
    return ok


'''
s = s.replace(ANCHOR, HELPER + ANCHOR)

# make the fallback loud instead of silent
OLD_EXC = """except Exception:
    _agent = make_lm_agent(_deck, _profile, model=None)"""
NEW_EXC = """except Exception as _e:
    # LOUD: under the LM-only directive a silent degrade to engine_v2 is worse than an error,
    # because the live score would be recorded as the LM's.
    import traceback
    print("[env] SCORER UNAVAILABLE -> engine_v2 fallback: %r" % (_e,), flush=True)
    traceback.print_exc()
    _agent = make_lm_agent(_deck, _profile, model=None)"""
assert s.count(OLD_EXC) == 1, "except anchor"
s = s.replace(OLD_EXC, NEW_EXC)

open(P, "w").write(s)
print("patched", P)
