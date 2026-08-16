"""Pass TARGET_MODE to build_sft's workers.

build_sft deliberately uses `mp.get_context("spawn")` (its `_init_worker` docstring records that
fork after the parent fitted the value scorer deadlocked every child), so a module global set in
main() never reaches a worker -- the first index-mode run silently emitted action strings.
"""
import os

P = os.path.join(os.getcwd(), "tools/_legacy_decoder/build_sft.py")
s = open(P).read()

A = "def _init_worker(value_model, turn_boundary):"
B = 'def _init_worker(value_model, turn_boundary, target_mode="action"):'
if B in s:
    print("already patched")
    raise SystemExit(0)
assert s.count(A) == 1, "init_worker signature"
s = s.replace(A, B)

A2 = '''    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"'''
B2 = '''    global TARGET_MODE          # workers are SPAWNED, so the parent's global never arrives
    TARGET_MODE = target_mode
''' + A2
assert s.count(A2) == 1, "env anchor"
s = s.replace(A2, B2)

A3 = "                      initargs=(value_model, turn_boundary)) as pool:"
B3 = "                      initargs=(value_model, turn_boundary, TARGET_MODE)) as pool:"
assert s.count(A3) == 1, "initargs anchor"
s = s.replace(A3, B3)

open(P, "w").write(s)
print("patched", P)
