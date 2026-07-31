"""LM shared foundation for the Qwen3.5 agent (serializer / actions / vocab / adapter).

See docs/ml_agent_plan.md sec.4. These modules are shared by build_sft.py (training)
and the live agent (inference/submission) so train/inference distributions match.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "cg-lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
