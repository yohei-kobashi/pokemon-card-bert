"""Kaggle submission entrypoint: crustle piloted by the Qwen3.5 CPU scorer.

Layout (all under /kaggle_simulations/agent/ at run time):
    main.py            <- this file
    lm/                <- serialize / actions / vocab / agent / scorer
    agents/            <- engine_v2.py, _engine.py, tuning.json (engine pilot + fallback)
    cg/                <- game api
    decks/crustle.csv  <- our 60-card deck
    model.gguf         <- Qwen3.5-0.8B Q4_K_M
    llama_cpp/         <- bundled llama-cpp-python (glibc 2.35 / py3.11, matches Kaggle)

Every real decision is scored by the LLM (argmax over legal candidates -> always legal).
A cumulative time bank (LlamaScorer.time_budget) falls back to the engine before the
600s/game forfeit line, and any scorer exception also falls back -- so we never forfeit.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# make the bundled llama_cpp + local packages importable BEFORE anything else
for p in (HERE, os.path.join(HERE, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

import json

from lm.agent import make_lm_agent
from lm.scorer import LlamaScorer


def _load_deck(name):
    for p in (os.path.join(HERE, "decks", name + ".csv"),
              os.path.join(HERE, "deck.csv"),
              "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(p):
            with open(p) as f:
                return [int(x) for x in f if x.strip()]
    raise FileNotFoundError("deck not found for " + name)


DECK_NAME = "crustle"
GGUF = os.path.join(HERE, "model.gguf")

_deck = _load_deck(DECK_NAME)
_tuning = json.load(open(os.path.join(HERE, "agents", "tuning.json")))
_profile = _tuning.get(DECK_NAME, {})

# Build the scorer once (model load ~1s). On ANY failure, ship the pure-engine pilot so
# the submission still plays rather than crashing at import.
try:
    _scorer = LlamaScorer(GGUF, n_threads=4, time_budget=480.0)
    _agent = make_lm_agent(_deck, _profile, model=_scorer)
except Exception:
    _agent = make_lm_agent(_deck, _profile, model=None)


def agent(obs_dict: dict) -> list:
    return _agent(obs_dict)
