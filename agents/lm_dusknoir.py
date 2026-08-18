"""Deck agent: dragapult_dusknoir piloted by the fine-tuned cross-encoder reranker.

Unlike every other file in agents/, this one is NOT generated from tuning.json -- it exists so
play_server can put a trained model in the AI seat and a person can play against it. Select it
on the /manage page (ai_agent = lm_dusknoir, ai_deck = dragapult_dusknoir) and pick whatever
deck you want to play against it with.

Which model it loads, in order:
    PTCG_MODEL=/path/to/checkpoint    a directory (e.g. a Colab result synced from Drive)
    PTCG_MODEL=user/repo              a HuggingFace repo id, downloaded on first use
    unset                             yoheikobashi/ptcg-dusknoir-deberta-reranker (the baseline)

    PTCG_LM_DEVICE  cpu | cuda | auto (default auto)
    PTCG_LM_WRAP    0 to drop the hand-authored plan rules (default: on, as shipped)
    PTCG_LM_MAXLEN  prompt truncation length (default 512, what the model was trained at)

If the model cannot be loaded -- no torch, no weights, no network -- the deck still plays, as
the plain heuristic engine. Losing the model must not mean losing the game.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "cg-lib"), os.path.join(_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lm.dusk_pilot import DECK_NAME, make_pilot  # noqa: E402

DEFAULT_MODEL = "yoheikobashi/ptcg-dusknoir-deberta-reranker"


def _build():
    src = os.environ.get("PTCG_MODEL", DEFAULT_MODEL)
    wrap = os.environ.get("PTCG_LM_WRAP", "1") not in ("", "0")
    try:
        from lm.hf_scorer import HfRerankerScorer, resolve_model
        path = resolve_model(src)
        model = HfRerankerScorer(path, device=os.environ.get("PTCG_LM_DEVICE", "auto"),
                                 max_len=int(os.environ.get("PTCG_LM_MAXLEN", "512")))
        print("[lm_dusknoir] model %s on %s (wrap %s)"
              % (src, model.device, "on" if wrap else "off"))
        return make_pilot(model=model, wrap=wrap)
    except Exception as e:  # noqa: BLE001
        print("[lm_dusknoir] model unavailable (%r) -- playing as engine_v2" % (e,))
        return make_pilot(model=None)


_AGENT = _build()


def agent(obs_dict: dict) -> list[int]:
    return _AGENT(obs_dict)
