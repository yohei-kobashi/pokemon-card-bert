"""LM-agent adapter (component D): ``agent(obs_dict) -> list[int]``.

The SAME adapter serves local evaluation and the Kaggle submission. With
``model=None`` it is a pure engine_v2 agent (the SHIPPED pilot). With a model, EVERY
real decision is made by SCORING the legal candidates and taking the argmax -- so the
output is always a legal move (no free-generation, no illegal-output fallback needed).

STATELESS: the prompt is ``serialize_stateless(obs)`` (current board only, full card
rules, no episode history) -- exactly what build_sft trains on, tagged ``[ACT]``.

Decision shapes (all reduced to single-pick scoring):
  - exactly-one (minCount==maxCount==1, incl. MAIN): score each option, take argmax.
  - multi-pick / optional (choose k of n, or up-to-1): pick ONE AT A TIME -- score the
    not-yet-picked options (+ a STOP candidate once the min is satisfied), take the
    best, repeat; the intermediate state is built by ``serialize.multipick_substate``,
    the SAME builder build_sft trains on, so train and inference prompts match.
  - forced / no real choice (n_options < 2): resolved by engine_v2, never sent to the
    model (build_sft skips these too).

``model`` exposes ``score(prompt, candidates: list[str], obs=None) -> list[float]``
(higher = more likely; length-normalized log-prob recommended). The real model scores
from the prompt alone and ignores ``obs``; it is passed only so a test stand-in can
consult the heuristic. ``deck`` is the card-id list; ``profile`` is the tuning.json entry.
"""
from agents.engine_v2 import make_policy
from lm.serialize import serialize_stateless, multipick_substate, STOP
from lm.actions import encode_option

_ACT_PROMPT = "[ACT]\n"


def _real_choice(sel):
    """True if this select is a genuine decision the model was trained on: at least
    two options AND you don't have to take them all. Everything else is forced."""
    opts = sel.get("option") or []
    n = len(opts)
    if n < 2:
        return False
    lo = sel.get("minCount", 1) or 0
    return lo < n                       # must-take-all (lo >= n) is forced


def _argmax(xs):
    best_i = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best_i]:
            best_i = i
    return best_i


def _dedup(texts):
    """-> (unique texts, first original index of each).

    Two menu entries often encode to the SAME string -- three copies of the same card in
    hand give three ``evolve:c742@BENCH1`` options (measured: 9.3% of all candidate texts
    are duplicates). A cross-encoder re-encodes the ENTIRE state once per candidate, so
    each duplicate is a full wasted forward pass. Identical text -> identical score, so
    taking the argmax over the unique list and mapping back to the first occurrence picks
    the same move. build_rerank._emit dedups the training records the same way."""
    seen, uniq, pos = set(), [], []
    for i, t in enumerate(texts):
        if t not in seen:
            seen.add(t)
            uniq.append(t)
            pos.append(i)
    return uniq, pos


def make_lm_agent(deck, profile=None, model=None, glossary="full", deck_name=None,
                  deck_glossary=True, deck_mode="static", deck_shuffle=False,
                  board_facts=False, identify="both"):
    """``glossary`` / ``deck_name`` / ``deck_glossary`` MUST match what build_rerank or
    build_sft rendered the TRAINING data with -- the prompt format is part of the model,
    not a runtime option.

    ``deck_glossary=False`` reverts to the visible-only (v1) glossary. That is what every
    rerank record was actually built with: build_rerank read the deck from the per-step obs,
    which carries only ``deckCount``, so it always passed an EMPTY deck list and
    glossary_ids fell back to visible-only. Inference meanwhile passed the real 60 cards,
    pushing the prompt to ~1394 tokens against a 1024 truncation that cuts from the RIGHT --
    deleting the board and the option menu (measured: the SEL menu survived in 1% of
    decisions). Set False to make inference match that data."""
    policy = make_policy(deck, profile or {})     # engine_v2 = shipped pilot + fallback
    deck_ids = deck if deck_glossary else None
    _ser = lambda o: serialize_stateless(o, deck_ids=deck_ids, glossary=glossary,  # noqa: E731
                                         deck_name=deck_name, deck_mode=deck_mode,
                                         deck_shuffle=deck_shuffle,
                                         board_facts=board_facts, identify=identify)

    def _score_pick(obs):
        sel = obs["select"]
        opts = sel.get("option") or []
        lo = sel.get("minCount", 1) or 0
        hi = sel.get("maxCount", 1) or 1
        prompt = _ACT_PROMPT + _ser(obs)
        if lo == 1 and hi == 1:                        # exactly one -> argmax
            uniq, pos = _dedup([encode_option(o, obs) for o in opts])
            scores = model.score(prompt, uniq, obs)
            if not scores or len(scores) != len(uniq):
                return None
            return [pos[_argmax(scores)]]
        # multi-pick / optional -> one at a time
        picked = []
        while len(picked) < hi:
            sub, remaining, allow_stop = multipick_substate(obs, picked)
            if not remaining:
                break
            uniq, pos = _dedup([encode_option(opts[i], obs) for i in remaining])
            cands = uniq + [STOP] if allow_stop else uniq   # STOP never collides
            scores = model.score(_ACT_PROMPT + _ser(sub), cands, obs)
            if not scores or len(scores) != len(cands):
                return None
            j = _argmax(scores)
            if allow_stop and j == len(cands) - 1:     # chose STOP
                break
            picked.append(remaining[pos[j]])
        return picked if len(picked) >= lo else None   # never return an illegal short pick

    def agent(obs_dict):
        sel = obs_dict.get("select")
        if sel is None:                                # deck-selection phase = new game
            if model is not None:
                reset = getattr(model, "reset_bank", None)
                if reset is not None:
                    reset()                            # refill the per-game time bank
            return deck
        if model is None or not _real_choice(sel):
            return policy.act(obs_dict)                # no model, or a forced/trivial select
        try:
            idx = _score_pick(obs_dict)
            if idx is not None:
                return idx
        except Exception:
            pass
        return policy.act(obs_dict)                    # scoring failed/errored -> never forfeit

    return agent
