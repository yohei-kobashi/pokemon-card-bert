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


def _dedup(texts, obs=None):
    """-> (unique texts, first original index of each).

    Two menu entries often encode to the SAME string -- three copies of the same card in
    hand give three ``evolve:c742@BENCH1`` options (measured: 9.3% of all candidate texts
    are duplicates). A cross-encoder re-encodes the ENTIRE state once per candidate, so
    each duplicate is a full wasted forward pass. Identical text -> identical score, so
    taking the argmax over the unique list and mapping back to the first occurrence picks
    the same move.

    Equal TEXT is not the only equal MOVE, though. `card:c305@DECK1` and `card:c305@DECK6`
    are two copies of one card in a shuffled pile; `facedown:PRIZE2` and `facedown:PRIZE3`
    are two face-down prizes. Collapsing those as well removes a further 5.6% of forward
    passes, and 5.65% of decisions turn out to have no choice in them at all.

    Board slots go the same way once the observation is available. Three copies of one Basic
    sitting on the bench at full HP with nothing attached are rendered identically in the
    prompt, so `attach:c7@BENCH0` and `attach:c7@BENCH2` differ only by a number that carries
    no information -- measured at 38.6% of attach decisions, capping a perfect model at 86.3%
    top1 on that kind. Passing ``obs`` collapses them; omitting it degrades to the text-only
    behaviour rather than failing.

    This MUST match build_rerank._emit and collect_dagger, which dedup the same way: a model
    trained with the twins collapsed has never had to rank them apart, so leaving them separate
    here would ask it for a comparison its training never contained."""
    from lm.action_token import dedup_options
    uniq, pos, _keys = dedup_options(texts, obs)
    return uniq, pos


def make_lm_agent(deck, profile=None, model=None, glossary="full", deck_name=None,
                  deck_glossary=True, deck_mode="static", deck_shuffle=False,
                  board_facts=False, identify="both", menu_dedup=False,
                  hidden_facts=False, defer_kinds=()):
    """``glossary`` / ``deck_name`` / ``deck_glossary`` MUST match what build_rerank or
    build_sft rendered the TRAINING data with -- the prompt format is part of the model,
    not a runtime option.

    ``deck_glossary=False`` reverts to the visible-only (v1) glossary. That is what every
    rerank record was actually built with: build_rerank read the deck from the per-step obs,
    which carries only ``deckCount``, so it always passed an EMPTY deck list and
    glossary_ids fell back to visible-only. Inference meanwhile passed the real 60 cards,
    pushing the prompt to ~1394 tokens against a 1024 truncation that cuts from the RIGHT --
    deleting the board and the option menu (measured: the SEL menu survived in 1% of
    decisions). Set False to make inference match that data.

    ``defer_kinds`` routes whole ACTION KINDS to engine_v2 instead of the model: a decision
    where any candidate encodes to one of these kinds is answered by the heuristic. This is
    not a fallback -- it is the shipped form of a measured result. The model's attach
    decisions rank at 16-29% top1 against a 14% chance baseline, and handing ONLY attach to
    engine_v2 was worth +11.4pt. The same mechanism has lived in tools/mirror_match.py as
    make_defer, where that number was measured, but never in the adapter that ships, so the
    win was unreachable from a submission. Passing () keeps the pure-LM behaviour."""
    policy = make_policy(deck, profile or {})     # engine_v2 = shipped pilot + fallback
    defer = frozenset(defer_kinds or ())
    deck_ids = deck if deck_glossary else None
    _ser = lambda o: serialize_stateless(o, deck_ids=deck_ids, glossary=glossary,  # noqa: E731
                                         deck_name=deck_name, deck_mode=deck_mode,
                                         deck_shuffle=deck_shuffle,
                                         board_facts=board_facts, identify=identify,
                                         menu_dedup=menu_dedup, hidden_facts=hidden_facts)

    def _score_pick(obs):
        sel = obs["select"]
        opts = sel.get("option") or []
        lo = sel.get("minCount", 1) or 0
        hi = sel.get("maxCount", 1) or 1
        prompt = _ACT_PROMPT + _ser(obs)
        if lo == 1 and hi == 1:                        # exactly one -> argmax
            uniq, pos = _dedup([encode_option(o, obs) for o in opts], obs)
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
            # `sub` is what gets serialized, so the descriptors must come from it too
            uniq, pos = _dedup([encode_option(opts[i], obs) for i in remaining], sub)
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
        if defer:
            # Kind is the head of the encoding, before ':' and before any '@target' -- the
            # same derivation make_defer uses, so a deferral measured there transfers here.
            for o in (sel.get("option") or []):
                if encode_option(o, obs_dict).split(":", 1)[0].split("@", 1)[0] in defer:
                    return policy.act(obs_dict)
        try:
            idx = _score_pick(obs_dict)
            if idx is not None:
                return idx
        except Exception:
            pass
        return policy.act(obs_dict)                    # scoring failed/errored -> never forfeit

    return agent
