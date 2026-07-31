"""Opponent identification for the LM prompt: which deck / archetype are we facing?

The LM sees only the board. It cannot know the opponent's list, but the cards they have
REVEALED pin it down fast: measured on self-play, archetype is right **82.8% on turn 1
and 97.6% by turn 3**, exact deck 74.6% / 96.0% (tools/predict_archetype.py). Live, on
766 real ladder games, 98.9% where the label was checkable, and **95% of games show ZERO
cards outside our 60-deck pool**, so deck-matching covers the field.

Emission rule (user-directed): give BOTH deck and archetype, and when the posterior is
not confident enough, emit SEVERAL candidates rather than one wrong one. The prompt
therefore carries either `d_alakazam_xero:9` or, when it is genuinely ambiguous,
`d_crustle:5 d_crustle_stall:3`.

CRITICAL — train and inference must both use the PREDICTION. Building the training data
with the true opponent deck would teach the model to trust a feature it cannot have live;
this module is the single source for both paths.
"""
import collections
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

EPS = 0.25          # smoothing: an off-list tech card must not delete the right deck
TOP_P = 0.90        # keep candidates until this much posterior mass is covered
MAX_CAND = 3        # never emit more than this many
MIN_SHOW = 0.10     # ...and never emit a candidate below this probability

_STATE = {}


def _read_deck(name):
    """Card ids from decks/<name>.csv, or [] — the same bytes ``library.read_deck`` returns.

    Read directly rather than through ``library``: that module imports ``battle_log``, which
    pulls in the whole training stack, none of which ships in the Kaggle bundle. With
    ``import library`` here the bundle raised ModuleNotFoundError inside ``_fleet``,
    ``serialize._identify`` swallowed it (it is documented never to raise), and EVERY prompt
    silently lost the ``ID ME d_x a_y OP ...`` segment — present in 100% of training rows and
    the one segment ablation shows the model actually uses (-3.0pt top1 without it). No crash,
    no size change, just a worse pilot. So the fleet must load from files the bundle ships.
    """
    try:
        with open(os.path.join(ROOT, "decks", name + ".csv")) as f:
            return [int(ln) for ln in f if ln.strip()]
    except OSError:
        return []


def _fleet():
    """{deck_name: (archetype, Counter(card_id -> copies))} — built once.

    Every deck in tuning.json must resolve, in the bundle too: the posterior is over the decks
    we know, so a fleet missing 62 of 63 lists would not fail, it would confidently name the
    only deck it has. build_rerank_submission.py therefore ships all of decks/ (276 KB)."""
    if "fleet" in _STATE:
        return _STATE["fleet"]
    tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    fleet = {}
    for name, cfg in tun.items():
        if not isinstance(cfg, dict) or not cfg.get("archetype"):
            continue
        d = _read_deck(name)
        if len(d) == 60:
            fleet[name] = (cfg["archetype"], collections.Counter(d))
    _STATE["fleet"] = fleet
    return fleet


def _tables():
    """Per-card log-likelihood rows, precomputed once."""
    if "ll" in _STATE:
        return _STATE["ll"], _STATE["names"], _STATE["arch"], _STATE["miss"]
    fleet = _fleet()
    names = sorted(fleet)
    cards = set()
    for _a, c in fleet.values():
        cards |= set(c)
    Z = 60.0 + EPS * len(cards)
    ll = {c: [math.log((fleet[n][1].get(c, 0) + EPS) / Z) for n in names] for c in cards}
    _STATE.update(ll=ll, names=names,
                  arch={n: fleet[n][0] for n in names}, miss=math.log(EPS / Z))
    return ll, names, _STATE["arch"], _STATE["miss"]


def observed_cards(st, opp):
    """Every opponent card we can SEE: board bodies (+ their pre-evolutions, attached
    energy and tools) and the discard pile. Their hand, deck and prizes are hidden."""
    c = collections.Counter()
    ps = st.players[opp]
    for pk in list(ps.active or []) + list(ps.bench or []):
        if pk is None:
            continue
        if getattr(pk, "id", None) is not None:
            c[pk.id] += 1
        for grp in ("energyCards", "tools", "preEvolution"):
            for x in (getattr(pk, grp, None) or []):
                if getattr(x, "id", None) is not None:
                    c[x.id] += 1
    for d in (ps.discard or []):
        if getattr(d, "id", None) is not None:
            c[d.id] += 1
    return c


def identify(st, opp):
    """-> ([(deck, p), ...], [(archetype, p), ...]) — the candidates worth showing.

    Memoised on the observed multiset: consecutive decisions in a game usually reveal
    nothing new, and this runs on every sample of a multi-million-row build."""
    obs_c = observed_cards(st, opp)
    key = tuple(sorted(obs_c.items()))
    memo = _STATE.setdefault("memo", {})
    hit = memo.get(key)
    if hit is not None:
        return hit
    ll, names, arch, miss = _tables()
    s = [0.0] * len(names)
    for c, k in obs_c.items():
        row = ll.get(c)
        for i in range(len(names)):
            s[i] += k * (row[i] if row else miss)
    m = max(s)
    e = [math.exp(v - m) for v in s]
    tot = sum(e) or 1.0
    dpost = sorted(((names[i], e[i] / tot) for i in range(len(names))),
                   key=lambda x: -x[1])
    apost = collections.defaultdict(float)
    for n, p in dpost:
        apost[arch[n]] += p
    apost = sorted(apost.items(), key=lambda x: -x[1])
    out = (_cut(dpost), _cut(apost))
    if len(memo) < 200000:
        memo[key] = out
    return out


def _cut(ranked):
    """Top candidates until TOP_P of the mass is covered (<=MAX_CAND).

    Returns EMPTY when the leader is under MIN_SHOW: before anything is revealed the
    posterior is near-uniform over 60 decks, and printing `d_alakazam:0` there states a
    guess the evidence does not support. An empty list renders as `?`."""
    if not ranked or ranked[0][1] < MIN_SHOW:
        return []
    out, acc = [], 0.0
    for name, p in ranked:
        if out and (acc >= TOP_P or p < MIN_SHOW or len(out) >= MAX_CAND):
            break
        out.append((name, p))
        acc += p
    return out


def render(st, me, my_deck_name=None):
    """The prompt segment. Our OWN side is known exactly; the opponent is predicted."""
    from lm import vocab
    parts = []
    if my_deck_name:
        a = _fleet().get(my_deck_name, (None, None))[0]
        parts.append("ME " + vocab.deck_tok(my_deck_name)
                     + (" " + vocab.arch_tok(a) if a else ""))
    decks, arches = identify(st, 1 - me)
    # probability as a single digit 1-9 (round, not floor: 0.15 must not print as 0)
    q = lambda p: max(1, min(9, int(round(p * 10))))            # noqa: E731
    parts.append("OP " + (" ".join(f"{vocab.deck_tok(n)}:{q(p)}" for n, p in decks)
                          if decks else "?"))
    if arches:
        parts.append(" ".join(f"{vocab.arch_tok(a)}:{q(p)}" for a, p in arches))
    return " ".join(parts)
