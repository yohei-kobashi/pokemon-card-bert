"""Per-deck card roles AS SEEN BY THE PROMPT.

Deliberately separate from `card_roles`, which engine_v2 turns into search / discard priorities
via `_TIER_VALUE` and `_card_need`. engine_v2 is the SFT teacher, the RL opponent, the playout
policy and the in-agent fallback, so a prompt experiment that edited `card_roles` would move its
own control. `prompt_roles` is read ONLY here.

Until a deck defines `prompt_roles` the resolver returns `card_roles`, so the format work can
start on labels that already exist for 62 decks at 100% coverage of their distinct cards, with
20% of cards carrying a different role in different decks (Dunsparce is engine / line / tech
depending on the deck; Latias ex is engine / tech / win).
"""

# Order is the rendering order, not a priority: the prompt shows groups in this sequence so a
# card's ROLE is readable from its POSITION. Anything unlabelled lands in the trailing bucket.
ROLE_ORDER = ("win", "engine", "line", "fuel", "tech", "filler")
UNLABELLED = "other"


def resolve(profile):
    """{card_id(int): role(str)} for one deck's profile dict."""
    raw = (profile or {}).get("prompt_roles")
    if raw is None:
        raw = (profile or {}).get("card_roles") or {}
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


def group(card_ids, roles):
    """[(role, [card_id, ...]), ...] in ROLE_ORDER, ids sorted WITHIN each group.

    Sorting by id (not by the decklist file's order) is what removes the fingerprint that
    `--deck-shuffle` exists to break: the order becomes a function of the contents, so it cannot
    encode which deck this is beyond what the contents already say.
    """
    buckets = {}
    for cid in card_ids:
        buckets.setdefault(roles.get(cid, UNLABELLED), []).append(cid)
    ordered = []
    for r in ROLE_ORDER:
        if r in buckets:
            ordered.append((r, sorted(buckets.pop(r))))
    for r in sorted(buckets):                       # UNLABELLED and anything unexpected, last
        ordered.append((r, sorted(buckets[r])))
    return ordered


_TUNING = None


def for_deck(name):
    """Roles for a deck NAME, read from agents/tuning.json (cached).

    Looked up rather than passed so a caller cannot forget it: the prompt format already lives
    in build_rerank, build_sft, lm/agent and rl_rollout, and a mismatch between any two of them
    is silent. tuning.json ships in the submission bundle.
    """
    global _TUNING
    if not name:
        return {}
    if _TUNING is None:
        import json
        import os as _os
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        try:
            _TUNING = json.load(open(_os.path.join(root, "agents", "tuning.json")))
        except Exception:
            _TUNING = {}
    return resolve(_TUNING.get(name) or {})
