"""Give the PROMPT its own role labels, separate from engine_v2's search priorities.

`card_roles` is consumed by `_TIER_VALUE` -> `_card_need` -> engine_v2's search / discard
ordering. engine_v2 is the SFT teacher, the RL opponent, the rollout policy inside rl_branch
playouts AND the in-agent fallback, so editing `card_roles` to improve a prompt would silently
move the labels, the objective and the deployed agent at the same time -- the experiment would
change its own control.

They also want different things. Today's audit found the Basic of an evolution line tiered BELOW
the evolution in 52 of 63 decks: as a SEARCH PRIORITY that is wrong (it fetches an unplayable
Stage 1 while the bench is empty), as a DESCRIPTION it is right (the evolution really is the
deck's win condition). One label cannot be both.

So: `prompt_roles` is a new, optional per-deck key. Until it exists the resolver falls back to
`card_roles`, so nothing changes today and the prompt experiment can start on free labels.
engine_v2 never reads `prompt_roles`.
"""
import os

P = os.path.join(os.getcwd(), "lm/roles.py")
if os.path.exists(P):
    print("already exists:", P)
    raise SystemExit(0)

SRC = '''"""Per-deck card roles AS SEEN BY THE PROMPT.

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
'''

open(P, "w").write(SRC)
print("wrote", P)
