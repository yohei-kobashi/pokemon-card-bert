"""v2 serialization: the COMPLETE observable board, one action out.

See docs/lm_input_format_v2.md. v1 emitted the hand as a bare count and dropped both
discards, appearThisTurn, preEvolution, tool identity and special-energy identity --
which made identical strings carry different correct actions (the model matched its own
training data only 70%). Domain tokens make a card 1 token, so putting all of it back
costs ~+67 tokens.

Everything the simulator exposes is legal to use: it already hides the opponent's hand
(handCount only), both prize piles (`[null,...]`) and both decks (deckCount only).
"""
from lm import vocab

_CONDS = (("PSN", "poisoned"), ("BRN", "burned"), ("SLP", "asleep"),
          ("PAR", "paralyzed"), ("CNF", "confused"))
_FLAGS = (("E", "energyAttached"), ("S", "supporterPlayed"),
          ("D", "stadiumPlayed"), ("R", "retreated"))
ACT_TAG = "[ACT]"


def _cards(seq):
    return " ".join(vocab.card_tok(c["id"]) for c in (seq or []) if c)


def _pk(p):
    """c<id>:<hp>/<maxHp>[|E<letters>][|S<ids>][|T<ids>][|P<ids>][|new]"""
    if not p:
        return "-"
    s = f"{vocab.card_tok(p['id'])}:{p.get('hp')}/{p.get('maxHp')}"
    e = p.get("energies") or []
    if e:
        s += f"|E{vocab.energy_letters(e)}"
    # special energy IDENTITY: basic energy is already covered by the type letters, but a
    # special energy changes legality/effects (e.g. Legacy Energy alters prize value).
    sp = [c for c in (p.get("energyCards") or []) if _is_special(c.get("id"))]
    if sp:
        s += "|S" + _cards(sp)
    # v1 emitted tools as a COUNT ("t1"), which cannot tell Hero's Cape from anything else
    if p.get("tools"):
        s += "|T" + _cards(p["tools"])
    # what this evolved from: needed for devolve / Rare Candy legality / what a KO returns
    if p.get("preEvolution"):
        s += "|P" + _cards(p["preEvolution"])
    if p.get("appearThisTurn"):
        s += "|new"          # played this turn -> evolution/attack legality
    return s


_SPECIAL = None


def _is_special(cid):
    global _SPECIAL
    if _SPECIAL is None:
        from agents._engine import _CARDS
        from cg.api import CardType
        _SPECIAL = {c.cardId for c in _CARDS.values()
                    if c.cardType == CardType.SPECIAL_ENERGY}
    return cid in _SPECIAL


def _side(pl, me):
    active = (pl.get("active") or [None])[0]
    bench = [b for b in (pl.get("bench") or []) if b]
    s = f"A[{_pk(active)}]"
    s += " B[" + ",".join(_pk(b) for b in bench) + "]" if bench else " B[]"
    s += (f" bm{pl.get('benchMax')} pz{len(pl.get('prize') or [])}"
          f" dk{pl.get('deckCount')} h{pl.get('handCount')}")
    if me:
        s += f" HAND[{_cards(pl.get('hand'))}]"      # ours only; the sim hides theirs
    s += f" DISC[{_cards(pl.get('discard'))}]"       # public for BOTH seats
    cond = [c for c, f in _CONDS if pl.get(f)]
    if cond:
        s += " " + ",".join(cond)
    return s


def _looking(cur):
    """Cards a card effect has just REVEALED to us (AreaType.LOOKING = 12).

    Card effects routinely open up hidden zones -- Pokegear 3.0 shows the top 7 of your
    deck, Drakloak the top 2, Snorunt reveals a random card from the opponent's HAND,
    Durant ex looks at the opponent's deck. That information lives ONLY in
    ``current.looking`` for the moment it is visible, and the option that picks from it
    is ``area=LOOKING, index=N, cardId=None`` -- a reference into this list, with
    ``sel.deck`` EMPTY. Measured over 30 games: 94 such decisions. Without this the model
    is choosing "1 of these 7" while blind to all 7 -- the same failure as the deck-search
    bug. ``None`` marks a card revealed as face-down (never observed here, but the schema
    allows it: "None if the card is facedown")."""
    lk = cur.get("looking")
    if not lk:
        return ""
    return " LOOK[" + " ".join(vocab.card_tok(c["id"]) if c else "?" for c in lk) + "]"


def render_state(obs):
    cur = obs["current"]
    yi = cur["yourIndex"]
    me, op = cur["players"][yi], cur["players"][1 - yi]
    flags = "".join(f for f, k in _FLAGS if cur.get(k))
    stad = cur.get("stadium") or []
    return (f"T{cur['turn']}.{cur['turnActionCount']}"
            f"{('/' + flags) if flags else ''} first{int(cur.get('firstPlayer') == yi)}\n"
            f"ME {_side(me, True)}\n"
            f"OP {_side(op, False)}\n"
            f"STAD[{vocab.card_tok(stad[0]['id']) if stad else '-'}]"
            f"{_looking(cur)}")


def render_options(obs):
    from lm.serialize import encode_option        # unchanged option encoding
    sel = obs["select"]
    items = " ".join(f"{i}={encode_option(o, obs)}" for i, o in enumerate(sel["option"]))
    return f"SEL {vocab.ctx_name(sel['context'])} n{sel['minCount']}-{sel['maxCount']} :: {items}"


def render_prompt(obs):
    """The whole v2 prompt for one decision. Stateless: no running context, no deltas --
    the board is fully described every time, so there is nothing to accumulate."""
    return f"{ACT_TAG}\n{render_state(obs)}\n{render_options(obs)}"
