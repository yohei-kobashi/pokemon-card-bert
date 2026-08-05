"""obs -> compact text (component A). Shared by build_sft (training) and the live
agent (inference), so the two distributions match.

Designed **append-only**: ``EpisodeSerializer.update`` emits the full state on the
first move of an episode, then only the *new* event logs + the current option menu
each subsequent move. Keeping a stable, growing prefix lets llama.cpp reuse the KV
cache across moves within a game (measured ~5 ms/move vs ~1-4.5 s for a full
re-prefill). The board state after move 1 is implied by the running event log.
"""
import random
import re
import zlib
from collections import Counter

from lm import costs, vocab
from lm.actions import encode_option

_TRAINER_KIND = {1: "ITEM", 2: "TOOL", 3: "SUP", 4: "STAD", 5: "NRG", 6: "SP-NRG"}


def _tok_multiset(ids):
    """Deduped card-token list with counts, e.g. 'c5x3,c1079' (for hand / discard)."""
    return ",".join(f"{vocab.card_tok(k)}{('x' + str(n)) if n > 1 else ''}"
                    for k, n in Counter(ids).items())


def render_card_rules(cid, prose=True):
    """FULL rules of one card on a single line (for the append-only glossary):
    Pokemon -> stage + evolves-from/to tokens + HP/type/weakness/resistance/retreat
    + every Ability and Attack (cost letters, damage, effect text). Trainer/Energy ->
    kind + effect text. Trash cards are NOT rendered here (token-only by design).

    ``prose=False`` drops every free-text effect ("structured" glossary), keeping the
    numbers a damage calculation needs. Measured on 200 real decisions: 54 tokens/line ->
    28, i.e. the prose is 48% of the glossary. See GLOSSARY_MODES."""
    c = vocab.card(cid)
    if not c:
        return vocab.card_tok(cid)
    if c.cardType == 0:                                   # Pokemon
        stage = ("MEGA" if c.megaEx else "S2" if c.stage2 else "S1" if c.stage1 else "B")
        tags = "".join(k for k, f in (("ex", c.ex), ("tera", c.tera)) if f)
        evo = ""
        fid = vocab.evolves_from_id(cid)
        if fid:
            evo += f" <-{vocab.card_tok(fid)}"
        tids = vocab.evolves_to_ids(cid)
        if tids:
            evo += " ->" + ",".join(vocab.card_tok(x) for x in tids)
        parts = [f"{vocab.card_tok(cid)} {c.name} [{stage}{('/' + tags) if tags else ''}{evo}]"
                 f" HP{c.hp} {vocab.etype_letter(c.energyType)} wk:{vocab.etype_letter(c.weakness)}"
                 f" rs:{vocab.etype_letter(c.resistance)} rt:{c.retreatCost}"]
        for s in (getattr(c, "skills", None) or []):
            nm = (s.name or '').strip()
            parts.append(f"AB {nm}: {(s.text or '').strip()}" if prose else f"AB {nm}")
        for aid in (c.attacks or []):
            a = vocab._ATTACKS.get(aid)
            if a:
                head = (f"ATK {vocab.attack_tok(aid)} {(a.name or '').strip()} "
                        f"[{vocab.energy_letters(a.energies)}]{a.damage}")
                parts.append(head + (f": {a.text.strip()}" if (prose and a.text) else ""))
        return " | ".join(parts)
    kind = _TRAINER_KIND.get(c.cardType, "?")             # Trainer / Energy
    txt = ((getattr(c, "skills", None) and (c.skills[0].text or "").strip()) or "")
    return f"{vocab.card_tok(cid)} {c.name} [{kind}]" + (f": {txt}" if (prose and txt) else "")


def visible_card_ids(obs):
    """Card ids whose rules belong in the glossary: my hand + both boards + stadium.
    Deliberately EXCLUDES discard/deck/prize (trash is token-only; hidden zones unknown)."""
    cur = obs.get("current") or {}
    yi = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    ids = []
    if yi < len(players):
        ids += [h["id"] for h in (players[yi].get("hand") or [])]
    for pl in players:
        for z in ("active", "bench"):
            for x in (pl.get(z) or []):
                if x:
                    ids.append(x["id"])
                    for t in (x.get("tools") or []):
                        if isinstance(t, dict) and t.get("id"):
                            ids.append(t["id"])
    for s in (cur.get("stadium") or []):
        ids.append(s["id"])
    return ids


def _ids(seq):
    out = []
    for t in (seq or []):
        if isinstance(t, dict) and t.get("id") is not None:
            out.append(t["id"])
        elif isinstance(t, int):
            out.append(t)
    return out


def _energy_counts(elist):
    """`GG C` -> `G2C` : one letter per DISTINCT type, with its count when above 1."""
    seen = []
    cnt = Counter(vocab.energy_letters(elist))
    for ch in vocab.energy_letters(elist):
        if ch not in seen:
            seen.append(ch)
    return "".join(ch + (str(cnt[ch]) if cnt[ch] > 1 else "") for ch in seen)


_ATTACKS = None


def _attack_table():
    global _ATTACKS
    if _ATTACKS is None:
        from cg.api import all_attack
        _ATTACKS = {a.attackId: a for a in all_attack()}
    return _ATTACKS


def _shortfall(cost, attached):
    """How many MORE energies this cost needs, honouring types.

    Typed symbols must be paid by their own type (RAINBOW counts as any); whatever is left over
    -- of any type -- pays the colourless symbols.
    """
    from cg.api import EnergyType
    col = int(EnergyType.COLORLESS)
    rainbow = int(EnergyType.RAINBOW)
    req = Counter(int(x) for x in (cost or []))
    have = Counter(int(x) for x in (attached or []))
    short = 0
    for t, k in req.items():
        if t == col:
            continue
        use = min(k, have.get(t, 0))
        have[t] -= use
        k -= use
        if k and have.get(rainbow, 0):          # rainbow pays any typed symbol
            w = min(k, have[rainbow])
            have[rainbow] -= w
            k -= w
        short += k
    left = sum(v for v in have.values() if v > 0)
    short += max(0, req.get(col, 0) - left)
    return short


def _need_energy(cid, attached):
    """Smallest number of extra energies that makes ANY damaging attack payable.

    Not "the cheapest attack by card count": with types, a longer cost whose symbols are
    already covered can be closer to payable than a shorter one demanding a type we lack.
    """
    c = vocab._CARDS.get(cid)
    if not c or not c.attacks:
        return None
    at = _attack_table()
    shorts = [_shortfall(at[a].energies, attached)
              for a in c.attacks if at.get(a) and at[a].damage]
    return min(shorts) if shorts else None


def _board_facts(p, obs=None, pi=None, dec=None):
    """` need:N rt:N` for one in-play Pokemon -- absent from v37 prompts entirely.

    With ``dec`` (the decoded hidden state, lm/hidden.py) BOTH numbers are the engine's own
    arithmetic rather than a reimplementation: `need` via GameUtil.h:InsufficientEnergyCount and
    `rt` via State.h:retreatCost. Measured against the engine over 63 decks, the fallbacks are
    wrong on 0.21% of `need` renders and 1.16% of `rt` -- always in the direction that tells the
    model a move it can make right now is unaffordable. Team Rocket's Energy ("2 in any
    combination of {P} and {D}") alone accounts for most of the `need` gap.

    `rt` is the LIVE retreat cost when the observation is available, not the card's printed one.
    The printed number is wrong wherever an effect changes it -- 19.4% of all offered retreats
    fleet-wide, 47% on ns_zoroark (N's Castle) and 46% on ethan_hooh (Latias ex) -- and it is
    wrong in the direction that tells the model a free retreat is unaffordable. See
    [[prompt-lies-about-retreat-cost]]; tools/audit_costs.py is the regression guard, and it
    checks the computed value against the menu (offered => cost <= attached energy).
    """
    cid = p.get("id")
    out = []
    need = None
    if dec is not None:                  # exact: GameUtil.h:InsufficientEnergyCount
        from lm import hidden as _hidden
        need = _hidden.need_energy(dec, obs, p.get("serial"))
    if need is None:
        need = _need_energy(cid, p.get("energies") or [])
    if need is not None:
        out.append("need:%d" % need)
    rt = None
    if dec is not None:                  # exact: the engine's own arithmetic (lm/hidden.py)
        from lm import hidden as _hidden
        rt = _hidden.retreat_cost(dec, p.get("serial"))
    if rt is None and obs is not None and pi is not None:
        rt = costs.effective_retreat_cost(obs, pi, p)
    if rt is None:                       # post-hoc rendering with no observation to read
        c = vocab._CARDS.get(cid)
        rt = c.retreatCost if c is not None else None
    if rt is not None:
        out.append("rt:%d" % rt)
    return (" " + " ".join(out)) if out else ""


def _pk(p, board_facts=False, obs=None, pi=None, extra=None, dec=None):
    if not p:
        return "-"
    # '*' = appeared THIS turn (history-derived: can't evolve yet; some effects care)
    s = f"{vocab.card_tok(p['id'])}{'*' if p.get('appearThisTurn') else ''}:{p.get('hp')}/{p.get('maxHp')}"
    e = p.get("energies") or []
    if e:
        # count form: |G3 not |GGG. Same tokens on average, but the QUANTITY is a number the
        # model reads instead of a run of characters it has to count against a cost.
        s += "|" + (_energy_counts(e) if board_facts else vocab.energy_letters(e))
    tools = _ids(p.get("tools"))
    if tools:                                   # tool CARDS (Cape/Belt change HP/damage)
        s += "|" + ",".join(vocab.card_tok(t) for t in tools)
    if board_facts:
        s += _board_facts(p, obs, pi, dec)
    if extra:
        s += extra.get(p.get("serial"), "")
    return s


def _side(pl, me, board_facts=False, obs=None, pi=None, extra=None, dec=None):
    active = (pl.get("active") or [None])[0]
    bench = [b for b in (pl.get("bench") or []) if b]
    s = f"A[{_pk(active, board_facts, obs, pi, extra, dec)}]"
    if bench:
        s += " B[" + ",".join(_pk(b, board_facts, obs, pi, extra, dec) for b in bench) + "]"
    s += f" pz{len(pl.get('prize') or [])} dk{pl.get('deckCount')} bm{pl.get('benchMax')}"
    if me:                                          # my hand CONTENTS (tokens)
        s += f" H[{_tok_multiset([h['id'] for h in (pl.get('hand') or [])])}]"
    else:                                           # opponent hand is hidden -> count
        s += f" h{pl.get('handCount')}"
    disc = [d["id"] for d in (pl.get("discard") or [])]   # discard is public, token-only
    if disc:
        s += f" D[{_tok_multiset(disc)}]"
    cond = [c for c, f in (("PSN", "poisoned"), ("BRN", "burned"), ("SLP", "asleep"),
                           ("PAR", "paralyzed"), ("CNF", "confused")) if pl.get(f)]
    if cond:
        s += " " + ",".join(cond)
    return s


def render_state(obs, deck_name=None, board_facts=False, identify="both", hidden_facts=False,
                 dec=None):
    """``hidden_facts`` adds the engine's live damage model -- ``dmg:+N`` on our attacker,
    ``tk:+N`` / ``tk:x0`` / ``fx:x`` on what it would hit. Those come from lm/hidden.py, which
    decodes the state blob the official library already puts in the observation; nothing is
    inferred from card text. Rendered only where non-default, so the cost is ~0 on the ~82% of
    decisions with nothing modified. See [[hidden-effect-state-audit]]."""
    cur = obs["current"]
    yi = cur["yourIndex"]
    me, op = cur["players"][yi], cur["players"][1 - yi]
    stad = cur.get("stadium") or []
    sd = f" stad:{vocab.card_tok(stad[0]['id'])}" if stad else ""
    flags = "".join(f for f, k in (("E", "energyAttached"), ("S", "supporterPlayed"),
                                   ("R", "retreated"), ("M", "stadiumPlayed")) if cur.get(k))
    extra = {}
    if hidden_facts:
        from lm import hidden as _hidden
        try:
            if dec is None:
                dec = _hidden.read(obs)
            extra = _hidden.board_extra(obs, dec) if dec else {}
        except Exception:
            extra, dec = {}, None
    return (f"T{cur['turn']}.{cur['turnActionCount']}"
            f"{('/' + flags) if flags else ''} ME {_side(me, True, board_facts, obs, yi, extra, dec)} "
            f"| OP {_side(op, False, board_facts, obs, 1 - yi, extra, dec)}{sd}{_identify(obs, yi, None if identify == 'op' else deck_name)}")


def _identify(obs, yi, deck_name=None):
    """Who is who? The LM cannot see the opponent's list, but their revealed cards pin it
    down (archetype 82.8% on turn 1, 97.6% by turn 3). Emits BOTH deck and archetype, and
    SEVERAL candidates when the posterior is genuinely split, e.g.
    ` ID ME d_crustle_stall a_control OP d_crustle:5 d_crustle_stall:3 a_beatdown:5`.

    Our OWN deck belongs HERE, next to the prediction, not in a separate segment: the board
    already uses the bare token ``ME`` for our side, so a second free-standing ``ME`` would
    be ambiguous. Passing ``deck_name`` matters most with glossary='none', where nothing
    else tells the model which deck it is piloting.
    Never raises: identification is an aid, not a dependency."""
    try:
        from cg.api import to_observation_class
        from lm import identify as _id
        st = to_observation_class(obs).current
        if not st or len(st.players or []) != 2:
            return ""
        return " ID " + _id.render(st, yi, my_deck_name=deck_name)
    except Exception:
        return ""


def _attack_damage_notes(obs, dec):
    """`{"attack:a123": " d:250"}` -- the base damage that attack would ACTUALLY use.

    31.2% of offered attack options have damage that moves (bench count, hand size, damage
    counters, prizes taken, attached energy, coins), and with glossary='none' the prompt carries
    no damage at all -- the `a123` token cannot encode a value that changes every turn. `d:` is
    exact, `d~` is an expectation over coins, and an attack whose damage cannot be resolved
    (a sub-select not yet made, an unimplemented condition) is left UNANNOTATED rather than
    guessed. See lm/damage.py; tools/verify_base_damage.py checks it against the base damage the
    engine actually used."""
    from lm import damage as _damage
    cur = obs.get("current") or {}
    yi = cur.get("yourIndex", 0)
    try:
        act = ((cur["players"][yi].get("active") or [None])[0])
    except (KeyError, IndexError, TypeError):
        return {}
    if not act or act.get("serial") is None:
        return {}
    out = {}
    for o in ((obs.get("select") or {}).get("option") or []):
        t = encode_option(o, obs)
        if not t.startswith("attack:") or t in out:
            continue
        try:
            aid = int(t.split(":")[1])
        except ValueError:
            continue
        val, kind = _damage.base_damage(obs, dec, act["serial"], aid, yi)
        if val is None:
            continue
        out[t] = " d:%d" % val if kind == "exact" else " d~%d" % val
    return out


def render_options(obs, menu_dedup=False, dec=None):
    """``menu_dedup`` shows one entry per ACT instead of one per menu position.

    The menu currently lists every option the engine offers, and 24.4% of those are the same act
    written twice: four copies of one energy in hand give four `attach:c3@ACTIVE0` entries, and
    two identical benched Basics give two targets nothing in the prompt distinguishes. Measured
    over 60,000 decisions: 7.08 entries -> 5.36 acts, menu 111 -> 79 characters, 4.8% off the
    whole prompt.

    It is also a CONSISTENCY fix, not only a length one. The cross-encoder is trained to rank the
    deduped candidate list (`lm/action_token.dedup_options`), so today it is shown 7.08 options
    while being asked about 5.36 acts.

    ENTRIES ARE RENUMBERED 0..n-1 over the surviving acts, so a pool built with this flag has
    `menu_index` values that no longer point into the raw option list. That is harmless for the
    cross-encoder, which never reads a menu index, but a DECODER pool must be rebuilt rather than
    re-rendered. Off by default for exactly that reason.
    """
    sel = obs["select"]
    # sub-select context: which card drives it + how much is left to place (damage/energy)
    cc = sel.get("contextCard")
    ccid = cc.get("id") if isinstance(cc, dict) else (cc if isinstance(cc, int) else None)
    extra = f" by:{vocab.card_tok(ccid)}" if ccid else ""
    if sel.get("remainDamageCounter"):
        extra += f" dmg:{sel['remainDamageCounter']}"
    if sel.get("remainEnergyCost"):
        extra += f" nrg:{sel['remainEnergyCost']}"
    mp = sel.get("_multipick")
    if mp:                                  # one step of a sequential multi-pick
        pk = ",".join(mp["picked"]) if mp["picked"] else "-"
        extra += f" MP[{len(mp['picked'])}/{mp['k']} picked:{pk} +upto{mp['need']}]"
    texts = [encode_option(o, obs) for o in sel["option"]]
    if menu_dedup:
        from lm.action_token import dedup_options
        texts = dedup_options(texts, obs)[0]
    ann = _attack_damage_notes(obs, dec) if dec is not None else {}
    items = " ".join(f"{i}={t}{ann.get(t, '')}" for i, t in enumerate(texts))
    if mp and mp.get("allow_stop"):         # may pick no more (min already satisfied)
        items += f" {len(texts)}={STOP}"
    return (f"SEL {vocab.ctx_name(sel['context'])}{extra} "
            f"n{sel['minCount']}-{sel['maxCount']} :: {items}")


STOP = "stop"     # pseudo-candidate for a sequential multi-pick: pick no more


def multipick_substate(obs, picked_pos):
    """ONE step of a sequential (one-at-a-time) multi-pick. ``picked_pos`` = the
    original option indices already chosen, in order. Returns
    ``(sub_obs, remaining_pos, allow_stop)`` where sub_obs is a single-pick obs whose
    menu is the REMAINING options, annotated with what's already picked and how many
    more may be taken. build_sft (training) and lm/agent (inference) BOTH build the
    prompt via ``serialize_stateless(sub_obs)`` and score the candidate list
    ``[enc(remaining)] (+ STOP if allow_stop)`` -- so train and inference match exactly.
    """
    sel = obs["select"]
    opts = sel.get("option") or []
    picked = list(picked_pos)
    remaining_pos = [i for i in range(len(opts)) if i not in picked]
    lo = sel.get("minCount", 1) or 0
    hi = sel.get("maxCount", 1) or 1
    allow_stop = (len(picked) >= lo) and (len(picked) < hi)
    ss = dict(sel)
    ss["option"] = [opts[i] for i in remaining_pos]
    ss["minCount"] = 1
    ss["maxCount"] = 1
    ss["_multipick"] = {"picked": [encode_option(opts[i], obs) for i in picked],
                        "need": hi - len(picked), "k": hi, "allow_stop": allow_stop}
    sub = dict(obs)
    sub["select"] = ss
    return sub, remaining_pos, allow_stop


def _norm_ids(deck_ids):
    out = []
    for x in (deck_ids or []):
        try:
            out.append(int(x["id"]) if isinstance(x, dict) else int(x))
        except (TypeError, ValueError):
            continue
    return out


def glossary_ids(obs, deck_ids=None):
    """Which card ids get RULES glossary lines, and IN WHAT ORDER.

    v2 (deck_ids given): our FULL deck first (a fixed set for the whole game -> a STABLE
    prompt prefix that llama.cpp's cross-decision KV cache reuses across every decision),
    then any other currently-visible card (opponent-revealed etc.). Only the small dynamic
    board+menu TAIL re-prefills each decision. v1 (deck_ids None): legacy visible-only,
    hand-first -> changes ~71% of decisions (measured), defeating the cache."""
    if deck_ids is None:
        return list(dict.fromkeys(visible_card_ids(obs)))
    base = list(dict.fromkeys(_norm_ids(deck_ids)))
    seen = set(base)
    extra = [c for c in dict.fromkeys(visible_card_ids(obs)) if c not in seen]
    return base + extra


# How much of the card glossary to emit. Measured on 200 real decisions (state = 838 tok,
# glossary = 671 = 80%, 12.5 lines x 54 tok, board+menu = 167), and projected onto the
# measured 4-vCPU cost of 433 s/game:
#
#   full        671 gloss -> 838 tok -> 433 s/game (72% of the 600 s bank)
#   structured  352       -> 571     -> 296 s (49%)   prose dropped, numbers kept
#   none          0       -> 167     ->  86 s (14%)   glossary dropped entirely
#
# WHY THIS EXISTS: the full-deck glossary was built to give llama.cpp a STABLE PREFIX so a
# DECODER could reuse its KV cache across decisions (see glossary_ids). A cross-encoder has
# no KV cache -- state and candidate attend to each other, so the pair is re-encoded per
# candidate -- which turns that same prefix into ~400 pointless re-encodes of identical text
# per game. Narrowing the glossary by CARD does not help (90% of its lines are already
# visible on board/hand/menu; excluding the discard saves 12 tokens): the cost is per-LINE.
GLOSSARY_MODES = ("full", "structured", "none")


DECK_MODES = ("static", "remaining", "roles")


def my_known_ids(obs):
    """Every card of OURS we can currently see: hand, discard, board (each Pokemon plus its
    evolution stack, attached energy CARDS and tools), and our stadium in play.

    ``energies`` is a list of TYPE codes, ``energyCards`` the actual cards -- subtracting the
    former would delete unrelated card ids. Validated by identity: the cards we cannot see
    are exactly the library and the face-down prizes, so ``60 - len(known)`` must equal
    ``deckCount + prizes``. It does, except during a sub-selection, where the card being
    resolved is in no zone yet and the count runs 1 high (~20% of decisions)."""
    cur = obs.get("current") or {}
    yi = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if yi >= len(players):
        return []
    pl = players[yi]
    out = [h["id"] for h in (pl.get("hand") or [])]
    out += [d["id"] for d in (pl.get("discard") or [])]
    for p in [(pl.get("active") or [None])[0]] + list(pl.get("bench") or []):
        if not p:
            continue
        out.append(p["id"])
        for k in ("preEvolution", "energyCards", "tools"):
            out += _ids(p.get(k))
    for s in (cur.get("stadium") or []):
        if s.get("playerIndex") == yi:
            out.append(s["id"])
    return out


def _deck_order_seed(obs, ids):
    """A seed that both build_rerank and lm/agent derive from the SAME obs.

    Shuffling DECK[] only helps if the order is UNPREDICTABLE, and it only stays a valid
    prompt if training and inference agree. Deriving the seed from the observation gives both
    at once -- identical obs produce an identical string on both sides, with no extra
    plumbing to keep in sync -- provided the seed carries real entropy. Turn alone would not:
    every T1.0 decision of a given deck would share one order, which is a per-turn fingerprint
    instead of a global one. The set of cards we can already see changes at nearly every
    decision, so it supplies the entropy. crc32, not hash(), because str/bytes hashing is
    salted per process."""
    cur = (obs or {}).get("current") or {}
    payload = repr((sorted(my_known_ids(obs)) if obs else [], cur.get("turn"),
                    cur.get("turnActionCount"), sorted(ids))).encode()
    return zlib.crc32(payload)


def render_my_deck(deck_ids, obs=None, mode="static", shuffle=False, roles=None):
    """Our deck as tokens, e.g. ``DECK[c344x4,c345x4,...]`` (~71 tokens for the average 21.9
    distinct cards). The 'full' glossary conveyed this IMPLICITLY by rendering every deck
    card; with glossary='none' that signal vanishes, and search/draw/discard decisions depend
    on knowing what is still gettable. Our deck IDENTITY (deck + archetype tokens) goes in
    the ID segment instead -- see _identify.

    ``mode='static'`` lists the original 60. That segment is REDUNDANT and measurably so:
    70% of its card tokens already appear elsewhere in the state (43% on turns 1-2, but
    **85% from turn 11**, and turn 11+ is half of all decisions). A model can therefore learn
    the same policy without ever reading it -- which is what happened (with segment dropout,
    deleting DECK[] costs nothing).

    ``mode='remaining'`` subtracts everything we can see (``my_known_ids``) and renders the
    counts still in deck+prizes. That is what a human tracks -- 'can I still get a Boss's
    Orders?' -- it is available from NO other part of the state at ANY turn, and it cannot be
    served by memorising 62 deck fingerprints because the multiset changes every turn. It
    also shrinks the segment as the game goes on, which is free speed."""
    if mode not in DECK_MODES:
        raise ValueError(f"deck mode must be one of {DECK_MODES}, got {mode!r}")
    ids = _norm_ids(deck_ids)
    if not ids:
        return ""
    if mode in ("remaining", "roles") and obs is not None:
        rem = Counter(ids) - Counter(my_known_ids(obs))
        ids = sorted(rem.elements())
        if not ids:
            return ("DECK" if mode == "roles" else "DECK[]")          # empty is INFORMATION (deck out / everything drawn)
    if mode == "roles":
        # Groups in a FIXED order so a card's ROLE is readable from its POSITION, ids sorted
        # WITHIN a group so the order is a function of the CONTENTS and cannot fingerprint the
        # deck. Shuffling is therefore neither needed nor applied. Empty groups are omitted --
        # each group carries its own marker, so position need not be reserved, and "no win
        # cards left in the library" is itself information.
        from lm.roles import group as _group, UNLABELLED as _UNL
        mark = {"win": "win", "engine": "eng", "line": "line", "fuel": "fuel",
                "tech": "tech", "filler": "fil", _UNL: "oth"}
        cnt = Counter(ids)
        parts = []
        for r, g in _group(sorted(cnt), roles or {}):
            body = ",".join(vocab.card_tok(k) + (("x%d" % cnt[k]) if cnt[k] > 1 else "")
                            for k in g)
            parts.append("%s[%s]" % (mark.get(r, r), body))
        return "DECK " + " ".join(parts)
    if not shuffle:
        return "DECK[" + _tok_multiset(ids) + "]"
    # A CANONICAL order makes the token sequence itself the deck's signature -- and the
    # canonical order here was the decklist FILE's order, not even sorted by id, so it was a
    # perfect one. Permuting per decision leaves no sequence to memorise: the only way to use
    # the segment is to read it as a SET of cards with counts.
    entries = list(Counter(ids).items())
    random.Random(_deck_order_seed(obs, ids)).shuffle(entries)
    return "DECK[" + ",".join(f"{vocab.card_tok(k)}{('x' + str(n)) if n > 1 else ''}"
                              for k, n in entries) + "]"


def serialize_stateless(obs, deck_ids=None, glossary="full", deck_name=None,
                        deck_mode="static", deck_shuffle=False, roles=None,
                        board_facts=False, identify="both", menu_dedup=False,
                        hidden_facts=False):
    """STATELESS prompt: current board only, no episode history. Self-contained =
    RULES glossary (see glossary_ids: our full deck first for cache-stability when
    deck_ids is given, else legacy visible-only) + our own deck identity + full board
    (with appearThisTurn '*', tool cards, benchMax) + the select menu (with sub-select /
    multi-pick context). Used for MAIN, sub-selections, and each step of a multi-pick.

    ``glossary`` is one of GLOSSARY_MODES -- the deploy-cost knob for the cross-encoder.
    ``deck_mode`` is one of DECK_MODES -- whether DECK[] lists the original 60 or what is
    still in the library. ``deck_shuffle`` permutes that list per decision so its ORDER
    cannot be the deck's signature. All three are part of the PROMPT FORMAT: build_rerank and
    lm/agent must pass the same values or the model is scored on inputs it never trained
    on. So is ``menu_dedup`` -- see render_options."""
    if glossary not in GLOSSARY_MODES:
        raise ValueError(f"glossary must be one of {GLOSSARY_MODES}, got {glossary!r}")
    head = ""
    if glossary != "none":
        rule_ids = glossary_ids(obs, deck_ids)
        rules = "\n".join(render_card_rules(c, prose=(glossary == "full")) for c in rule_ids)
        head = ("RULES " + rules + "\n") if rules else ""
    if roles is None and deck_mode == "roles":
        from lm.roles import for_deck
        roles = for_deck(deck_name)
    mine = render_my_deck(deck_ids, obs, deck_mode, deck_shuffle, roles)
    dec = None
    if hidden_facts:
        from lm import hidden as _hidden
        try:
            dec = _hidden.read(obs)
        except Exception:
            dec = None
    return (head + (mine + " " if mine else "")
            + render_state(obs, deck_name, board_facts=board_facts,
                           identify=identify, hidden_facts=hidden_facts, dec=dec)
            + " || " + render_options(obs, menu_dedup, dec))


# Both DECK renderings: the flat `DECK[c1x4,...]` of static/remaining mode, and the
# role-grouped `DECK win[...] eng[...] line[...]` of deck_mode="roles". The pattern used to
# cover only the first, so on v39 prompts `drop_deck` silently removed NOTHING and the ablation
# reported `-DECK[] 67.3%` against `full 67.3%` -- which reads as "the model ignores the deck"
# when in fact the deck was never taken away.
# ONE definition, exported: tools/ablate_rerank.py had its own copy of the old anchored
# pattern, so the fix here would not have reached swapDECK and that mask would have gone on
# silently substituting nothing.
DECK_SEG_RE = re.compile(r"^DECK(?:\[[^\]]*\]|(?:\s+\w+\[[^\]]*\])+)")
_RE_DECK = re.compile(DECK_SEG_RE.pattern + r"\s*")
_RE_MYID = re.compile(r"(?<= ID )ME (d_\S+)(?: (a_\S+))?\s*")


def mask_segments(state, drop_deck=False, drop_identity=False, swap_identity=False):
    """Remove (or corrupt) the two segments that tell us WHICH DECK WE ARE PILOTING.

    Used for two things that must stay in lockstep:
      * training augmentation -- ``DECK[c1152x4,...]`` determines our deck exactly, so
        ``ID ME d_alakazam`` is REDUNDANT and gets no gradient pressure however correct the
        label is. Dropping one at random forces the model to read the other.
      * ablation -- measuring how much the trained model actually USES each of them.

    ``swap_identity`` rewrites ``ME d_x`` to the opponent's top predicted deck: the exact
    corruption the v34 data shipped with (see build_rerank._deck_names), so its cost can be
    measured on a model rather than argued about."""
    s = state
    if drop_deck:
        s = _RE_DECK.sub("", s)
        if s == state:
            raise ValueError(
                "drop_deck removed nothing from %r... -- the pattern does not match this "
                "prompt format. A mask that silently no-ops does not read as broken, it reads "
                "as 'the model does not use this segment', which is the opposite conclusion."
                % state[:60])
    before = s
    if swap_identity:
        m = re.search(r"ID (?:ME \S+(?: a_\S+)? )?OP (d_\S+?):", s)
        if m:
            s = _RE_MYID.sub("ME " + m.group(1) + " ", s)
    elif drop_identity:
        s = _RE_MYID.sub("", s)
    if (swap_identity or drop_identity) and s == before and " ID ME " in before:
        raise ValueError("identity mask removed nothing from a prompt that HAS `ID ME`")
    return s


def render_logs(logs):
    out = []
    for lg in (logs or []):
        parts = []
        if lg.get("cardId"):
            parts.append(vocab.card_tok(lg["cardId"]))
        if lg.get("attackId"):
            parts.append(vocab.attack_tok(lg["attackId"]))
        if lg.get("value") is not None:
            parts.append(f"v{lg['value']}")
        tag = f"L{lg.get('type')}"
        out.append(f"{tag}({','.join(parts)})" if parts else tag)
    return " ".join(out)


def serialize_full(obs):
    """One-shot full serialization of a single obs (training samples / debugging)."""
    return render_state(obs) + " || " + render_options(obs)


# The SFT mix is multi-task and the TASK IS SELECTED BY A PROMPT PREFIX:
#   "[ACT]\n..."     -> emit the move            (10,045 rows)
#   "[COMPARE]\n..." -> emit A/B rollouts        ( 5,553 rows)
#   no prefix        -> emit the FUTURE LOG      ( 4,402 rows, the "reason" task)
# and an [ACT] prompt is the WHOLE running context of the episode so far, tagged once at
# the head (measured over 14,935 act rows: mean 2556 chars, p50 327, p90 8240, max 36181;
# the short ones are simply early turns, e.g. "[ACT]\nL0 | SEL MAIN n1-1 :: 0=retreat 1=end").
ACT_TAG = "[ACT]\n"


class EpisodeSerializer:
    """Append-only running context for one episode (enables KV-cache reuse).

    ``update`` returns the FULL context each call (tagged, cumulative) and ``delta``
    exposes just the new suffix for a KV-cache-backed model.

    Two bugs this fixes, both found by watching the trained model play:
      1. No [ACT] tag. The model saw an untagged prompt, correctly inferred the "reason"
         task, and dutifully wrote future logs -- decodes like
         "') L0 L11(c5) L3 L2 L5 L12(c742)'" instead of a move.
      2. Only the delta was returned, and lm/agent.py passed it straight to the model as
         the entire prompt. Training prompts are cumulative, so the model was being asked
         to move from a context fragment with no history.
    Together: 37% of decisions were unusable and silently fell back to the heuristic.
    """

    def __init__(self):
        self.started = False
        self.last_turn = None
        self.ctx = ""
        self.last_delta = ""
        self.seen = set()         # cards whose rules are already in the glossary

    def reset(self):
        self.started = False
        self.last_turn = None
        self.ctx = ""
        self.last_delta = ""
        self.seen = set()

    def delta(self):
        """The suffix appended by the most recent update (for KV-cache reuse)."""
        return self.last_delta

    def _glossary(self, obs):
        """Rule blocks for cards newly visible this move (append-only). Each card's
        FULL rules are emitted ONCE, the first time it appears in hand/board/stadium;
        thereafter it is just a token. Trash cards get no rule block (token-only)."""
        new = []
        for cid in visible_card_ids(obs):
            if cid not in self.seen:
                self.seen.add(cid)
                new.append(render_card_rules(cid))
        return ("RULES " + "\n".join(new) + "\n") if new else ""

    def update(self, obs):
        cur = obs.get("current") or {}
        turn = cur.get("turn")
        if self.last_turn is not None and turn is not None and turn < self.last_turn:
            self.reset()          # a new episode started in a reused process
        self.last_turn = turn
        glos = self._glossary(obs)                    # new card rules (append-only)
        body = serialize_full(obs)                    # full board + my hand + discard + menu
        if not self.started:
            self.started = True
            self.last_delta = ACT_TAG + glos + body
        else:
            self.last_delta = "\n" + glos + body
        self.ctx += self.last_delta
        return self.ctx
