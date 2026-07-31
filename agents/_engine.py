"""Shared heuristic engine for deck agents.

A single, deck-agnostic policy that plays a competent game of the cabt Pokémon
TCG. Each deck agent is a tiny wrapper that calls ``act(obs_dict, DECK, style)``.

Design notes
------------
* STATELESS: every decision is computed from ``obs`` alone (no module globals),
  so the same module can safely pilot both players in a self-play match.
* Deck knowledge is DERIVED from the card/attack databases (``all_card_data`` /
  ``all_attack``): attack damage & energy cost, weakness/resistance, evolution
  stage, ex/megaEx, retreat cost. A deck only supplies its 60 IDs + a ``style``.
* The engine picks the single best *next* action on each MAIN selection (the
  simulator re-invokes the agent after every action within a turn) and handles
  every sub-selection context (targets, discards, coin flips, setup, ...).

Styles: 'aggro' (default), 'evolve', 'spread', 'control', 'combo'. They only
nudge a few branches; the shared core does the heavy lifting.
"""

import os

from cg.api import (
    AreaType, CardType, EnergyType, OptionType, SelectContext, SelectType,
    all_card_data, all_attack, to_observation_class,
)


def load_deck(name):
    """Load a deck's 60 card IDs. Works both locally (decks/<name>.csv) and in a
    Kaggle submission (deck.csv next to main.py)."""
    candidates = [
        os.path.join("decks", name + ".csv"),
        name + ".csv",
        "deck.csv",
        "/kaggle_simulations/agent/deck.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return [int(line) for line in f if line.strip()]
    raise FileNotFoundError("deck not found: " + ", ".join(candidates))

_CARDS = {c.cardId: c for c in all_card_data()}
_ATTACKS = {a.attackId: a for a in all_attack()}

# Per-call hint context. act() sets this at the start of each (synchronous)
# call, so it is safe even when the same module pilots both players.
#   main_attackers: set/iterable of card IDs to concentrate energy on, promote
#                   to active, and treat as attackers even if their listed base
#                   damage is 0 (scaling attacks).
#   accel:          extra energy-accel item name substrings to play proactively.
#   play:           item name substrings to always play (deck engine pieces).
_ctx = {"hints": {}, "mains": set(), "main_line": set(), "copies": {}, "card_roles": {}}


def _main_boost(pk):
    if pk is None:
        return 0
    if pk.id in _ctx["mains"]:
        return 500
    if pk.id in _ctx["main_line"]:  # a pre-evolution leading to a main attacker
        return 200
    return 0


def _compute_main_line(deck):
    """mains + all their in-deck pre-evolutions, so we set up / promote the whole
    line toward the main attacker (e.g. Ralts/Kirlia for Mega Gardevoir ex).

    Also caches the deck's COPY COUNTS: how many of each card the builder chose to run
    is a per-deck statement of importance that _deck_importance reads."""
    import collections as _c
    _ctx["copies"] = _c.Counter(deck)
    _ctx["card_roles"] = _ctx["hints"].get("card_roles") or {}
    mains = set(_ctx["hints"].get("main_attackers", ()))
    _ctx["mains"] = mains
    line = set(mains)
    name_to_id = {}
    for cid in set(deck):
        c = _CARDS.get(cid)
        if c and c.name not in name_to_id:
            name_to_id[c.name] = cid
    changed = True
    while changed:
        changed = False
        for cid in list(line):
            c = _CARDS.get(cid)
            pid = name_to_id.get(c.evolvesFrom) if c and c.evolvesFrom else None
            if pid is not None and pid not in line:
                line.add(pid)
                changed = True
    _ctx["main_line"] = line


# ---- card-role helpers (derived from the card DB) --------------------------
def _name(cid):
    c = _CARDS.get(cid)
    return c.name if c else ""


def _is_energy(cid):
    c = _CARDS.get(cid)
    return c is not None and c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)


def _is_supporter(cid):
    c = _CARDS.get(cid)
    return c is not None and c.cardType == CardType.SUPPORTER


# Curated name sets for the few effects worth recognising explicitly.
# Supporters the engine will spend the once-per-turn slot on. It is a NAME list, so a
# card missing from it is INVISIBLE to named() -- not deprioritised, INVISIBLE.
# "Dawn" added 2026-07-17 (A/B: _DAWN_SUPPORTER): "search your deck for a Basic, a
# Stage 1 AND a Stage 2 Pokemon ... put them into your hand". For alakazam that is the
# entire Abra->Kadabra->Alakazam line in ONE card AND **+3 cards of hand = +60 damage**,
# because Powerful Hand is 20 x HAND SIZE. Measured: our alakazam played Dawn 0.25/game
# (offered on 451 menus!) while the LIVE alakazam opponents that beat v29 **2-11** played
# it **1.77/game** and attacked with a **13.92**-card hand (278 dmg) vs our 10.33 (207).
# A/B, default OFF. Dawn IS invisible to named() -- a real bug: 14 decks run it and the
# engine can never pick it (a NAME missing from _DRAW_SUPPORTERS is invisible, not
# deprioritised). But turning it on is NOT validated: measured at 150 games/matchup it
# fires 3x (Dawn 0.22 -> 0.61/game) and moves NOTHING -- hand at Powerful Hand 10.49 ->
# 10.19, winrate 38 -> 38. Same shape as the refuted SlowkingL2 sequencing fix. And it
# would change 14 decks, only 2 of which were measured. Do not flip it on without a
# fleet gate (tools/fingerprint.py).
_DAWN_SUPPORTER = False
_DRAW_SUPPORTERS = ("Determination", "Carmine", "Professor's Research", "Iono",
                    "Judge", "Hilda", "Cyrano", "Colress", "Jacinthe", "Bianca",
                    "Petrel", "Ariana", "Adventure", "Explorer's Guidance") + (
                    ("Dawn",) if _DAWN_SUPPORTER else ())
_GUST = ("Boss's Orders", "Boss’s Orders", "Prime Catcher", "Counter Catcher", "Petrel")
_SEARCH_ITEMS = ("Ultra Ball", "Nest Ball", "Dusk Ball", "Buddy-Buddy Poffin",
                 "Poffin", "Poke Pad", "Poké Pad", "Pokegear", "Pokégear",
                 "Earthen Vessel", "Energy Search", "Tera Orb", "Bug Catching Set",
                 "Master Ball")   # Master Ball: unrestricted Pokemon search, no discard cost
_RARE_CANDY = ("Rare Candy",)

# TO_ACTIVE promotion resolves the actual Pokemon (energy-aware) instead of scoring
# a cardId that is always None on field-ref options. OPT-IN PER DECK via the
# 'promote_ready' hint: it is NOT universally right. A 300HP Kangaskhan tank wants to
# be Active the moment it is fuelled (crustle: +8pt), but a 140HP Alakazam is a fragile
# engine that would rather feed a cheap body and keep the loaded copy safe on the bench
# (alakazam: -6pt when forced). Master switch below is only for A/B.
_PROMOTE_READY = True

# Deck-search options carry cardId=None and only resolve via sel.deck[index]; nothing
# read sel.deck, so every search on this engine picked blind. Set True to A/B the old
# behaviour. _NEED_AWARE=False reverts _card_usefulness to the bare hierarchy.
_SEARCH_BLIND = False
_NEED_AWARE = True
# Copy-count as a proxy for per-deck importance: MEASURED DEAD END. A built deck is
# almost entirely 4-ofs (alakazam 44/60 = 73%, crustle 52/60 = 87%) because anything
# marginal was already cut to reach 60 -- so the count carries almost no signal, all the
# 4-ofs tie at +300, and the ranking inside them falls back to the same wrong global
# hierarchy (Rare Candy still pitched as "Item = 25"). Discards of 4-ofs stayed at 52%.
# Superseded by explicit per-deck card_roles.
_DECK_VALUE = False
# A/B switch for explicit per-deck card_roles.
_USE_ROLES = True
# HAND resolution feeds the DISCARD scorer, which is a different question from search.
_RESOLVE_HAND = True
# Promotion options are field/hand REFERENCES (cardId=None), so _setup_score(o.cardId)
# tied EVERY option: measured 100% of promotion menus fully tied (alakazam 28/28,
# crustle 22/22, crustle_stall 23/23) -- promotion was by slot index. Set False to A/B
# the old blind behaviour. (_promote_score's energy-awareness stays behind the per-deck
# ``promote_ready`` hint; this flag is only about SEEING the option at all.)
_RESOLVE_PROMOTE = True
_SWITCH_CARDS = ("Switch", "Escape Rope")
# Energy-acceleration / energy-recovery items worth playing proactively.
_ENERGY_ACCEL = ("PP Up", "Fighting Gong", "Wondrous Patch", "Energy Switch",
                 "Superior Energy Retrieval", "Energy Retrieval", "Earthen Vessel",
                 "Brilliant Blender", "Sparkling Crystal", "Tera Orb",
                 "Scramble Switch", "Energy Recycler", "Precious Trolley")


def _has(name, keys):
    return any(k in name for k in keys)


def _best_attack(pk):
    """Return (attackId, damage) of the highest-damage attack this Pokémon has."""
    c = _CARDS.get(pk.id)
    best = (None, -1)
    if not c:
        return best
    for aid in c.attacks:
        a = _ATTACKS.get(aid)
        if a and a.damage is not None and a.damage > best[1]:
            best = (aid, a.damage)
    return best


def _attack_cost(aid):
    a = _ATTACKS.get(aid)
    return len(a.energies) if a and a.energies else 0


# ---- opponent-target scoring (adapted from the sample agent) ---------------
def _prize_value(pk):
    c = _CARDS.get(pk.id)
    if not c:
        return 1
    v = 3 if c.megaEx else 2 if c.ex else 1
    for e in pk.energyCards:
        if _name(e.id) == "Legacy Energy":
            v -= 1
    return max(0, v)


def _target_score(pk):
    """How valuable is it to damage/KO this opponent Pokémon?"""
    c = _CARDS.get(pk.id)
    score = _prize_value(pk) * 1000
    score += len(pk.energies) * 120
    score += len(pk.tools) * 80
    if c:
        if c.stage2:
            score += 220
        elif c.stage1:
            score += 110
    score += pk.hp
    return score


def _cheapest_cost(pk):
    """Minimum energy cost among this Pokémon's damaging attacks (0 if none)."""
    c = _CARDS.get(pk.id)
    if not c:
        return 0
    costs = [_attack_cost(a) for a in c.attacks
             if _ATTACKS.get(a) and _ATTACKS[a].damage]
    return min(costs) if costs else 0


def _my_attacker_score(pk):
    """How good an attacker is this of mine (for attach / promote / effect target).

    Driven by BASE attack damage and ex status -- deliberately NOT by how much
    energy is already attached, so energy doesn't snowball onto a non-attacking
    middle-evolution (e.g. Drakloak) instead of the real attacker. A deck can
    force its true attacker(s) via the 'main_attackers' hint (also overrides the
    "no damaging attack" exclusion, for scaling attacks listed as 0 damage).
    """
    if pk is None:
        return -1
    boost = _main_boost(pk)
    _, dmg = _best_attack(pk)
    if dmg <= 0 and boost <= 0:
        return -1  # not an attacker (e.g. a Pokémon with no damaging attack)
    c = _CARDS.get(pk.id)
    s = max(dmg, 0) + boost
    if c and (c.ex or c.megaEx):
        s += 100
    s += min(len(pk.energies), 1) * 10  # tiny tie-break toward a started attacker
    return s


# ---- generic selection utilities -------------------------------------------
def _mk(indices, sel):
    """Clamp an index list to [minCount, maxCount] and guarantee legality."""
    n = len(sel.option)
    out = []
    for i in indices:
        if 0 <= i < n and i not in out:
            out.append(i)
        if len(out) >= sel.maxCount:
            break
    # pad up to minCount with the first unused legal indices
    i = 0
    while len(out) < sel.minCount and i < n:
        if i not in out:
            out.append(i)
        i += 1
    return out


def _opp_field(state, my_index):
    op = state.players[1 - my_index]
    cards = []
    for pk in op.active:
        if pk is not None:
            cards.append(pk)
    for pk in op.bench:
        if pk is not None:
            cards.append(pk)
    return cards


# ---- MAIN decision ---------------------------------------------------------
def _choose_main(obs, sel, deck, style, allow_retreat=True):
    state = obs.current
    mi = state.yourIndex
    me = state.players[mi]
    opts = sel.option

    # bucket options by type
    evolves, abilities, plays, attaches, attacks = [], [], [], [], []
    retreat_idx = end_idx = None
    for i, o in enumerate(opts):
        t = o.type
        if t == OptionType.EVOLVE:
            evolves.append(i)
        elif t == OptionType.ABILITY:
            abilities.append(i)
        elif t == OptionType.PLAY:
            plays.append(i)
        elif t == OptionType.ATTACH:
            attaches.append(i)
        elif t == OptionType.ATTACK:
            attacks.append(i)
        elif t == OptionType.RETREAT:
            retreat_idx = i
        elif t == OptionType.END:
            end_idx = i

    def _opt_atk_dmg(i):
        a = _ATTACKS.get(opts[i].attackId)
        base = a.damage if a and a.damage else 0
        # A hinted main attacker's scaling attack lists 0 damage (e.g. Mega
        # Symphonia = 50x energy, N's Zoroark scaling): value it by energy so the
        # engine actually chooses it once powered.
        if base == 0 and me.active and me.active[0] and me.active[0].id in _ctx["mains"]:
            board_e = sum(len(p.energies) for p in ([me.active[0]] + list(me.bench)) if p)
            return max(30 * len(me.active[0].energies), 20 * board_e)
        return base

    # Safety net: if this turn has already taken many actions, stop setting up
    # (avoids any infinite MAIN->ability->MAIN loop) and just attack or end.
    if (state.turnActionCount or 0) >= 40:
        if attacks:
            return [max(attacks, key=_opt_atk_dmg)]
        if end_idx is not None:
            return [end_idx]
        if retreat_idx is not None and not state.retreated:
            return [retreat_idx]
        return _mk([0], sel)

    def hand_card(o):
        h = me.hand
        return h[o.index] if (h is not None and o.index is not None and o.index < len(h)) else None

    # 1) Evolve toward attackers (always good; core of 'evolve' decks)
    if evolves:
        # prefer evolving into the higher-damage form
        best, bi = -1, None
        for i in evolves:
            card = hand_card(opts[i])
            if card is None:
                continue
            _, dmg = _best_attack(card)
            if dmg > best:
                best, bi = dmg, i
        if bi is not None:
            return [bi]

    # 2) Beneficial abilities (draw/accel/setup). Use them; harmful ones are rare.
    if abilities and style != "control_no_ability":
        return [abilities[0]]

    # Precompute best attack available right now (offered ATTACK = already legal)
    best_atk = max(attacks, key=_opt_atk_dmg) if attacks else None
    atk_dmg = _opt_atk_dmg(best_atk) if best_atk is not None else 0

    # 3) Supporter: gust (Boss) if it can enable a KO, else draw when hand is low
    if not state.supporterPlayed:
        gust = [i for i in plays if hand_card(opts[i]) and _has(_name(hand_card(opts[i]).id), _GUST)]
        draw = [i for i in plays if hand_card(opts[i]) and _has(_name(hand_card(opts[i]).id), _DRAW_SUPPORTERS)]
        # draw early / when hand thin
        if draw and me.handCount <= 5:
            return [draw[0]]
        # gust to drag a juicy target if we have a real attack ready
        if gust and atk_dmg >= 60 and _opp_field(state, mi):
            return [gust[0]]
        if draw and me.handCount <= 6 and style != "control":
            return [draw[0]]

    # 4) Item plays: rare candy, search, switch when stuck
    def play_named(keys):
        for i in plays:
            c = hand_card(opts[i])
            if c and _has(_name(c.id), keys):
                return i
        return None
    for keys in (_RARE_CANDY, _SEARCH_ITEMS):
        i = play_named(keys)
        if i is not None:
            return [i]

    # 4a2) Deck engine pieces the deck flagged to always play (hint 'play').
    hint_play = tuple(_ctx["hints"].get("play", ()))
    if hint_play:
        i = play_named(hint_play)
        if i is not None:
            return [i]

    # 4b) Play energy-acceleration items to power up an attacker (N's PP Up,
    #     Fighting Gong, Wondrous Patch, Energy Switch/Retrieval...) plus any
    #     deck-specific accel (hint 'accel') -- only when no attack is ready yet.
    if best_atk is None:
        i = play_named(_ENERGY_ACCEL + tuple(_ctx["hints"].get("accel", ())))
        if i is not None:
            return [i]
    # play a stadium only if none is currently in play (avoid churn)
    if not state.stadium:
        for i in plays:
            c = hand_card(opts[i])
            cd = _CARDS.get(c.id) if c else None
            if cd and cd.cardType == CardType.STADIUM:
                return [i]

    # switch only if active can't attack at all and we have no attack ready
    if best_atk is None:
        i = play_named(_SWITCH_CARDS)
        if i is not None:
            return [i]

    # 5) Attach energy to the best attacker that still needs energy (concentrate;
    #    prioritise an attacker not yet at its attack cost, prefer the real
    #    attacker over a middle-evolution).
    if attaches and not state.energyAttached:
        def attach_target(o):
            area = o.inPlayArea
            idx = o.inPlayIndex if o.inPlayIndex is not None else 0
            if area == AreaType.ACTIVE and me.active:
                return me.active[0]
            if area == AreaType.BENCH and idx < len(me.bench):
                return me.bench[idx]
            return None

        def attach_score(o):
            pk = attach_target(o)
            s = _my_attacker_score(pk)  # base damage + ex; NOT energy count
            if s < 0:
                return -10  # non-attacker: last resort
            # small nudge toward an attacker still short of its cheapest attack,
            # but never penalise loading a big attacker further (Zacian/Raging
            # Bolt etc. want to overload their single attacker).
            if len(pk.energies) < _cheapest_cost(pk):
                s += 150
            if _ctx["hints"].get("promote_ready"):
                # CONCENTRATE (opt-in). With N copies of one attacker (crustle runs 4
                # Mega Kangaskhan ex) every copy scores identically -- _my_attacker_score
                # is deliberately energy-blind and the +150 "short of cost" nudge applies
                # to all of them -- so max() fell through to option order and energy
                # scattered across the bench. The live mirror lost 0-6 with TWO E3
                # Kangaskhan idle on the bench and an empty Active. Prefer the Active
                # (it is the body that can actually attack), then the copy closest to
                # its threshold, so one attacker reaches cost instead of four reaching 1.
                if o.inPlayArea == AreaType.ACTIVE:
                    s += 60
                s += 8 * len(pk.energies)
            return s
        return [max(attaches, key=lambda i: attach_score(opts[i]))]

    # 6) Attack if it does something meaningful (style-gated)
    if best_atk is not None:
        min_dmg = 10 if style != "control" else 30
        if atk_dmg >= min_dmg:
            return [best_atk]
        # control: also attack if it would KO the active
        op = _opp_field(state, mi)
        if op and op[0].hp <= atk_dmg:
            return [best_atk]

    # 7) fall back: retreat if stuck & no attack, else end.
    #    NOTE: an "anti-waste" guard here (skip retreat when it would discard energy
    #    just attached to the active) was tried and REVERTED -- clean A/B: alakazam
    #    -4.7 vs field, others neutral. The attach-then-retreat that LOOKS wasteful
    #    is mostly a beneficial reposition; blocking it to save energy loses more
    #    than it saves. So the plain retreat stays.
    if allow_retreat and best_atk is None and retreat_idx is not None and not state.retreated:
        return [retreat_idx]
    if end_idx is not None:
        return [end_idx]
    if best_atk is not None:
        return [best_atk]
    return _mk([0], sel)


# ---- sub-selection contexts ------------------------------------------------
def _choose_sub(obs, sel, deck, style):
    ctx = sel.context
    opts = sel.option
    state = obs.current
    mi = state.yourIndex if state else 0

    # deck/hand/setup card selections that carry a Card per option
    def card_of(o):
        if o.cardId is not None:
            return _CARDS.get(o.cardId)
        return None

    # YES/NO
    if sel.type == SelectType.YES_NO or all(o.type in (OptionType.YES, OptionType.NO) for o in opts):
        yes = next((i for i, o in enumerate(opts) if o.type == OptionType.YES), None)
        no = next((i for i, o in enumerate(opts) if o.type == OptionType.NO), None)
        if ctx == SelectContext.MULLIGAN:
            return _mk([no if no is not None else 0], sel)  # keep hand, avoid churn
        if ctx == SelectContext.IS_FIRST:
            want = yes if style != "control" else (no if no is not None else yes)
            return _mk([want if want is not None else 0], sel)
        return _mk([yes if yes is not None else 0], sel)  # default: activate/heads

    # Setup: choose active/bench basics -> prefer the main attacker's own line
    # (so an evolving scaling attacker like Mega Gardevoir ex ends up active),
    # otherwise the best listed attacker.
    def _setup_score(cid):
        s = _best_attack_by_id(cid)
        if cid in _ctx["mains"]:
            s += 1000
        elif cid in _ctx["main_line"]:
            s += 400
        return s

    def _opt_pk_id(o):
        """Which Pokemon does a promotion / bench option refer to?

        These options are REFERENCES: they carry cardId=None and point either at a FIELD
        slot (playerIndex/area/index) or at a HAND index (the setup promotion, where
        nothing is on the field yet). Passing the raw ``o.cardId`` to _setup_score
        therefore scored EVERY option 0 -- measured: 100% of promotion menus fully tied
        (alakazam 28/28, crustle 22/22, crustle_stall 23/23), i.e. promotion was by slot
        index, blind to what was standing there. _promote_score resolved the field half
        of this, but only for decks that opt into the ``promote_ready`` hint (crustle
        alone), and never the hand half. Resolving the reference is a BUG FIX and applies
        to every deck; the energy-awareness in _promote_score stays opt-in."""
        if not _RESOLVE_PROMOTE:
            return o.cardId                  # A/B: the old blind key
        pk = _pokemon_at(state, o)
        if pk is not None:
            return pk.id
        if o.cardId is not None:
            return o.cardId
        return _opt_card_id(o, sel, state, mi)

    def _promote_score(o):
        """Score a TO_ACTIVE / setup option by the ACTUAL Pokemon behind it.

        TO_ACTIVE options are FIELD REFS: they carry ``cardId=None`` (only
        playerIndex/area/index), so ``_setup_score(o.cardId)`` scored every option
        identically and ``max()`` fell through to bench slot 0 -- i.e. promotion was
        by slot index, blind to what was standing there. Live crustle mirror
        (Hexylab, 0-6, ZERO attacks in 22 turns) caught it: with a fully-loaded
        Mega Kangaskhan ex (E3, ready to attack) on the bench it promoted an
        energy-less Crustle, and next time an 80HP Shaymin. Resolve the Pokemon and
        prefer one that can attack NOW; the opponent's whole edge in that mirror was
        simply always having a fuelled attacker in the Active spot.
        """
        pk = _pokemon_at(state, o) if o.cardId is None else None
        if pk is None:
            # setup promotion: nothing on the field yet, the option is a HAND index --
            # _setup_score(o.cardId) was 0 for every one of them. No energy is attached
            # to a card in hand, so the energy terms below simply do not apply here.
            return _setup_score(_opt_pk_id(o))
        # ADD energy-awareness ON TOP of the existing card ranking rather than
        # replacing it: _my_attacker_score returns -1 for attacks listed as 0 damage
        # (Alakazam's hand-scaling Powerful Hand), so scoring by it alone stopped
        # Alakazam promoting its own Alakazam (-3.3pt field). _setup_score keeps the
        # mains/main_line intent; the energy terms only break ties between bodies.
        s = _setup_score(pk.id)
        cost = _cheapest_cost(pk)
        if cost and len(pk.energies) >= cost:
            s += 2000                        # can attack THIS turn -> dominates
        s += 20 * len(pk.energies)           # else: closest to its threshold
        if _my_attacker_score(pk) < 0 and pk.id not in _ctx["mains"]:
            s -= 500                         # genuine non-attacker (Shaymin): last resort
        return s
    if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE):
        # Repelling-Veil tech: vs Alakazam (743, "Powerful Hand" PLACES damage counters
        # so it bypasses HP walls / damage reduction), promote Team Rocket's Articuno
        # (414) if it is a choice -- its Repelling Veil makes it immune to those attack
        # EFFECTS, an auto-loss for Alakazam. Narrow + only when we actually run Articuno.
        opp = state.players[1 - mi] if state else None
        if opp is not None:
            opp_ids = {p.id for p in
                       ((opp.active or []) + list(opp.bench or [])) if p}
            if 743 in opp_ids:
                veil = next((i for i, o in enumerate(opts) if _opt_pk_id(o) == 414), None)
                if veil is not None:
                    return _mk([veil], sel)
            # Crustle wall: Crustle (345) "Mysterious Rock Inn" prevents ALL damage
            # to ITSELF from the opponent's {ex} Pokémon. If the opponent's Active is
            # an ex attacker (e.g. Mega Starmie ex 210), promote Crustle -- it walls
            # the ex for free while chipping Superb Scissors 120. Only when we run it.
            oact = opp.active[0] if opp.active else None
            oc = _CARDS.get(oact.id) if oact is not None else None
            if oc is not None and (oc.ex or oc.megaEx):
                crus = next((i for i, o in enumerate(opts) if _opt_pk_id(o) == 345), None)
                # Only wall with Crustle when this deck has no real attacker to
                # promote instead -- a Kangaskhan box's 200-hitter is a far better
                # active than the 120 wall, so it must NOT be overridden.
                best_atk_opt = max((_best_attack_by_id(_opt_pk_id(o)) for o in opts), default=0)
                if crus is not None and best_atk_opt < 180:
                    return _mk([crus], sel)
        if _PROMOTE_READY and _ctx["hints"].get("promote_ready"):
            best = max(range(len(opts)), key=lambda i: _promote_score(opts[i]))
        else:
            best = max(range(len(opts)), key=lambda i: _setup_score(_opt_pk_id(opts[i])))
        return _mk([best], sel)
    if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        order = sorted(range(len(opts)), key=lambda i: -_setup_score(_opt_pk_id(opts[i])))
        return _mk(order, sel)

    # Gust (SWITCH the opponent's benched Pokémon to their Active, e.g. Boss's
    # Orders): pull up a target my Active can KO THIS turn -> convert to a prize /
    # remove a developing threat. Gusting an un-KO-able big attacker just PROMOTES
    # the opponent's win condition for them, so prefer KO-able targets and only
    # fall to raw value when none is KO-able. Handles hand-scaling attacks
    # (Alakazam's Powerful Hand = 20 x handSize) via _my_active_dmg.
    if ctx == SelectContext.SWITCH and any(
            o.playerIndex is not None and o.playerIndex != mi for o in opts):
        dmg = _my_active_dmg(state, mi)

        def _gust(i):
            pk = _pokemon_at(state, opts[i])
            if pk is None:
                return (-1, 0)
            koable = 1 if (dmg > 0 and pk.hp <= dmg) else 0
            return (koable, _target_score(pk))
        order = sorted(range(len(opts)), key=_gust, reverse=True)
        return _mk(order, sel)

    # Opponent-targeting contexts -> highest-value opposing Pokémon
    OPP_TARGET = {SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
                  SelectContext.DAMAGE_COUNTER_ANY, SelectContext.SWITCH,
                  SelectContext.EFFECT_TARGET, SelectContext.TO_ACTIVE}
    if ctx in OPP_TARGET:
        def sc(i):
            o = opts[i]
            pk = _pokemon_at(state, o)
            base = _target_score(pk) if pk else 0
            # if the option is on our own side, prefer our best attacker instead
            if o.playerIndex is not None and o.playerIndex == mi and pk:
                return _my_attacker_score(pk)
            return base
        order = sorted(range(len(opts)), key=lambda i: -sc(i))
        return _mk(order, sel)

    # Card acquisition (search / draw / to hand) -> most useful cards.
    # Special case: energy-attach TARGET selection (e.g. Grimmsnarl ex "Punk Up"
    # attaches up to 5 Basic Dark to "your Pokémon"; the target select carries
    # my on-field Pokémon as options with cardId=None). _card_usefulness is blind
    # to field refs (returns 0 -> arbitrary order), so the accel landed on a random
    # basic/pre-evolution instead of the attacker. Route energy to my best attacker.
    if ctx in (SelectContext.TO_HAND, SelectContext.LOOK, SelectContext.EVOLVES_FROM,
               SelectContext.EVOLVES_TO, SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO):
        def acq_score(i):
            o = opts[i]
            if (o.cardId is None and o.playerIndex == mi
                    and o.area in (AreaType.ACTIVE, AreaType.BENCH)):
                pk = _pokemon_at(state, o)
                if pk is not None:
                    return 10000 + _my_attacker_score(pk)  # my field target -> best attacker
            # Recover-from-discard (Night Stretcher etc.): the option is a discard slot
            # (cardId=None), so _card_usefulness is blind to it -> resolve it and prefer
            # the biggest (evolved) Pokemon = the attacker worth chaining back.
            if o.cardId is None and o.area == AreaType.DISCARD and o.index is not None:
                disc = getattr(state.players[mi], "discard", None) or []
                if o.index < len(disc):
                    dcid = getattr(disc[o.index], "id", None)
                    c = _CARDS.get(dcid) if dcid is not None else None
                    if c is not None:
                        if c.cardType == CardType.POKEMON:
                            return 5000 + (c.hp or 0)
                        return 1000  # energy / other recoverable
            return _card_usefulness(o, state, mi, sel)
        order = sorted(range(len(opts)), key=lambda i: -acq_score(i))
        return _mk(order, sel)

    # Enemy energy removal (Enhanced Hammer etc.): a DISCARD_ENERGY whose options
    # point at the OPPONENT's Pokemon means "strip an energy from them" -- take it off
    # the most threatening / most-loaded opponent (deny a swing), NOT "my worst card".
    if ctx == SelectContext.DISCARD_ENERGY and any(
            o.playerIndex is not None and o.playerIndex != mi for o in opts):
        def esc(i):
            o = opts[i]
            pk = _pokemon_at(state, o)
            if pk is None:
                return (-1, 0, 0)
            is_active = 1 if o.area == AreaType.ACTIVE else 0
            return (is_active, len(pk.energies), _target_score(pk))
        order = sorted(range(len(opts)), key=esc, reverse=True)
        return _mk(order, sel)

    # Discard / return-to-deck -> least useful first
    if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM,
               SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD):
        order = sorted(range(len(opts)), key=lambda i: _card_usefulness(opts[i], state, mi, sel))
        return _mk(order, sel)

    # Counts -> draw/place the maximum (usually beneficial)
    if sel.type == SelectType.COUNT:
        # options are NUMBER; choose the largest number
        best = max(range(len(opts)), key=lambda i: (opts[i].number or 0))
        return _mk([best], sel)

    # default: take the first minCount options
    return _mk(list(range(len(opts))), sel)


def _best_attack_by_id(cid):
    c = _CARDS.get(cid) if cid is not None else None
    if not c:
        return 0
    best = 0
    for aid in c.attacks:
        a = _ATTACKS.get(aid)
        if a and a.damage and a.damage > best:
            best = a.damage
    return best


def _my_active_dmg(state, mi):
    """Best damage my Active can deal THIS turn (for gust-to-KO decisions).

    Uses the highest listed attack damage, plus known hand/board-scaling attacks
    whose listed damage is 0: Alakazam (743) "Powerful Hand" = 20 x my handSize.
    """
    if state is None:
        return 0
    me = state.players[mi]
    act = me.active[0] if me.active else None
    if act is None:
        return 0
    best = _best_attack_by_id(act.id)
    if act.id == 743:  # Alakazam -- Powerful Hand scales 20 x handSize
        best = max(best, 20 * (me.handCount or 0))
    return best


def _pokemon_at(state, o):
    if state is None or o.playerIndex is None or o.area is None or o.index is None:
        return None
    try:
        ps = state.players[o.playerIndex]
        if o.area == AreaType.ACTIVE:
            return ps.active[o.index]
        if o.area == AreaType.BENCH:
            return ps.bench[o.index]
    except (IndexError, TypeError):
        return None
    return None


_AREA_PILE = {AreaType.HAND: "hand", AreaType.DISCARD: "discard"}


def _opt_card_id(o, sel, state=None, mi=None):
    """Resolve the card an option refers to. Options are REFERENCES, not cards.

    ``cardId`` is None for every option that points into a zone; the card lives at
    ``<zone>[index]``. Nothing here read those zones, so such options scored 0, tied, and
    the sort returned an arbitrary order. Audited over 48 games (card-choice menus only):

        DECK 3599 blind 0   <- fixed earlier via sel.deck
        HAND 1010 blind 1010  <- "discard 2 cards from your hand" chosen AT RANDOM
        LOOKING 333 blind 333 <- Pokegear's top 7 etc: choosing 1 of 7 while blind to all 7
        PRIZE 1073 blind 0    <- face-down for BOTH seats: unresolvable BY DESIGN, correct

    ``sel.deck`` is empty on LOOKING menus, so the deck fix never covered them. Mirrors
    lm/actions.py::_card_at, which already resolved all of these correctly."""
    if o.cardId is not None:
        return o.cardId
    if _SEARCH_BLIND:                       # A/B switch: reproduce the old blind pick
        return None
    idx = o.index
    if idx is None:
        return None
    if o.type == OptionType.ENERGY and state is not None:
        # OptionType.ENERGY is a TWO-STEP reference and the only one of its kind:
        #   {"type":6, "area":ACTIVE, "index":0, "energyIndex":0}
        #    -> players[pi].<area>[index] picks the POKEMON,
        #    -> .energyCards[energyIndex] picks the energy CARD attached to it.
        # Resolving it one-step returned the Pokemon's id (or nothing), so every energy
        # tied and "which energy do I discard?" was arbitrary. Measured: 21 such menus in
        # 24 games, 11 of them (52%) with more than one candidate -- i.e. a real choice.
        # It matters: alakazam was ripping off its own Enriching Energy (ACE: attaching it
        # DRAWS 4 = +80 Powerful Hand damage) as readily as a Basic {P}.
        pi = o.playerIndex if o.playerIndex is not None else mi
        key = {AreaType.ACTIVE: "active", AreaType.BENCH: "bench"}.get(o.area)
        ei = getattr(o, "energyIndex", None)
        if key is None or ei is None:
            return None
        try:
            body = (getattr(state.players[pi], key, None) or [])[idx]
        except (IndexError, TypeError):
            return None
        ecs = getattr(body, "energyCards", None) or []
        return getattr(ecs[ei], "id", None) if ei < len(ecs) else None
    if o.area == AreaType.DECK:
        lst = getattr(sel, "deck", None) or []
    elif o.area == AreaType.LOOKING:
        lst = (getattr(state, "looking", None) or []) if state is not None else []
    elif o.area in _AREA_PILE and state is not None:
        if o.area == AreaType.HAND and not _RESOLVE_HAND:
            return None
        pi = o.playerIndex if o.playerIndex is not None else mi
        try:
            lst = getattr(state.players[pi], _AREA_PILE[o.area], None) or []
        except (IndexError, TypeError):
            return None
    else:
        # PRIZE is [null]*n for both seats -- face-down by the rules, so there is nothing
        # to resolve and every prize slot SHOULD tie.
        return None
    if idx < len(lst) and lst[idx]:
        return getattr(lst[idx], "id", None)
    return None


# Per-deck card importance. The ENGINE GENERATES THE LM'S TRAINING DATA, so a card
# ranking that is wrong for a deck teaches the LM to misplay that deck -- there is no
# budget for "close enough" here.
#
# A single global hierarchy (Pokemon 50 > Energy 40 > Supporter 35 > Item 25) cannot
# express what a deck actually wants, and measurably did not: alakazam's Rare Candy /
# Poffin / Dawn ARE its engine yet scored "Item/Supporter" and were discarded first;
# hydrapple's damage IS energy but "Pokemon > Energy" made its searches fetch bodies
# (-13pt). Copy count looked like a free per-deck signal but is not: a finished deck is
# almost all 4-ofs (alakazam 73%, crustle 87%), so it separates nothing.
#
# Tiers, highest first. Search fetches from the top; discards go from the bottom.
_TIER_VALUE = {
    "win":    900,   # the win condition itself
    "engine": 700,   # finds / enables / accelerates the win condition
    "line":   600,   # the evolution line's other pieces
    "fuel":   400,   # energy (still need-adjusted by _card_need)
    "tech":   200,   # situational, matchup-dependent
    "filler":  50,   # never actually wanted
}


def _card_tier_value(cid):
    """Explicit per-deck tier from tuning.json's card_roles, or None if unclassified."""
    if not _USE_ROLES:
        return None
    t = _ctx["card_roles"].get(cid) or _ctx["card_roles"].get(str(cid))
    return _TIER_VALUE.get(t) if t else None


def _deck_importance(cid):
    """How much does THIS DECK care about this card? Read from the deck itself.

    A single global hierarchy (Pokemon > Energy > Supporter > Item) cannot be right for
    every deck, and it demonstrably was not: alakazam's Rare Candy / Poffin / Poke Pad
    ARE its engine but score "Item = 25" and got discarded first; hydrapple's damage IS
    energy but "Pokemon > Energy" made searches fetch bodies (-13pt). Every attempt to
    give the engine BETTER INFORMATION (deck search, LOOKING, hand) then failed to move
    the win rate, or moved it in inconsistent directions -- because seeing more cards
    does nothing when the ranking applied to them is wrong.

    The deckbuilder already stated importance: COPY COUNT. A 4-of is something the deck
    wants every game; a 1-of is situational. That signal is per-deck and free.

    Exception: ACE SPEC is capped at 1 BY RULE, not by choice -- Hero's Cape / Prime
    Catcher are 1-ofs precisely because they are too strong to run in multiples. Treating
    them as marginal would throw away the best card in the deck.
    """
    n = _ctx["copies"].get(cid, 0)
    if not n:
        return 0
    c = _CARDS.get(cid)
    if c is not None and getattr(c, "aceSpec", False):
        return 300                    # 1-of by rule, not by preference
    if _is_energy(cid):
        return 0                      # fungible fuel; valued by need, not by count
    return {4: 300, 3: 200, 2: 100}.get(n, 0)    # 1-of -> situational tech


def _base_usefulness(cid):
    c = _CARDS.get(cid) if cid is not None else None
    if not c:
        return 0
    if c.cardType == CardType.POKEMON:
        s = 50 + _best_attack_by_id(cid) // 10 + (20 if (c.ex or c.megaEx) else 0)
    elif _is_energy(cid):
        s = 40
    elif c.cardType == CardType.SUPPORTER:
        s = 35
    elif c.cardType in (CardType.ITEM, CardType.TOOL):
        s = 25
    elif c.cardType == CardType.STADIUM:
        s = 15
    else:
        s = 10
    tv = _card_tier_value(cid)
    if tv is not None:
        # An explicit per-deck tier REPLACES the global guess (the small type-based
        # value is kept only as a tiebreak inside a tier).
        return tv + s // 10
    return s + (_deck_importance(cid) if _DECK_VALUE else 0)


def _card_usefulness(o, state, mi, sel=None):
    """Value of a card option, adjusted by what the board actually needs.

    The bare hierarchy (Pokemon 50+ > Energy 40 > Supporter 35 > Item 25) ignores the
    board. That only mattered once searches could see the deck at all: on engine_v2,
    hydrapple then took Pokemon over Energy every time and dropped 13pt, because its
    attackers read "30 more damage for each Energy attached" -- energy IS their damage.
    Three general signals fix it (hydrapple -13.0 -> +4.9)."""
    cid = _opt_card_id(o, sel, state, mi) if sel is not None else o.cardId
    if cid is None:
        return 0
    s = _base_usefulness(cid)
    c = _CARDS.get(cid)
    if not c or _NEED_AWARE is False:
        return s
    # THE DECK'S OWN WIN-CONDITION FIRST. _base_usefulness ranks Pokemon by PRINTED
    # damage, and a scaling attack prints 0: Alakazam's Powerful Hand is "20 x your hand
    # size", so Alakazam scores 50 -- dead last among its own line -- while Fezandipiti
    # ex scores 70 on the ex bonus alone. Blind-random used to fetch Alakazam sometimes;
    # once the search could see the deck it obediently took Fezandipiti EVERY time and
    # never the payoff card (-6.9pt, reproduced on two independent seed sets). Reuse the
    # mains/main_line context the engine already computes (_compute_main_line walks the
    # evolution line), so a deck's declared attacker outranks a shiny irrelevant ex.
    if cid in _ctx["mains"]:
        s += 500
    elif cid in _ctx["main_line"]:
        s += 200
    if state is None:
        return s
    try:
        me = state.players[mi]
        field = [p for p in ((me.active or []) + list(me.bench or [])) if p]
    except Exception:
        return s
    if _is_energy(cid):
        # (1) an attacker that cannot pay for its attack yet -> fuel is the bottleneck
        short = sum(1 for p in field if _cheapest_cost(p) > len(p.energies or []))
        if short:
            s += 30 + 10 * min(short, 3)
        # (2) an attacker whose damage SCALES with energy never stops wanting it
        for p in field:
            pc = _CARDS.get(p.id)
            if not pc:
                continue
            if any("for each" in ((_ATTACKS[a].text or "").lower())
                   and "energy" in ((_ATTACKS[a].text or "").lower())
                   for a in (pc.attacks or []) if _ATTACKS.get(a)):
                s += 40
                break
    elif c.cardType == CardType.POKEMON and getattr(c, "basic", False):
        # (3) a thin board loses to a single KO -- a body beats a spell
        if len(field) <= 2:
            s += 40
    return s


# ---- public entry point ----------------------------------------------------
def act(obs_dict, deck, style="aggro", hints=None, policy=None):
    """Return the option-index list for this observation.

    hints:  optional per-deck dict (main_attackers / accel / play) — see _ctx.
    policy: optional per-deck callable policy(obs, sel, deck) for bespoke decks.
            Called first on MAIN selections; return option indices to override,
            or None to fall through to the generic engine. Sub-selections still
            use the generic handlers. Engine helpers (OptionType, CardType,
            AreaType, _CARDS, _ATTACKS, _name, ...) are importable for policies.
    """
    _ctx["hints"] = hints or {}
    _compute_main_line(deck)
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return deck  # deck-selection phase
    sel = obs.select
    if not sel.option:
        return []
    try:
        if sel.context == SelectContext.MAIN:
            if policy is not None:
                r = policy(obs, sel, deck)
                if r is not None:
                    return _mk(r, sel)
            # Opt-in (policy.suppress_engine_retreat): keep the engine's development but
            # never let its fallback retreat fire -- the policy owns every retreat decision.
            ar = not (policy is not None and getattr(policy, "suppress_engine_retreat", False))
            return _choose_main(obs, sel, deck, style, allow_retreat=ar)
        # FULL-CONTROL policies (policy.full_control = True) are also consulted on
        # SUB-selects (spread targets, damage-counter destinations, gust picks, ...)
        # so a policy can carry a multi-turn plan (e.g. commit spread damage onto one
        # kill target). Return None to fall through to the generic handlers.
        if policy is not None and getattr(policy, "full_control", False):
            r = policy(obs, sel, deck)
            if r is not None:
                return _mk(r, sel)
        return _choose_sub(obs, sel, deck, style)
    except Exception:
        if os.environ.get("ENGINE_DEBUG"):
            raise
        # never crash the match: fall back to a legal minimal selection
        return _mk(list(range(len(sel.option))), sel)
