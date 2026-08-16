"""Engine v2 — L0 generic ``BasePolicy`` (see docs/engine_v2_spec.md §2, §8).

A deck-agnostic **develop-and-attack floor**. It is NOT aggro-specialised: a
control/combo deck that falls through to L0 alone still plays passably. The shape
looks aggro only because attacking = taking prizes = the universal win condition;
aggro-specific optimisations (minimal bench, T1-2 rush, never-retreat, aggressive
gust) deliberately live in the L1 ``AggroPolicy``, not here.

Design:
* STATELESS across matches: every decision derives from ``obs`` alone, so one
  instance can (in principle) pilot both seats. The only per-instance state is the
  deck + inferred roles, both fixed at construction.
* Decision logic is split into **perception** (read-only ``assess_*`` building a
  ``Ctx``) and **decision** (``decide_*`` returning option indices or ``None``).
  ``choose_main`` walks a priority ladder that bridges to the deciders;
  ``choose_sub`` reuses the SAME deciders. L1/L2 subclasses override only the
  methods they change.
* Non-breaking: the legacy ``agents/_engine.py`` stays; decks opt in via
  ``make_policy(deck, profile).act`` or the compat ``act()`` shim.
"""

import os
import re
from collections import Counter

from cg.api import (
    AreaType, CardType, EnergyType, OptionType, SelectContext, SelectType,
)

# Reuse the (static, hint-free) card/attack DBs and keyword sets from the legacy
# engine so there is a single source of truth. We deliberately do NOT import the
# _ctx-dependent scorers (_my_attacker_score / _main_boost) — L0 is hint-free.
from agents._engine import (
    _CARDS, _ATTACKS, _name, _has,
    _DRAW_SUPPORTERS, _GUST, _SEARCH_ITEMS, _ENERGY_ACCEL, _SWITCH_CARDS,
    _RARE_CANDY,
)

# Recovery trainers (discard -> hand/deck). Kept local; small curated set.
_RECOVERY = ("Night Stretcher", "Energy Recycler", "Super Rod", "Ordinary Rod",
             "Pal Pad", "Klara", "Miriam", "Energy Retrieval",
             "Superior Energy Retrieval")

_HIGH_HP_WALL = 120  # HP threshold for the wall heuristic


# --------------------------------------------------------------------------- #
# card-fact helpers (no _ctx, no hints — pure card DB)                          #
# --------------------------------------------------------------------------- #
_SCALING_NOMINAL = 30   # nominal damage credited to a listed-0 scaling attack


def _is_scaling_attack(a):
    """A listed-0-damage attack that actually deals damage that scales (e.g.
    Alakazam's Powerful Hand = 20x hand size, listed as 0). Card-fact based
    (reads the card's own attack text) — NOT a per-deck hint."""
    raw = a.text or ""
    t = raw.lower()
    return ("for each" in t) or ("×" in raw) or ("place" in t and "damage counter" in t)


# L0 text-based damage estimator (pipeline v2.1, layer fix for the fleet-wide
# "display-0 utility attack declined" pathology: Cruel Arrow 0->100 snipe,
# Full Moon Rondo 20->est, Garland Ray 0->240, Split Bomb 0->120...).
# Deterministic card-fact parsing, NOT per-deck hints. Estimates are
# deliberately conservative (count=3) except exact forms ("up to N").
_RE_SNIPE = re.compile(r"does (\d+) damage to 1 of your opponent")
_RE_MULTI = re.compile(r"does (\d+) damage to each(?: of (\d+))?")
_RE_EACH = re.compile(r"(\d+)\s*(?:more\s+)?damage\s+for\s+each", re.I)
_RE_UPTO = re.compile(r"up to (\d+)", re.I)
_RE_E_SEARCH = re.compile(r"search your deck for (?:up to \d+|a) basic(?: {\w})? energy", re.I)
# Discard RECOVERY. Derived from the pool, not guessed: this matches exactly the four
# such cards our decks run -- Night Stretcher (1097, in 47 of 60 decks), Energy Retrieval
# (1118), Lana's Aid (1184), Tarragon (1238). They never touch the DECK, which is why
# every keyword bucket in decide_trainer missed them and Night Stretcher was played
# **1 time in 1372 offers**. Wally's Compassion deliberately does NOT match (it returns
# energy from a Pokemon, not from the discard) and keeps its bespoke per-deck rule.
_RE_RECOVER = re.compile(r"from your discard pile into your hand", re.I)
_RE_EXACTKO = re.compile(r"exactly (\d+) damage counters.{0,40}knocked out", re.I | re.S)
_RE_AB_SWITCH = re.compile(r"switch 1 of your benched .{0,60}with your active", re.I)
_RE_MOVER_AB = re.compile(r"move up to \d+ damage counters from 1 of your", re.I)
_RE_SELF_COUNTERS = re.compile(r"put (\d+) damage counters on this pok", re.I)
# v2.4 conditional-damage clauses (5 classes surveyed fleet-wide, 16 attacks)
_RE_OPPDMG_BONUS = re.compile(
    r"opponent.{1,3}s active pok\S*mon already has any damage counters on it, "
    r"this attack does (\d+) more damage", re.I)
_RE_OPPDMG_BASE = re.compile(
    r"opponent.{1,3}s active pok\S*mon already has any damage counters on it, "
    r"this attack.{1,3}s base damage is (\d+)", re.I)
_RE_OPPNODMG_NOTHING = re.compile(
    r"opponent.{1,3}s active pok\S*mon has no damage counters on it before "
    r"this attack does damage, this attack does nothing", re.I)
_RE_SELFDMG_BONUS = re.compile(
    r"if this pok\S*mon has any damage counters on it, "
    r"this attack does (\d+) more damage", re.I)
_RE_SELFNODMG_BONUS = re.compile(
    r"if this pok\S*mon has no damage counters on it, "
    r"this attack does (\d+) more damage", re.I)


def _skill_text(card):
    return (card.skills[0].text if getattr(card, "skills", None) else "") or ""


_RE_EACH_WHAT = re.compile(
    r"damage\s+for\s+each\s+(.*?)(?:\.|,|$)", re.I)
# Energy symbol in rules text -> EnergyType. DERIVED from the card DB (see
# _ctx_each_count): for every attack naming exactly one {X}, which EnergyType the card
# itself is. {C}/{N} are never used as a "for each {X} Energy" filter in this pool and
# are mapped by convention.
# A/B switch (house idiom, cf. _SPARE_EX_GUARD / _USE_ROLES): typed-energy and
# stage-filtered-bench resolution in _ctx_each_count. Off == the pre-2026-07-17 model.
_EACH_TYPED = True
_RE_EACH_ETYPE = re.compile(r"\{([A-Z])\}\s*energy", re.I)
_RE_DISCARD_MY_ENERGY = re.compile(
    r"discard any amount of basic energy from your pok", re.I)
_SYMBOL_ETYPE = {"G": 1, "R": 2, "W": 3, "L": 4, "P": 5,
                 "F": 6, "D": 7, "M": 8, "N": 9, "C": 0}


def _ctx_each_count(ctx, text):
    """Resolve a "for each ..." count against the LIVE board (L0 dynamic
    estimator, pipeline v2.1). Returns None when the subject is not board-
    derivable (coin flips etc.) so callers fall back to the static estimate."""
    m = _RE_EACH_WHAT.search(text or "")
    if m is None:
        return None
    what = m.group(1).lower()
    me, opp = ctx.me, ctx.opp
    if "benched pok" in what or "on your bench" in what:
        if "both" in (text or "").lower():
            return len(me.bench) + len(opp.bench)
        if "opponent" in what:
            return len(opp.bench)
        # FILTERED bench ("for each Stage 2 Pokemon on your Bench" -- Mamoswine ex's
        # Rumbling March, 180 + 40x). Unfiltered "benched Pokemon" keeps counting all.
        if _EACH_TYPED and "stage 2" in what:
            return sum(1 for v in me.bench
                       if getattr(_CARDS.get(v.id), "stage2", False))
        if _EACH_TYPED and "stage 1" in what:
            return sum(1 for v in me.bench
                       if getattr(_CARDS.get(v.id), "stage1", False))
        return len(me.bench)
    if "card" in what and "hand" in what:
        return ctx.opp_ps.handCount if "opponent" in what else ctx.me.hand_count
    if "damage counter on this" in what:
        a = me.active
        return ((a.max_hp - a.hp) // 10) if a is not None else None
    # NOTE: resolving "70 damage for each card you discarded in this way" from the board
    # (count our own basic energy when the clause is "discard any amount of Basic Energy
    # from your Pokemon") was implemented and REVERTED -- it changes nothing. On
    # iono_bellibolt, the fleet's worst deck (24.2%), the theory was that Raging Bolt ex's
    # Bellowing Thunder is under-valued at a static 210 and so loses to Iono's Bellibolt
    # ex's printed 230, firing only 1% of attacks. But at the moment of evaluation the
    # board holds ~3 basic energy, so the dynamic count gives 70*3 = **210 -- exactly the
    # static value**. Measured at 150 games x 4 panel decks: Bellowing Thunder 0.38 ->
    # 0.39/game, win 24% -> 21%. The ranking never moved. The deck's real problem is that
    # its payoff needs the energy CONCENTRATED first (Electric Streamer peaks at 6.15 =
    # a real 420) while the 230 self-locking attack is always available -- a sequencing
    # problem, not a valuation one. Do not re-try the valuation angle.
    if "energy attached" in what:
        # TYPED energy. The clause is usually "for each {G} Energy attached to all of
        # your Pokemon" (Hydrapple ex Syrup Storm) / "{P}" (Mega Gardevoir ex Mega
        # Symphonia). Counting ALL energy here overcounts every multi-type board, which
        # is why those decks each hand-patch _expected_dmg. Symbol->EnergyType is derived
        # from the card DB, not assumed: for attacks naming exactly one symbol, it
        # co-occurs with the card's own energyType {G}->GRASS x15, {R}->FIRE x10,
        # {W}->WATER x10, {P}->PSYCHIC x9, {L}->LIGHTNING x8, {D}->DARKNESS x7,
        # {F}->FIGHTING x7, {M}->METAL x5 (single-card runner-ups are noise).
        sym = _RE_EACH_ETYPE.search(what) if _EACH_TYPED else None
        want_t = _SYMBOL_ETYPE.get(sym.group(1).upper()) if sym else None

        def _n(views):
            views = [v for v in views if v is not None]
            if want_t is None:
                return sum(v.energy_count for v in views)
            return sum(sum(1 for t in v.energy if t == want_t) for v in views)

        if "attached to this" in what:
            return _n([me.active]) if me.active is not None else None
        if "opponent" in what:                    # Alakazam's Psychic: opp's Active
            return _n([opp.active]) if opp.active is not None else None
        if "all of your" in what or "your pok" in what:
            return _n(me.inplay())
        return None
    return None


def _atk_value(a):
    """Effective damage of an attack for ranking (scaling/utility 0-dmg ->
    text-derived estimate; last resort nominal)."""
    if a is None:
        return 0
    d = a.damage or 0
    t = a.text or ""
    m = _RE_SNIPE.search(t)
    if m:
        d += int(m.group(1))
    m = _RE_MULTI.search(t)
    if m:
        d += int(m.group(1)) * int(m.group(2) or 1)
    m = _RE_EACH.search(t)
    if m:
        per = int(m.group(1))
        up = _RE_UPTO.search(t)
        d += per * (int(up.group(1)) if up else 3)
    if _RE_EXACTKO.search(t):
        d = max(d, 250)          # conditional instant-KO: strong static rank value
    if d:
        return d
    return _SCALING_NOMINAL if _is_scaling_attack(a) else 0


def _best_attack(card):
    """(attackId, value) of the highest-value attack, or (None, -1). ``value``
    counts scaling 0-damage attacks at a nominal so they read as attacks."""
    best = (None, -1)
    if not card:
        return best
    for aid in card.attacks:
        a = _ATTACKS.get(aid)
        if a is None:
            continue
        v = _atk_value(a)
        if v > best[1]:
            best = (aid, v)
    return best


def _best_dmg(card):
    return _best_attack(card)[1] if card else -1


def _attack_cost(aid):
    a = _ATTACKS.get(aid)
    return len(a.energies) if a and a.energies else 0


def _cheapest_cost(card):
    """Minimum energy cost among this card's damaging attacks (0 if none)."""
    if not card:
        return 0
    costs = [_attack_cost(a) for a in card.attacks
             if _ATTACKS.get(a) and _ATTACKS[a].damage]
    return min(costs) if costs else 0


def _main_cost(card):
    """Energy cost of the card's BEST (highest-damage) attack — what a primary
    attacker actually wants to reach, not merely its cheapest chip."""
    aid, _ = _best_attack(card)
    c = _attack_cost(aid) if aid is not None else 0
    return c or _cheapest_cost(card)


def _can_pay(cost, have):
    """Can ``have`` (list[EnergyType]) satisfy attack ``cost`` (list[EnergyType])?

    COLORLESS in the cost is any energy. RAINBOW / TEAM_ROCKET on the Pokémon are
    treated as flexible providers (wildcard). Approximate but adequate for the
    floor — actual ATTACK legality always comes from the offered options.
    """
    pool = list(have)
    typed = [e for e in cost if e != EnergyType.COLORLESS]
    colorless = sum(1 for e in cost if e == EnergyType.COLORLESS)
    wild = {EnergyType.RAINBOW, EnergyType.TEAM_ROCKET}
    for t in typed:
        idx = next((i for i, h in enumerate(pool) if h == t or h in wild), None)
        if idx is None:
            return False
        pool.pop(idx)
    return len(pool) >= colorless


def _prize_value(pk):
    c = _CARDS.get(pk.id)
    if not c:
        return 1
    v = 3 if c.megaEx else 2 if c.ex else 1
    for e in getattr(pk, "energyCards", None) or []:
        if _name(e.id) == "Legacy Energy":
            v -= 1
    return max(0, v)


def _target_score(pk):
    """Value of damaging/KOing this opponent Pokémon (prize > loaded > evolved)."""
    c = _CARDS.get(pk.id)
    score = _prize_value(pk) * 1000
    score += len(pk.energies) * 120
    score += len(getattr(pk, "tools", None) or []) * 80
    if c:
        if c.stage2:
            score += 220
        elif c.stage1:
            score += 110
    score += pk.hp
    return score


def _attacker_score(view):
    """How good an attacker this Pokémon of MINE is (for attach/promote/target).

    Driven by BASE potential damage + ex — NOT by attached energy, so energy does
    not snowball onto a non-attacking middle-evolution. Hint-free (unlike the
    legacy _my_attacker_score). Returns -1 for a non-attacker.
    """
    if view is None or view.card is None:
        return -1
    dmg = view.best_potential_dmg
    if dmg <= 0:
        return -1
    s = dmg
    if view.card.ex or view.card.megaEx:
        s += 100
    s += min(view.energy_count, 1) * 10  # tiny tie-break toward a started attacker
    return s


# Per-deck card importance (tuning.json card_roles), highest first. Searches fetch from
# the top, discards go from the bottom. A single global hierarchy cannot be right for
# every deck and measurably was not: alakazam's Rare Candy / Poffin ARE its engine yet
# score "Item"; hydrapple's damage IS energy but "Pokemon > Energy" made its searches
# fetch bodies (-13pt). Copy count looked like a free per-deck signal but is not -- a
# finished deck is almost all 4-ofs (alakazam 73%, crustle 87%), separating nothing.
_TIER_VALUE = {
    "win":    900,   # the win condition itself
    "engine": 700,   # finds / enables / accelerates the win condition
    "line":   600,   # the evolution line's other pieces
    "fuel":   400,   # energy (still need-adjusted by _card_need)
    "tech":   200,   # situational, matchup-dependent
    "filler":  50,   # never actually wanted
}
_USE_ROLES = True    # A/B switch for explicit per-deck card_roles

# Discard-order sentinels for per-deck doctrine ("never throw this" / "throw this first").
# They must out-rank the whole _card_need range, which the old literals no longer do:
# those were 1000 / -100, chosen against _card_usefulness (0..~100), but a tiered score
# reaches 900 + board bonuses + a +500 protect ~= 1470 -- i.e. "never" would have lost.
_KEEP = 10 ** 6
_SHED = -(10 ** 6)


def _norm_name(s):
    """Normalize a card name for matching: the DB mixes BOTH apostrophes.

    64 of the 65 "Team Rocket's ..." cards use U+0027 ('), one uses U+2019 (’); across the
    whole pool 53 names carry a curly apostrophe. A literal `"Team Rocket's" in c.name`
    therefore matches almost-but-not-quite, and the miss is SILENT -- it reads exactly like
    "this card just isn't a Team Rocket's card". Today only Team Rocket's Hypnotizer (1154)
    slips through and no deck runs it, so this is a latent bug, not a live one; it is fixed
    here because the failure mode is indistinguishable from correct behaviour.

    MATCH ON CARD ID WHERE YOU CAN. This exists for the genuinely name-keyed rules (a card
    FAMILY like "Team Rocket's *" has no id list)."""
    return (s or "").replace("’", "'").lower()


def _card_usefulness(card):
    """Rough keep/discard value of a card (higher = keep)."""
    if not card:
        return 0
    if card.cardType == CardType.POKEMON:
        return 50 + max(_best_dmg(card), 0) // 10 + (20 if (card.ex or card.megaEx) else 0)
    if card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
        return 40
    if card.cardType == CardType.SUPPORTER:
        return 35
    if card.cardType in (CardType.ITEM, CardType.TOOL):
        return 25
    if card.cardType == CardType.STADIUM:
        return 15
    return 10


# --------------------------------------------------------------------------- #
# perception objects                                                           #
# --------------------------------------------------------------------------- #
class PokemonView:
    """A field Pokémon decoded into a convenient shape (spec §8.2)."""

    __slots__ = ("pk", "id", "card", "hp", "max_hp", "energy", "energy_count",
                 "appear_this_turn", "ready", "best_ready_dmg",
                 "best_potential_dmg", "role")

    def __init__(self, pk, role=None):
        self.pk = pk
        self.id = pk.id
        self.card = _CARDS.get(pk.id)
        self.hp = pk.hp
        self.max_hp = pk.maxHp
        self.energy = list(pk.energies or [])
        self.energy_count = len(self.energy)
        self.appear_this_turn = pk.appearThisTurn
        self.role = role
        self.ready = []            # [(aid, dmg, cost)] payable with current energy
        self.best_ready_dmg = 0
        self.best_potential_dmg = 0
        if self.card:
            for aid in self.card.attacks:
                a = _ATTACKS.get(aid)
                if not a:
                    continue
                dmg = _atk_value(a)
                self.best_potential_dmg = max(self.best_potential_dmg, dmg)
                if _can_pay(list(a.energies or []), self.energy):
                    self.ready.append((aid, dmg, len(a.energies or [])))
                    self.best_ready_dmg = max(self.best_ready_dmg, dmg)

    def can_ko(self, target_hp):
        return self.best_ready_dmg >= target_hp

    @property
    def loaded(self):
        """Has enough energy to fire its cheapest damaging attack."""
        return self.card is not None and self.best_potential_dmg > 0 \
            and self.energy_count >= _cheapest_cost(self.card)


class SideView:
    """One player's board (spec §8.2)."""

    __slots__ = ("ps", "is_me", "active", "bench", "hand_count", "prizes_left",
                 "deck_count", "energy_in_play")

    def __init__(self, ps, roles, is_me):
        self.ps = ps
        self.is_me = is_me
        act = ps.active[0] if (ps.active and ps.active[0] is not None) else None
        self.active = PokemonView(act, roles.get(act.id) if act else None) if act else None
        self.bench = [PokemonView(p, roles.get(p.id)) for p in (ps.bench or []) if p is not None]
        self.hand_count = ps.handCount
        self.prizes_left = len(ps.prize or [])
        self.deck_count = ps.deckCount
        self.energy_in_play = sum(v.energy_count for v in self.inplay())

    def inplay(self):
        return ([self.active] if self.active else []) + list(self.bench)


class PrizeInfo:
    __slots__ = ("mine", "opp", "can_close", "posture")

    def __init__(self, mine, opp, can_close, posture):
        self.mine = mine          # my prizes still to take
        self.opp = opp            # opponent's prizes still to take
        self.can_close = can_close  # a KO this turn wins the game
        self.posture = posture    # "race" (ahead/even) | "stall" (behind)


class Ctx:
    """Per-call decision context (spec §8.2). Alive only during one act()."""

    __slots__ = ("obs", "sel", "mi", "state", "me_ps", "opp_ps", "me", "opp",
                 "roles", "evolves", "abilities", "plays", "attaches", "attacks",
                 "retreat_idx", "end_idx", "my_setup", "opp_threat", "ko_targets",
                 "prize", "my_active_dmg")

    def __init__(self, obs, sel, mi, me, opp, roles):
        self.obs = obs
        self.sel = sel
        self.mi = mi
        self.state = obs.current
        self.me_ps = obs.current.players[mi]
        self.opp_ps = obs.current.players[1 - mi]
        self.me = me
        self.opp = opp
        self.roles = roles
        self.evolves = []
        self.abilities = []
        self.plays = []
        self.attaches = []
        self.attacks = []
        self.retreat_idx = None
        self.end_idx = None
        self.my_setup = None
        self.opp_threat = None
        self.ko_targets = []
        self.prize = None
        self.my_active_dmg = 0

    def bucket(self):
        for i, o in enumerate(self.sel.option):
            t = o.type
            if t == OptionType.EVOLVE:
                self.evolves.append(i)
            elif t == OptionType.ABILITY:
                self.abilities.append(i)
            elif t == OptionType.PLAY:
                self.plays.append(i)
            elif t == OptionType.ATTACH:
                self.attaches.append(i)
            elif t == OptionType.ATTACK:
                self.attacks.append(i)
            elif t == OptionType.RETREAT:
                self.retreat_idx = i
            elif t == OptionType.END:
                self.end_idx = i

    # option -> concrete card / pokemon resolution
    def hand_card(self, o):
        h = self.me_ps.hand
        if h is not None and o.index is not None and 0 <= o.index < len(h):
            return _CARDS.get(h[o.index].id)
        return None

    def field_pk(self, o):
        """The field Pokémon an option's inPlay* / area+index points at."""
        area = o.inPlayArea if o.inPlayArea is not None else o.area
        idx = o.inPlayIndex if o.inPlayIndex is not None else o.index
        pi = o.playerIndex if o.playerIndex is not None else self.mi
        try:
            ps = self.state.players[pi]
            if area == AreaType.ACTIVE:
                return ps.active[0]
            if area == AreaType.BENCH:
                return ps.bench[idx]
        except (IndexError, TypeError):
            return None
        return None

    def opp_pokemon_at(self, o):
        if o.playerIndex is None or o.area is None or o.index is None:
            return None
        try:
            ps = self.state.players[o.playerIndex]
            if o.area == AreaType.ACTIVE:
                return ps.active[0]
            if o.area == AreaType.BENCH:
                return ps.bench[o.index]
        except (IndexError, TypeError):
            return None
        return None


# --------------------------------------------------------------------------- #
# L0 policy                                                                    #
# --------------------------------------------------------------------------- #
class BasePolicy:
    archetype = "base"
    draw_threshold = 5      # play a draw supporter when hand <= this
    bench_target = 3        # develop bench up to this many Pokémon
    # A/B switch for the decide_bench ordering change (see decide_bench docstring).
    attack_min_dmg = 10     # attack if it does at least this much
    deck_low = 4            # stop optional search/draw at/below this deck size (anti-deckout; calibrated: 8 hurt healthy decks, 4 only fires near true deckout)

    def __init__(self, deck, profile=None):
        self._read_line(profile)
        self.deck = list(deck)
        self.profile = profile or {}
        # Per-deck card importance from tuning.json (see _TIER_VALUE). gen_selfplay and
        # make_policy already hand us the whole tuning entry as `profile`, so this was
        # authored and then never read: engine_v2 scored every card with the same fixed
        # Pokemon>Energy>Supporter>Item guess the legacy engine was measured to be wrong
        # about. JSON keys are strings; the option path yields int ids.
        self.card_roles = {int(k): v for k, v in (self.profile.get("card_roles") or {}).items()}
        # PER-DECK supporter tuning. Both default to the class attribute / empty, so a
        # profile without them behaves exactly as before -- the point is that a deck can
        # fix its own piloting without touching the 59 others.
        #   draw_threshold: "play a draw supporter only while hand <= N". Sensible as a
        #     fleet default, CATASTROPHIC for alakazam, whose Powerful Hand IS 20 x hand
        #     size: the gate caps the hand at ~5-6 and therefore caps the damage. The
        #     live alakazam opponents that beat v29 2-11 attack with a **13.92**-card
        #     hand (278 dmg); ours attacked with 10.33 (207).
        #   draw_supporters: extra card IDs that count as draw supporters for THIS deck.
        #     _DRAW_SUPPORTERS is a global NAME list, so a card missing from it is
        #     INVISIBLE to named(), not deprioritised. Dawn (1231) -- "search your deck
        #     for a Basic, a Stage 1 AND a Stage 2 ... put them into your hand" -- is the
        #     whole Abra->Kadabra->Alakazam line AND +3 cards of hand (= +60 damage). It
        #     was offered on 451 menus and played 14 times; the live agents play it
        #     1.77/game. Adding it globally would touch the 14 decks that run it, so it
        #     is per-deck.
        if self.profile.get("draw_threshold") is not None:
            self.draw_threshold = int(self.profile["draw_threshold"])
        #   bench_target: how many bodies to develop. ConfigL2 already reads this from
        #     its `line` config, but a bespoke L2 (AlakazamL2 is a ComboPolicy) was stuck
        #     with the class attribute. Measured: our alakazam holds a 2.41 bench vs the
        #     3.25 of the live agents that beat v29 2-11.
        if self.profile.get("bench_target") is not None:
            self.bench_target = int(self.profile["bench_target"])
        # Keyword buckets match on card NAME, so a card whose name does not look like
        # its function is invisible to decide_trainer no matter how central it is. The
        # `draw_supporters` escape hatch existed for exactly that; generalise it, because
        # the same hole silently killed rockets_spidops' whole engine -- Team Rocket's
        # Transceiver (its Supporter search) was offered 246 times and played **0**,
        # Proton 0/135, Giovanni 0/109. Per-deck opt-in; no global name-list growth.
        self._extra_draw_supporters = frozenset(self.profile.get("draw_supporters") or ())
        # HAND DISRUPTION (Xerosic's Machinations class): "opponent discards down to 3".
        # No keyword bucket described disruption, so it was a dead slot in 7 decks (audit:
        # offered 4694x, played 0). It is per-deck opt-in because it is a STRATEGY, not a
        # staple -- only control/stall lists want it. Probe vs alakazam: playable ~ every
        # turn it is in hand, and the opponent sat on a 12-14 card hand (Powerful Hand is
        # 20x hand size, so alakazam HOARDS) -- stripping it to 3 attacks the win condition
        # itself. Gated on opp hand size so it never fires for <2 cards of value.
        self._disrupt_cards = frozenset(self.profile.get("disrupt_cards") or ())
        self._disrupt_min = int(self.profile.get("disrupt_min_opp_hand", 5))
        self._bucket_extra = {
            id(_DRAW_SUPPORTERS): self._extra_draw_supporters,
            id(_SEARCH_ITEMS): frozenset(self.profile.get("search_items") or ()),
            id(_GUST): frozenset(self.profile.get("gust_cards") or ()),
            id(_SWITCH_CARDS): frozenset(self.profile.get("switch_cards") or ()),
            id(_ENERGY_ACCEL): frozenset(self.profile.get("energy_accel") or ()),
        }
        self._setup_by_role = bool(self.profile.get("setup_by_role"))
        self.roles = self.infer_roles(self.deck)
        # role sets used by L1 archetype policies
        self.primary_ids = {c for c, r in self.roles.items()
                            if r.get("role") == "attacker" and r.get("tier") == "primary"}
        self.backup_ids = {c for c, r in self.roles.items()
                           if r.get("role") == "attacker" and r.get("tier") == "backup"}
        self.wall_ids = {c for c, r in self.roles.items() if r.get("role") == "wall"}
        # TYPE-DEAD attackers: pokemon whose damaging attacks can NEVER be paid by
        # this deck's energy types (e.g. Munkidori's Mind Bend needs {P} in a
        # mono-Dark marnie deck). The count-based ``loaded`` heuristic marks them
        # "charged" and the ping-pong-safe retreat then refuses to swap them out —
        # a wall gets stuck in the Active forever. Computed once from the deck.
        etypes = {_CARDS[c].energyType for c in set(deck)
                  if _CARDS.get(c) and _CARDS[c].cardType in
                  (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)}
        wild = {EnergyType.RAINBOW, EnergyType.TEAM_ROCKET}
        self.type_dead_ids = set()
        for cid in set(deck):
            c = _CARDS.get(cid)
            if not c or c.cardType != CardType.POKEMON:
                continue
            payable = False
            for aid in c.attacks:
                a = _ATTACKS.get(aid)
                if not a or not (a.damage or _is_scaling_attack(a)):
                    continue
                typed = {e for e in (a.energies or []) if e != EnergyType.COLORLESS}
                if typed <= (etypes | wild) or (etypes & wild):
                    payable = True
                    break
            if not payable and _best_dmg(c) > 0:
                self.type_dead_ids.add(cid)

    # ---- shared attach helpers (used by L0 + L1 subclasses) --------------- #
    def _energy_attach_opts(self, ctx):
        """Indices of ATTACH options whose hand card is an Energy (1/turn limited)."""
        opt = ctx.sel.option
        out = []
        for i in ctx.attaches:
            c = ctx.hand_card(opt[i])
            if c and c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                out.append(i)
        return out

    def _tool_attach_opts(self, ctx):
        return [i for i in ctx.attaches if i not in set(self._energy_attach_opts(ctx))]

    def _target_view(self, ctx, o):
        pk = ctx.field_pk(o)
        return PokemonView(pk, self.roles.get(pk.id)) if pk is not None else None

    def _type_fit(self, ctx, o, view):
        """+bonus if attaching THIS energy card advances an UNMET TYPED requirement
        of the target's best attack (fleet fix: L0 attaches by damage without type,
        so a mixed-cost attacker like a Fire+Psychic spreader never gets the 1 Fire
        it strictly needs and can never fire — dragapult H1, generalises to every
        multi-type deck)."""
        if view is None or view.card is None:
            return 0
        card = ctx.hand_card(o)
        if card is None or card.cardType not in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            return 0
        et = card.energyType
        aid = _best_attack(view.card)[0]
        a = _ATTACKS.get(aid) if aid is not None else None
        if not a:
            return 0
        need = Counter(e for e in (a.energies or []) if e != EnergyType.COLORLESS)
        if not need:
            return 0
        have = Counter(view.energy)
        wild = {EnergyType.RAINBOW, EnergyType.TEAM_ROCKET}
        for t, n in need.items():
            if have.get(t, 0) + sum(have[w] for w in wild) < n and (et == t or et in wild):
                return 300                          # fills a missing typed requirement
        return 0

    def _energy_cap(self, view):
        """Energy a primary attacker should be capped at: its declared energy_need,
        else the cost of its BEST attack (load to full power, THEN spill to backup)."""
        need = view.role.get("energy_need") if view.role else None
        return need if need else _main_cost(view.card)

    # ----- public entry ---------------------------------------------------- #
    def act(self, obs_dict):
        from cg.api import to_observation_class
        obs = to_observation_class(obs_dict)
        # keep the RAW dict alongside: lm/hidden decodes the engine's hidden effect state
        # out of obs["search_begin_input"], which to_observation_class does not carry.
        self._raw_obs = obs_dict
        if obs.select is None:
            return self.deck                      # deck-selection phase
        sel = obs.select
        if not sel.option:
            return []
        try:
            ctx = self._perceive(obs, sel)
            r = self.choose_main(ctx) if sel.context == SelectContext.MAIN \
                else self.choose_sub(ctx)
            return self._mk(r, sel)
        except Exception:
            if os.environ.get("ENGINE_DEBUG"):
                raise
            return self._mk(list(range(len(sel.option))), sel)

    # ----- perception ------------------------------------------------------ #
    def _perceive(self, obs, sel):
        state = obs.current
        mi = state.yourIndex
        me = self.assess_self(state, mi)
        opp = self.assess_opponent(state, mi)
        ctx = Ctx(obs, sel, mi, me, opp, self.roles)
        ctx.bucket()
        ctx.my_active_dmg = self._my_active_dmg(ctx)
        ctx.my_setup = self.assess_self_setup(me, opp)
        ctx.opp_threat = self.assess_opponent_setup(opp, me)
        ctx.ko_targets = self.assess_ko_targets(ctx)
        ctx.prize = self.assess_prize_race(ctx)
        # the live context, for the few scorers whose signature predates Ctx (_bench_score /
        # _setup_score take a bare cardId). Read-only, and only valid inside this act().
        self._ctx = ctx
        return ctx

    def assess_self(self, state, mi):
        return SideView(state.players[mi], self.roles, True)

    def assess_opponent(self, state, mi):
        return SideView(state.players[1 - mi], self.roles, False)

    def assess_self_setup(self, me, opp):
        if me.active is None:
            return "MULLIGAN"
        active_can_attack = me.active.best_potential_dmg > 0
        if not active_can_attack and any(v.best_potential_dmg > 0 for v in me.bench):
            return "STALLED"                       # active is a non-attacker, better waits
        if me.active.loaded and me.bench:
            return "READY"
        return "DEVELOPING"

    def assess_opponent_setup(self, opp, me):
        if opp.active is None or me.active is None:
            return "LOW"
        if opp.active.best_ready_dmg and opp.active.best_ready_dmg >= me.active.hp:
            return "CAN_KO_ME_NOW"
        if opp.active.best_potential_dmg and opp.active.best_potential_dmg >= me.active.hp:
            return "CAN_KO_ME_NEXT"
        return "LOW"

    def assess_ko_targets(self, ctx):
        """Opponent Pokémon my Active could KO this turn (spec judgment 5)."""
        dmg = ctx.my_active_dmg
        if dmg <= 0:
            return []
        return [v for v in ctx.opp.inplay() if v.hp <= dmg]

    # ---- reusable opponent-board perception (add methods as judgment needs) - #
    def ko_targets_with(self, ctx, dmg, bench_only=False):
        """Opponent Pokémon a GIVEN damage would KO (for scaling attackers whose
        real damage differs from ctx.my_active_dmg, and for gust/Boss planning)."""
        pool = list(ctx.opp.bench) if bench_only else ctx.opp.inplay()
        return [v for v in pool if 0 < v.hp <= dmg]

    def opp_energy_pokemon(self, ctx, special_only=False):
        """Opponent Pokémon carrying (special) Energy — for energy-denial decisions
        (Enhanced Hammer / Crushing Hammer / Xerosic). Reads the opponent board."""
        out = []
        for v in ctx.opp.inplay():
            cards = v.pk.energyCards or []
            if special_only:
                hit = any(_CARDS.get(c.id) and _CARDS[c.id].cardType == CardType.SPECIAL_ENERGY
                          for c in cards)
            else:
                hit = bool(cards)
            if hit:
                out.append(v)
        return out

    def assess_prize_race(self, ctx):
        """Prizes still to take + whether a KO closes the game (spec judgment 6)."""
        me, opp = ctx.me, ctx.opp
        can_close = False
        if opp.active is not None and ctx.my_active_dmg >= opp.active.hp:
            # KOing the active wins if it takes our last prize(s), or leaves the
            # opponent with no Pokémon in play.
            if me.prizes_left <= _prize_value(opp.active.pk) or not opp.bench:
                can_close = True
        posture = "race" if me.prizes_left <= opp.prizes_left else "stall"
        return PrizeInfo(me.prizes_left, opp.prizes_left, can_close, posture)

    def _my_active_dmg(self, ctx):
        """Best damage my Active can deal THIS turn (L0 dynamic estimator:
        display-0 scalers are valued against the live board)."""
        if ctx.attacks:
            return max(self._opt_atk_dmg(ctx, i) for i in ctx.attacks)
        return ctx.me.active.best_ready_dmg if ctx.me.active else 0

    # ----- MAIN: priority ladder (template method, spec §2.1/§8.4) --------- #
    def main_ladder(self):
        return [self.step_lethal, self.step_ability, self.step_evolve,
                self.step_bench, self.step_trainer, self.step_attach,
                self.step_attack, self.step_retreat, self.step_end]

    # L2 playbook support (pipeline v2, P3'): `ladder` lists rule-method NAMES
    # evaluated top-down BEFORE the generic main steps — legacy's proven
    # priority-ladder shape as a first-class primitive. `game_state` gives the
    # rules a per-game scratch dict (lock flags, designated bodies...) so lines
    # can span turns; it resets automatically when a new game starts.
    # v2.4: pipeline "line" config is read at L1 (BasePolicy) so support/mover
    # economies work under ANY archetype pilot — the focus doctrine (ConfigL2)
    # is optional, not a prerequisite (C-group decks: focus harms, movers help).
    ladder = ()

    def _read_line(self, profile):
        ln = (profile or {}).get("line") or {}
        self._support_e = {int(k): v for k, v in (ln.get("support_energy") or {}).items()}
        self._support_et = {int(k): v for k, v in (ln.get("support_etype") or {}).items()}
        self._predamage = bool(ln.get("predamage"))
        self._support_gate = ln.get("support_gate", "strict")
        self._combo = ln.get("combo") or {}

    def _support_attach(self, ctx):
        """Feed a mover its (type-correct) energy — but NEVER while an attacker
        is starving (v2.4 gate, from the starmie regression): require a ready
        non-mover attacker AND no non-mover exactly one energy short."""
        if not getattr(self, "_support_e", None) or ctx.state.energyAttached:
            return None
        ready = False
        for v in ctx.me.inplay():
            if v.id in self._support_e:
                continue
            if v.ready:
                ready = True
            if self._support_gate == "strict":
                need = _cheapest_cost(v.card)
                if need and v.energy_count == need - 1:
                    return None                    # attacker one short: no divert
        if not ready:
            return None

        def _tcount(v):
            es = getattr(v.pk, "energies", None) or []
            et = self._support_et.get(v.id)
            return sum(1 for e in es if e == et) if et is not None else len(es)
        needy = {v.id for v in ctx.me.inplay()
                 if v.id in self._support_e and _tcount(v) < self._support_e[v.id]}
        if not needy:
            return None
        for i in (self._energy_attach_opts(ctx) or []):
            pk = ctx.field_pk(ctx.sel.option[i])
            if pk is None or pk.id not in needy:
                continue
            et = self._support_et.get(pk.id)
            if et is not None:
                c = ctx.hand_card(ctx.sel.option[i])
                if c is None or c.energyType != et:
                    continue
            return [i]
        return None

    def game_state(self, ctx):
        gs = getattr(self, "_gs", None)
        if gs is None or (ctx.state.turn or 0) < gs.get("_turn", -1):
            gs = self._gs = {}
        gs["_turn"] = ctx.state.turn or 0
        return gs

    def choose_main(self, ctx):
        # loop guard: after many actions this turn, stop developing and just close
        if (ctx.state.turnActionCount or 0) >= 40:
            if ctx.attacks:
                return [self._best_attack_opt(ctx)]
            if ctx.end_idx is not None:
                return [ctx.end_idx]
            return [0]
        for name in self.ladder:                 # L2 playbook first
            r = getattr(self, name)(ctx)
            if r is not None:
                return r
        for step in self.main_ladder():
            r = step(ctx)
            if r is not None:
                return r
        return [0]

    def step_lethal(self, ctx):
        """Take a game-winning KO immediately (preempts development)."""
        if ctx.attacks and ctx.prize.can_close and ctx.ko_targets:
            return [self._best_attack_opt(ctx)]
        return None

    def step_ability(self, ctx):
        return self.decide_ability(ctx)

    def step_evolve(self, ctx):
        return self.decide_evolve(ctx)

    def step_bench(self, ctx):
        return self.decide_bench(ctx)

    def step_trainer(self, ctx):
        return self.decide_trainer(ctx)

    def step_attach(self, ctx):
        r = self._support_attach(ctx)
        if r is not None:
            return r
        return self.decide_energy_target(ctx)

    def step_attack(self, ctx):
        return self.decide_attack(ctx)

    def step_retreat(self, ctx):
        return self.decide_retreat(ctx)

    def step_end(self, ctx):
        if ctx.end_idx is not None:
            return [ctx.end_idx]
        if ctx.attacks:
            return [self._best_attack_opt(ctx)]
        return None

    # ----- decisions (judgments 7-15 + evolve/bench helpers) --------------- #
    @staticmethod
    def _ability_self_removes(card):
        """Ability that removes its OWNER from play (Dudunsparce Run Away Draw
        'shuffle this Pokémon ... into your deck', mamoswine-Abra Teleporter...).
        Firing one of these blindly cost a LIVE game: lone active Dudunsparce
        used Run Away Draw -> no Pokémon in play -> instant loss (reason 3)."""
        for s in (card.skills if card else []) or []:
            t = (s.text or "").lower()
            if (("shuffle this pok" in t or "put this pok" in t)
                    and ("into your deck" in t or "into your hand" in t)):
                return True
        return False

    def decide_ability(self, ctx):
        # Use a clearly-beneficial free ability — but NEVER a self-removing one
        # when it is our last body (or the active with an empty bench): that is
        # an instant self-loss. The turnActionCount guard still defends against
        # ability<->MAIN loops.
        if not ctx.abilities:
            return None
        opt = ctx.sel.option
        bodies = len(ctx.me.inplay())
        # DECK-LOW GUARD for abilities (v2.2 layer fix, symmetric with the
        # trainer guard): optional DRAW abilities (Trade / Recon-Directive
        # class) mill the deck every turn from multiple bodies — dragapult was
        # 42% deckout with the trainer-side guard alone. Stop them early.
        deck_thin = ctx.me.deck_count <= max(8, self.deck_low)
        safe = []
        for i in ctx.abilities:
            pk = ctx.field_pk(opt[i])
            if deck_thin and pk is not None:
                c = _CARDS.get(pk.id)
                for sk in (c.skills if c else []) or []:
                    if "draw" in (sk.text or "").lower():
                        break
                else:
                    sk = None
                if sk is not None:
                    continue
            if pk is not None:
                # self-switch UTILITY abilities (Subjugating-Chains class) must
                # never fire blindly — they randomize the front. Ladder rules
                # invoke them deliberately (v2.3 conversion fix).
                c = _CARDS.get(pk.id)
                if c and any(_RE_AB_SWITCH.search(sk.text or "")
                             for sk in (c.skills or [])):
                    continue
                # SELF-DAMAGE abilities (Torrential-Heart class, "put N damage
                # counters on this Pokémon") fire only while the body can absorb
                # 2x the cost — the original mega_feraligatr self-KO catastrophic,
                # generalized (card-fact gate, no deck hints).
                if c is not None:
                    m = next((_RE_SELF_COUNTERS.search(sk.text or "")
                              for sk in (c.skills or [])
                              if _RE_SELF_COUNTERS.search(sk.text or "")), None)
                    if m is not None and pk.hp <= 20 * int(m.group(1)):
                        continue
            if pk is not None and self._ability_self_removes(_CARDS.get(pk.id)):
                is_active = ctx.me.active is not None and pk is ctx.me.active.pk
                if bodies <= 1 or (is_active and not ctx.me.bench):
                    continue                       # suicide guard
            safe.append(i)
        return [safe[0]] if safe else None

    def decide_evolve(self, ctx):
        if not ctx.evolves:
            return None
        cands = [i for i in ctx.evolves
                 if not self._is_spare_evolution(ctx, ctx.sel.option[i])]
        if not cands:
            return None                         # every evolve here only feeds prizes
        best, bi = -1, None
        for i in cands:
            card = ctx.hand_card(ctx.sel.option[i])
            d = _best_dmg(card)
            if d > best:
                best, bi = d, i
        return [bi] if bi is not None else None

    @staticmethod
    def _prize_value(card):
        """Prizes the opponent takes for KO'ing this body."""
        if card is None:
            return 1
        if getattr(card, "megaEx", False):
            return 3
        return 2 if getattr(card, "ex", False) else 1

    def _is_spare_evolution(self, ctx, o):
        """Hook: should decide_evolve REFUSE this evolve as a spare multi-prize body?

        Default OFF. The rule is real but is NOT fleet-safe -- see MegaLucarioL2's
        override for the validated version and the fleet evidence against shipping it
        here. An L2 opts in by overriding this.
        """
        return False

    def decide_bench(self, ctx):
        """Develop a healthy bench: play a Basic Pokémon while below target.

        Ordered by _bench_score, not hand order. This used to return the FIRST Basic it
        found, so which body developed was decided by shuffle luck -- fine when every
        Basic is an interchangeable attacker, wrong when one of them is the engine. On
        the scouted Mega Lucario list the live 1059.7 player has Lunatone (its only draw
        source) down by turn ~1.2; hand-order benching left ours until turn ~6."""
        if len(ctx.me.bench) >= self.bench_target:
            return None
        cands = []
        for i in ctx.plays:
            card = ctx.hand_card(ctx.sel.option[i])
            if card and card.cardType == CardType.POKEMON and card.basic:
                if _SPARE_EX_GUARD and self._is_spare_ex(ctx, card):
                    continue
                cands.append((self._bench_score(card.cardId), i))
        if not cands:
            return None
        if _BENCH_HAND_ORDER:
            return [min(cands, key=lambda c: c[1])[1]]   # legacy: first Basic in hand order
        return [max(cands)[1]]

    def _is_spare_ex(self, ctx, card):
        """A SECOND copy of a multi-prize Basic ex/megaEx we already have in play.

        decide_bench plays any Basic while below bench_target, and a Basic megaEx is a
        Basic -- so a deck holding 4 Mega Zygarde ex / 4 Mega Kangaskhan ex parks its
        spares on the bench, where each is **3 free prizes** for the opponent and also
        eats the slot the engine pieces need. Nothing in the engine prices what WE
        concede (the same gap _my_attacker_score has: it ADDS +100 for our own ex).

        Measured per decision before this guard: mega_zygarde parked **4.13 prizes** of
        spare Megas on a **4.15/5** bench -- the opponent only needs 6 to win, and
        Lunatone/Solrock could not get down (pair online 22%, 235 "bench full" blocks).
        lillies_clefairy 3.92. Fleet-wide, **49 (deck, card) pairs** run >=2 copies of a
        benchable ex/megaEx, crustle's 4x Mega Kangaskhan ex among them.

        Keep the backup when the board is genuinely thin -- a second body beats a perfect
        prize map if a single KO would otherwise end us."""
        if not (card.ex or card.megaEx):
            return False
        if len(ctx.me.inplay()) < 2:
            return False                      # thin board: any body is worth it
        return any(v.id == card.cardId for v in ctx.me.inplay())

    def _is_spare_ex_sub(self, ctx, cid, taken):
        """Same doctrine as _is_spare_ex, for the TO_BENCH SUB-select.

        _is_spare_ex is reached only from decide_bench -- the MAIN-menu "play a Basic
        from hand". The deck-search path (Nest Ball / Poffin: 'put up to N Basics from
        deck onto the bench') lands in choose_sub, which ranked every option and let
        _mk fill to maxCount -- so the guard that prices what WE concede never ran
        there. Measured in the v31 self-play data: TO_BENCH is 18% of every selection
        that offers a decline, and the engine declined 0 times in 9M samples.

        Two differences from the MAIN-menu case, both deliberate:
          * ``taken`` -- within ONE multi-pick the second copy is spare relative to the
            first even though NEITHER is on the board yet, so the projected board must
            include what this selection has already committed to.
          * We paid a card to search, so the FIRST body is never refused here; only the
            redundant copies beyond it are."""
        card = _CARDS.get(cid)
        if not card or not (card.ex or card.megaEx):
            return False
        if len(ctx.me.inplay()) + len(taken) < 2:
            return False                      # thin board: any body is worth it
        return cid in taken or any(v.id == cid for v in ctx.me.inplay())

    def decide_trainer(self, ctx):
        if ctx.state.supporterPlayed and ctx.state.stadiumPlayed:
            pass  # supporters/stadium spent; items still allowed below
        opt = ctx.sel.option

        def named(keys, supporter=None):
            """Best match by per-deck value, NOT the first one in hand order.

            This returned ctx.plays's first hit, so a keyword bucket behaved as a set of
            interchangeable cards and the pick came down to where the card sat in hand.
            _DRAW_SUPPORTERS lumps Lillie's Determination (shuffle hand, DRAW 6) together
            with Hilda -- which is not a draw supporter at all ("search your deck for an
            Evolution Pokemon and an Energy card"). Measured on the three all-Basic decks,
            where Hilda's Evolution clause cannot even resolve and card_roles already
            tiers it `filler`: Hilda played 44x, **36 of them with a real supporter in the
            same hand**. card_roles never reached this decision.

            The keyword bucket stays as the FILTER (which cards are eligible); the
            per-deck tier decides WHICH of them."""
            best, best_s = None, None
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if not c:
                    continue
                if supporter is True and c.cardType != CardType.SUPPORTER:
                    continue
                if not (_has(c.name, keys)
                        or c.cardId in self._bucket_extra.get(id(keys), ())):
                    continue
                s = self._card_need(ctx, c)
                if best_s is None or s > best_s:
                    best, best_s = i, s
            return best

        best_dmg = ctx.my_active_dmg
        # DECK-LOW GUARD (fleet-wide fix, from the P1 triage: deckout was the #1
        # loss cause across ~20 decks — the engine kept firing optional search/draw
        # into a near-empty deck and milled itself out). Below `deck_low`, stop all
        # OPTIONAL card-flow (search items + draw supporters); keep only gust (a KO)
        # and Rare Candy (evolution, deck-neutral). Gust is always allowed below.
        deck_low = ctx.me.deck_count <= self.deck_low
        # Supporters (one per turn)
        if not ctx.state.supporterPlayed:
            gust = named(_GUST)
            draw = named(_DRAW_SUPPORTERS)
            if not deck_low and draw is not None and ctx.me.hand_count <= self.draw_threshold:
                return [draw]
            if gust is not None and best_dmg >= 60 and ctx.opp.inplay():
                return [gust]
            # hand disruption: strip the opponent when their hand is worth stripping.
            # After the thin-hand draw (so we never starve ourselves to disrupt) and
            # after a KO gust, but before topping up an already-adequate hand.
            i = self._disrupt_play(ctx, opt)
            if i is not None:
                return i
            if not deck_low and draw is not None and ctx.me.hand_count <= self.draw_threshold + 1:
                return [draw]
            # energy-search supporter (Crispin class): the ENERGY ENGINE of thin-
            # energy decks — L0 never played these (not in the draw keyword set),
            # starving every attacker (v2.1 layer fix; lillies wall 95%, Crispin
            # 0.0 plays). Fire when the hand holds no basic energy to attach.
            if not deck_low and not any(
                    _CARDS.get(x.id) is not None
                    and _CARDS[x.id].cardType == CardType.BASIC_ENERGY
                    for x in (ctx.me_ps.hand or [])):
                for i in ctx.plays:
                    c = ctx.hand_card(opt[i])
                    if (c is not None and c.cardType == CardType.SUPPORTER
                            and _RE_E_SEARCH.search(_skill_text(c))):
                        return [i]
        # RECOVERY from the discard pile (Night Stretcher / Tarragon / Energy Retrieval /
        # Energy Recycler class). An entire card CATEGORY the engine could not play:
        # decide_trainer only recognised draw / gust / rare-candy / deck-search / energy-
        # accel / stadium, and recovery matches none of them because it does not search
        # the DECK. Measured fleet-wide before this: **Night Stretcher played 1 of 1372
        # offers (0.1%)** across 14 decks -- and it sits in **47 of 60 decks, 114 copies**.
        # Tarragon (mega_zygarde's whole refuel loop) was 0 of 875.
        # Gated on NEED, not just availability: recovery is worthless if the discard holds
        # nothing we want, and it must not out-rank Rare Candy or a KO gust.
        i = self._recover_play(ctx, opt)
        if i is not None:
            return [i]
        # Items: rare candy (evolution, deck-neutral) always; search only when not deck-low
        i = named(_RARE_CANDY)
        if i is not None:
            return [i]
        # FREE PIVOT with a Switch item: only the 4 subclass ladders (FocusL2/ConfigL2/
        # EthanHoohL2/MegaVenusaurL2) ever played one -- the base had no branch, so
        # `Switch` was a dead slot in 11 decks (audit: offered 3665x, played 0). Same
        # class of hole as the recovery/Transceiver fixes. It is NOT a blanket enable:
        # measured JUSTIFIED_PIVOT (active can't attack AND a bench body is ready) at
        # 0.2-0.4/game on metagross/mega_lucario/manectric but 0/game on cynthia_garchomp
        # and slowking (bench never ready when stuck -> a Switch there only swaps in a
        # non-attacker). The gate below is exactly that condition, so it stays silent on
        # the zero-opportunity decks and does not become a Dawn-style no-op churn.
        i = self._switch_pivot(ctx, opt)
        if i is not None:
            return i
        # Item-class rules the engine had no concept of (all per-deck opt-in, all gated so
        # they never churn -- see each helper). Each was a dead slot in the audit.
        if _ITEM_RULES:
            for helper in (self._damage_boost_play, self._mega_heal_play,
                           self._heal_play, self._deck_recover_play,
                           self._energy_denial_play, self._hand_reset_play,
                           self._scoop_save_play):
                i = helper(ctx, opt)
                if i is not None:
                    return i
        if not deck_low:
            i = named(_SEARCH_ITEMS)
            if i is not None:
                return [i]
        # Energy acceleration only while no attack is ready yet
        if not ctx.attacks:
            i = named(_ENERGY_ACCEL)
            if i is not None:
                return [i]
        # Stadium if none in play
        if not ctx.state.stadium:
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c and c.cardType == CardType.STADIUM:
                    return [i]
        # RESIDUAL: surface otherwise-dead cards into training data at ~zero winrate cost.
        # Only reached when every productive rule above declined -- i.e. the engine was
        # about to play NO trainer this turn -- so the opportunity cost of a marginal card
        # is nil. This is the [[dead-card-loop-converged]] tail: cards that DROP winrate
        # when played proactively (they competed with real plays) are winrate-neutral when
        # played only on an otherwise-idle turn. Per-deck opt-in `residual_cards`.
        i = self._residual_play(ctx, opt)
        if i is not None:
            return [i]
        return None

    def _residual_play(self, ctx, opt):
        """Play an opt-in `residual_cards` card ONLY on an idle trainer turn, so it lands
        in self-play data without costing tempo. Safety rails: never below deck_low (don't
        churn a thin deck), never a supporter once the supporter slot is spent, and keep a
        floor of 3 cards in hand so it never empties us. Frequency stays low naturally --
        it only triggers on turns the engine had no productive trainer at all."""
        if not _RESIDUAL:
            return None
        cards = frozenset(self.profile.get("residual_cards") or ())
        if not cards or ctx.me.hand_count < 3 or ctx.me.deck_count <= self.deck_low:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is None or c.cardId not in cards:
                continue
            if c.cardType == CardType.SUPPORTER and ctx.state.supporterPlayed:
                continue
            return i
        return None

    def _scoop_best_save(self, ctx):
        """The in-play body most worth Scoop-Up-Cyclone'ing back to hand: a key attacker
        (engine-tier / ex / Mega) with >= 2 energy invested, damaged past half its HP, so
        we DENY the opponent the KO and recover the energy. If it is the Active, a benched
        body must exist to promote. Returns the PokemonView or None."""
        best = None
        for v in ctx.me.inplay():
            if v is None or v.card is None or v.energy_count < 2:
                continue
            if (v.max_hp - v.hp) * 2 < v.max_hp:               # < 50% damaged
                continue
            tier = self._tier_value(v.id) or 0
            key = (tier >= _TIER_VALUE["engine"] or getattr(v.card, "megaEx", False)
                   or getattr(v.card, "ex", False))
            if not key:
                continue
            if v is ctx.me.active and not ctx.me.bench:         # can't empty the Active
                continue
            if best is None or (v.max_hp - v.hp) > (best.max_hp - best.hp):
                best = v
        return best

    def _scoop_save_play(self, ctx, opt):
        """Play a save card (Scoop Up Cyclone, opt-in `save_cards`) only when a genuine
        save target exists (see _scoop_best_save). The target sub-select is resolved in
        choose_sub."""
        cards = frozenset(self.profile.get("save_cards") or ())
        if not cards or self._scoop_best_save(ctx) is None:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is not None and c.cardId in cards:
                return [i]
        return None

    def _damage_boost_play(self, ctx, opt):
        """Play a this-turn damage booster (Premium Power Pro / Black Belt's Training)
        ONLY when it CONVERTS a KO on the opponent's Active -- exactly the gate
        MegaLucarioL2 proved for its own boosters (a flat 'big attack only' gate killed
        the card). Opt-in `damage_boost_cards` = {cardId: added_damage}. Black Belt's
        Training (+40) applies only vs an {ex}; enforced below."""
        boosts = self.profile.get("damage_boost_cards") or {}
        if not boosts or not ctx.attacks:
            return None
        oa = ctx.opp.active if ctx.opp else None
        if oa is None:
            return None
        d = ctx.my_active_dmg
        if d <= 0 or d >= oa.hp:                 # no attack, or already lethal
            return None
        oc = _CARDS.get(oa.id)
        opp_ex = bool(oc and (getattr(oc, "ex", False) or getattr(oc, "megaEx", False)))
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is None:
                continue
            amt = boosts.get(str(c.cardId)) or boosts.get(c.cardId)
            if not amt:
                continue
            if c.cardId == 1211 and not opp_ex:   # Black Belt's Training: {ex} only
                continue
            if d + int(amt) >= oa.hp:             # the boost turns this into a KO
                return [i]
        return None

    def _mega_heal_play(self, ctx, opt):
        """Full-heal a Mega ex (Wally's Compassion): heal ALL damage + energy to hand.
        Opt-in `mega_heal_cards`. Gate: a Mega ex in play (Active or Bench) is damaged by
        at least `heal_min` (default 80) -- a big heal on a big investment."""
        cards = frozenset(self.profile.get("mega_heal_cards") or ())
        if not cards:
            return None
        thr = int(self.profile.get("heal_min", 80))
        megas = [v for v in ctx.me.inplay()
                 if v is not None and _CARDS.get(v.id) is not None
                 and getattr(_CARDS[v.id], "megaEx", False) and v.max_hp - v.hp >= thr]
        if not megas:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is not None and c.cardId in cards:
                return [i]
        return None

    def _heal_play(self, ctx, opt):
        """Heal the Active (Jumbo Ice Cream class) when it is worth healing.

        Opt-in `heal_cards`. Gate: the Active is damaged by at least `heal_min` (default
        60) and holds >= 3 energy (Jumbo Ice Cream's own condition; a heavily-invested
        attacker is exactly what a heal wants to keep alive). Never heals a fresh body."""
        cards = frozenset(self.profile.get("heal_cards") or ())
        if not cards or ctx.me.active is None:
            return None
        a = ctx.me.active
        if a.max_hp - a.hp < int(self.profile.get("heal_min", 60)) or a.energy_count < 3:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is not None and c.cardId in cards:
                return [i]
        return None

    def _deck_recover_play(self, ctx, opt):
        """Shuffle Pokemon from discard back into the deck (Sacred Ash class).

        Opt-in `deck_recover_cards`. Gate: the deck is running low (<= deck_low + 4) and
        the discard actually holds Pokemon to shuffle back -- an anti-deckout / refuel
        move for decks that recycle bodies, worthless otherwise."""
        cards = frozenset(self.profile.get("deck_recover_cards") or ())
        if not cards or ctx.me.deck_count > self.deck_low + 4:
            return None
        disc = getattr(ctx.me_ps, "discard", None) or []
        if not any(_CARDS.get(getattr(x, "id", None)) is not None
                   and _CARDS[x.id].cardType == CardType.POKEMON for x in disc):
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is not None and c.cardId in cards:
                return [i]
        return None

    def _energy_denial_play(self, ctx, opt):
        """Discard an energy from an opponent's Pokemon (Crushing Hammer class).

        Opt-in `energy_denial_cards`. Gate: the opponent's Active carries >= 2 energy (so
        a coin-flip discard is worth the card) -- targets a charged attacker, not an empty
        body. Denial decks live and die by keeping the opponent off their attack."""
        cards = frozenset(self.profile.get("energy_denial_cards") or ())
        if not cards or ctx.opp is None or ctx.opp.active is None:
            return None
        if ctx.opp.active.energy_count < 2:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is not None and c.cardId in cards:
                return [i]
        return None

    def _hand_reset_play(self, ctx, opt):
        """Both-players-discard-to-N reset (Hand Trimmer class).

        Opt-in `hand_reset_cards`. Gate: opponent's hand is bigger than ours by a clear
        margin AND ours is already <= 5 (so WE discard nothing) -- a net card swing in our
        favour. Symmetric disruption is a trap without the self-loss guard."""
        cards = frozenset(self.profile.get("hand_reset_cards") or ())
        if not cards or ctx.opp is None:
            return None
        if ctx.me.hand_count > 6 or ctx.opp.hand_count <= ctx.me.hand_count + 2:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is not None and c.cardId in cards:
                return [i]
        return None

    def _disrupt_play(self, ctx, opt):
        """Play a hand-disruption supporter (opt-in `disrupt_cards`) when the opponent's
        hand is large enough to be worth stripping. Gate = opp hand_count >= threshold
        (default 5, so we always remove >= 2 cards). Especially strong vs a hand-size
        payoff like Alakazam's Powerful Hand, whose whole plan is a fat hand."""
        if not _HAND_DISRUPT or not self._disrupt_cards:
            return None
        if ctx.opp is None or ctx.opp.hand_count < self._disrupt_min:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is not None and c.cardId in self._disrupt_cards:
                return [i]
        return None

    def _switch_pivot(self, ctx, opt):
        """Play a Switch-class item to escape a stuck active into a ready attacker.

        Gate = the measured JUSTIFIED_PIVOT: the active is in play but cannot attack
        (`not ctx.attacks`), and a benched body IS ready. Swapping a can't-attack active
        for a can-attack bencher buys an attack this turn at the cost of one item, and
        being cost-free it never strands attached energy the way a paid retreat can.
        Returns the play index, or None when no pivot is justified."""
        if not _SWITCH_PIVOT:
            return None
        if ctx.attacks or ctx.me.active is None:
            return None                          # can already attack -> no pivot
        if not any(v.ready for v in ctx.me.bench):
            return None                          # nothing better to switch to
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is not None and self._in_bucket(c, _SWITCH_CARDS):
                return [i]
        return None

    def _recover_play(self, ctx, opt):
        """Play a discard-recovery card, but only when the discard holds what we lack.

        Recovery (Night Stretcher, Tarragon, Energy Retrieval, Energy Recycler, ...) reads
        "put ... from your discard pile into your hand" -- it never touches the deck, which
        is why every keyword bucket missed it. Two needs justify it:
          (1) BODIES -- a win/engine-tier Pokemon is in the discard and we do not already
              hold one in hand or have it in play (a dead attacker is the usual reason a
              deck stalls out);
          (2) FUEL -- an attacker in play cannot pay its cheapest cost and we hold no
              basic energy to attach.
        Without the gate this would fire every turn it is in hand and just churn cards."""
        want_body, want_fuel = self._recover_need(ctx)
        if not (want_body or want_fuel):
            return None
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c is None or not _RE_RECOVER.search(_skill_text(c)):
                continue
            if c.cardType == CardType.SUPPORTER and ctx.state.supporterPlayed:
                continue
            return i
        return None

    def _recover_need(self, ctx):
        """(need a body back, need fuel back)."""
        hand = ctx.me_ps.hand or []
        disc = getattr(ctx.me_ps, "discard", None) or []
        if not disc:
            return False, False
        disc_ids = [getattr(x, "id", None) for x in disc]

        # (1) a key body sitting in the discard, and none in hand
        key = {c for c in set(self.deck) if self._tier_value(c) is not None
               and self._tier_value(c) >= _TIER_VALUE["engine"]}
        key |= set(getattr(self, "primary_ids", set()))
        key = {c for c in key if _CARDS.get(c) and _CARDS[c].cardType == CardType.POKEMON}
        in_hand = {h.id for h in hand}
        in_play = {v.id for v in ctx.me.inplay()}
        want_body = any(c in key and c not in in_hand and c not in in_play for c in disc_ids)

        # (2) an attacker short of its cost and no basic energy in hand to give it
        has_basic_e = any(_CARDS.get(h.id) is not None
                          and _CARDS[h.id].cardType == CardType.BASIC_ENERGY for h in hand)
        short = any(_cheapest_cost(_CARDS.get(v.id)) > len(v.pk.energies)
                    for v in ctx.me.inplay())
        disc_has_e = any(_CARDS.get(x) is not None
                         and _CARDS[x].cardType == CardType.BASIC_ENERGY for x in disc_ids)
        want_fuel = short and not has_basic_e and disc_has_e
        return want_body, want_fuel

    def decide_energy_target(self, ctx):
        """Attach this turn's energy to the best attacker still short of its cost."""
        if not ctx.attaches:
            return None
        opt = ctx.sel.option
        energy_atts = self._energy_attach_opts(ctx)
        tool_atts = self._tool_attach_opts(ctx)

        def score(i):
            v = self._target_view(ctx, opt[i])
            s = _attacker_score(v)
            if s < 0:
                return -10                          # non-attacker: last resort
            if v.energy_count < _cheapest_cost(v.card):
                s += 150                            # nudge toward reaching an attack
            s += self._type_fit(ctx, opt[i], v)     # TYPE-aware: this energy unlocks an attack
            return s

        if energy_atts and not ctx.state.energyAttached:
            return [max(energy_atts, key=score)]
        if tool_atts:
            pick = max(tool_atts, key=score)
            if score(pick) > -10:                   # only attach a tool to an attacker
                return [pick]
        return None

    def decide_attack(self, ctx):
        """Attack if it does something meaningful; prefer a KO of the Active."""
        # NOTE: a fleet-wide "energy-discard discipline" (redirect away from a
        # self-energy-nuking KO to a cheaper KO) was tried and REVERTED — the panel
        # triage showed net-negative fleet effect (e.g. volcanion 0.55->0.31): the
        # nominal-scaling KO detection misfired and redirected decks off their real
        # attack. This belongs in per-deck L2 (e.g. cynthia_garchomp's Draconic gate),
        # not the generic floor.
        if not ctx.attacks:
            return None
        best = self._best_attack_opt(ctx)
        dmg = self._opt_atk_dmg(ctx, best)
        opp_active = ctx.opp.active
        if opp_active is not None and dmg >= opp_active.hp:
            return [best]                           # KO
        if dmg >= self.attack_min_dmg:
            return [best]
        # no real hit: a self-switch attack (Trading Places / Strafe class) is a
        # PIVOT — use it if a benched body is ready to fight (v2.1 layer fix:
        # walls used to just pass while an armed attacker sat on the bench).
        if any(v.ready for v in ctx.me.bench):
            for i in ctx.attacks:
                at = _ATTACKS.get(ctx.sel.option[i].attackId)
                if at is not None and "switch this pok" in (at.text or "").lower():
                    return [i]
        return None

    def decide_retreat(self, ctx):
        """Ping-pong-safe retreat (spec §2.2): only leave a genuinely stuck active
        for a strictly better one, and never strand a loading attacker's energy."""
        if ctx.retreat_idx is None or ctx.state.retreated:
            return None
        active = ctx.me.active
        if active is None or ctx.attacks:
            return None                             # can attack -> never retreat
        # provably-FREE pivot (v2.1 layer fix): the active cannot attack, a
        # bench body is READY, and retreating discards nothing (printed rc0, or
        # zero energy attached while the option exists = cost already 0 via a
        # stadium/ability). Retreating never strands attached energy.
        if any(v.ready for v in ctx.me.bench):
            rc0 = active.card is not None and (active.card.retreatCost or 0) == 0
            if rc0 or active.energy_count == 0:
                return [ctx.retreat_idx]
        if active.loaded and active.id not in self.type_dead_ids:
            return None                             # genuinely loading attacker: keep
        # (a type-dead "attacker" — wrong-type energy, can never attack — is a wall,
        #  not a loading attacker; it MAY be swapped out for a real attacker)
        better = any((v.ready or v.best_potential_dmg > 0)
                     and v.best_potential_dmg >= active.best_potential_dmg
                     for v in ctx.me.bench)
        return [ctx.retreat_idx] if better else None

    def _opt_pk_id(self, ctx, o):
        """Resolve which Pokémon an option refers to. CRITICAL (fleet audit,
        pipeline v2.1): promotion options (SETUP_ACTIVE / TO_ACTIVE / own
        SWITCH) carry cardId=None — they point at a field slot or a HAND index
        — so ranking by o.cardId alone made EVERY promotion arbitrary
        (option[0]): type-dead Zekrom/Bulbasaur/Latias walls fleet-wide."""
        pk = ctx.field_pk(o)
        if pk is not None:
            return pk.id
        if o.cardId is not None:
            return o.cardId
        h = ctx.me_ps.hand
        if h is not None and o.index is not None and 0 <= o.index < len(h):
            return h[o.index].id
        return None

    def decide_active(self, ctx, mode="setup"):
        """Choose which Pokémon to promote (setup / KO-replacement). Best
        attacker by resolved id; field bodies that can ACT NOW rank above raw
        display (anti-passivity: an armed 170 beats an empty 250 wall)."""
        opt = ctx.sel.option

        def score(i):
            o = opt[i]
            cid = self._opt_pk_id(ctx, o)
            s = self._setup_score(cid)
            pk = ctx.field_pk(o)
            if pk is not None:
                v = PokemonView(pk, self.roles.get(pk.id))
                if v.ready:
                    s += 150                        # can attack immediately
                else:
                    s += 25 * v.energy_count        # part-charged beats empty
                if cid in self.type_dead_ids:
                    s -= 200
            return s
        return sorted(range(len(opt)), key=score, reverse=True)

    def decide_target(self, ctx, kind):
        """Unified target selection (spec judgment 13)."""
        opt = ctx.sel.option
        if kind == "gust":
            dmg = ctx.my_active_dmg

            def gsc(i):
                pk = ctx.opp_pokemon_at(opt[i])
                if pk is None:
                    return (-1, 0)
                koable = 1 if (dmg > 0 and pk.hp <= dmg) else 0
                return (koable, _target_score(pk))
            return sorted(range(len(opt)), key=gsc, reverse=True)
        if kind == "energy_strip":
            def esc(i):
                pk = ctx.opp_pokemon_at(opt[i])
                if pk is None:
                    return (-1, 0, 0)
                is_active = 1 if opt[i].area == AreaType.ACTIVE else 0
                return (is_active, len(pk.energies), _target_score(pk))
            return sorted(range(len(opt)), key=esc, reverse=True)

        # v2.4 predamage-enabler: when a deck attack gains a bonus vs a DAMAGED
        # opponent active (Mortal-Crunch class) and the active is still clean,
        # counter-placement effects chip the ACTIVE first to unlock it.
        if (kind == "spread" and getattr(self, "_predamage", False)
                and not getattr(self, "_combo", None)):
            oa = ctx.opp.active
            if oa is not None and oa.hp == oa.max_hp:
                for i in range(len(opt)):
                    o = opt[i]
                    pk = ctx.opp_pokemon_at(o)
                    if pk is not None and o.area == AreaType.ACTIVE:
                        return [i] + [j for j in range(len(opt)) if j != i]
        # attack / spread / effect: highest-value opponent, but spread onto the
        # lowest-HP (closest to KO). My-side options -> our best attacker.
        def sc(i):
            o = opt[i]
            pk = ctx.opp_pokemon_at(o)
            if o.playerIndex is not None and o.playerIndex == ctx.mi and pk is not None:
                return _attacker_score(PokemonView(pk, self.roles.get(pk.id)))
            if pk is None:
                return -1
            base = _target_score(pk)
            if kind == "spread":
                base += (10000 - pk.hp)             # bias toward finishing something
            return base
        return sorted(range(len(opt)), key=sc, reverse=True)

    def _opt_card_id(self, ctx, o):
        """Resolve the card an option refers to, INCLUDING deck-search options.

        FLEET-WIDE BUG this fixes: a deck search (Ultra Ball / Dusk Ball / Poke Pad /
        Buddy-Buddy Poffin / Fighting Gong / Nest Ball ...) presents its options as
        ``area=DECK, index=<i>, cardId=None`` -- the card itself lives at
        ``sel.deck[i]``. Nothing in the engine ever read sel.deck, so every such option
        fell through to ``_card_usefulness(_CARDS.get(None))`` = 0, every option tied,
        and the sort returned them in arbitrary order: **every deck search in the fleet
        was picking a card blind.** Caught while asking why the scouted Mega Lucario list
        never fetched Lunatone (its whole draw engine) despite running 4 Dusk Ball +
        4 Fighting Gong + 4 Poke Pad -- the search menu offered Lunatone 0 times out of
        223 because we could not see what was in it."""
        if o.cardId is not None:
            return o.cardId
        if _SEARCH_BLIND:                       # A/B switch: reproduce the old blind pick
            return None
        idx = o.index
        if idx is None:
            return None
        if o.area == AreaType.DECK:
            lst = getattr(ctx.sel, "deck", None) or []
        elif o.area == AreaType.LOOKING:
            # Card effects OPEN hidden zones (Pokegear = top 7, Drakloak = top 2,
            # Snorunt reveals a card from the opponent's HAND). That lives only in
            # current.looking, and sel.deck is EMPTY on those menus -- so the deck fix
            # never covered them. Audited: 333 such options in 48 games, all blind.
            lst = getattr(ctx.state, "looking", None) or []
        elif o.area in (AreaType.HAND, AreaType.DISCARD):
            if o.area == AreaType.HAND and not _RESOLVE_HAND:
                return None
            # "Discard 2 cards from your hand" (Carmine/Xerosic class) was choosing at
            # RANDOM: 1010 blind HAND options in 48 games, mostly in DISCARD contexts.
            pi = o.playerIndex if o.playerIndex is not None else ctx.mi
            key = "hand" if o.area == AreaType.HAND else "discard"
            try:
                lst = getattr(ctx.state.players[pi], key, None) or []
            except (IndexError, TypeError):
                return None
        else:
            # PRIZE is [null]*n for both seats -- face-down by the rules. Nothing to
            # resolve, and every prize slot SHOULD tie.
            return None
        if idx < len(lst) and lst[idx]:
            return getattr(lst[idx], "id", None)
        return None

    def _in_bucket(self, card, keys):
        """Bucket membership by NAME or by this deck's cardId opt-in.

        The keyword buckets are name lists, so a card whose name does not describe its
        function is invisible to every rule that gates on one. `_bucket_extra` is the
        per-deck escape hatch; this makes it reach the rules too, not just
        decide_trainer's `named()`. Measured hole: Team Rocket's Giovanni (switch your
        Active TR Pokemon with a benched one -- a FREE retreat for a 3-retreat Mewtwo ex)
        was offered 161 times and played 0, because `_has(name, _SWITCH_CARDS)` cannot
        match "Team Rocket's Giovanni"."""
        if card is None:
            return False
        return (_has(card.name, keys)
                or card.cardId in self._bucket_extra.get(id(keys), ()))

    def _tier_value(self, cid):
        """Explicit per-deck tier from tuning.json's card_roles, or None if unclassified."""
        if not _USE_ROLES or cid is None:
            return None
        t = self.card_roles.get(cid)
        return _TIER_VALUE.get(t) if t else None

    def _card_need(self, ctx, card):
        """What is this card worth TO US RIGHT NOW (search / keep decisions).

        _card_usefulness alone is a fixed hierarchy -- Pokemon(50+) > Energy(40) >
        Supporter(35) > Item(25) -- which ignores what the board is actually short of.
        That only became visible once deck searches could see the deck at all (see
        _opt_card_id): hydrapple then took Pokemon over Energy every time and dropped
        13pt, because Teal Mask Ogerpon ex / Hydrapple ex read "30 more damage for each
        Energy attached", i.e. energy IS their damage. Blind-random had been accidentally
        kinder to it. Adjust the base value by three board signals."""
        if not card:
            return _card_usefulness(card)
        tv = self._tier_value(card.cardId)
        # An explicit per-deck tier REPLACES the global guess (the type value survives
        # only as an intra-tier tiebreak); the board signals below still apply on top.
        s = (tv + _card_usefulness(card) // 10) if tv is not None else _card_usefulness(card)
        field = ([ctx.me.active] if ctx.me.active else []) + list(ctx.me.bench)
        if card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            # (1) an attacker that cannot pay for its attack yet -> fuel is the bottleneck
            short = sum(1 for v in field
                        if _cheapest_cost(_CARDS.get(v.id)) > len(v.pk.energies))
            if short:
                s += 30 + 10 * min(short, 3)
            # (2) an attacker whose damage SCALES with energy never stops wanting it
            for v in field:
                c = _CARDS.get(v.id)
                if not c:
                    continue
                for a in (c.attacks or []):
                    at = _ATTACKS.get(a)
                    txt = (at.text or "").lower() if at else ""
                    if "for each" in txt and "energy" in txt:
                        s += 40
                        break
                else:
                    continue
                break
        elif card.cardType == CardType.POKEMON and card.basic:
            # (3) a thin board loses to a single KO -- a body beats a spell
            if len(field) <= 2:
                s += 40
            # (4) bench EMPTY: the next KO ends the game, and only a BASIC can be benched.
            # +40 was not enough -- a "win"-tier Stage 2 outranked it, so the marnie engine
            # tutored Grimmsnarl ex off Spikemuth Gym into a lost bench-out (user-reported,
            # 2026-08-17). This is an absolute constraint, not a preference.
            if len(field) <= 1:
                s += 500
        elif card.cardType == CardType.POKEMON and not card.basic:
            # The mirror of (4): on a thin board an evolution whose pre-evolution is
            # NEITHER in play NOR in hand cannot become a body in time -- it loses to
            # the tier table (win 900 vs line 600+40) exactly when the game is on the
            # line. -350 flips that ordering; the full -500 applies once the bench is
            # empty and only the lone Active could carry the evolve.
            hand_ids = {getattr(h, "id", None) for h in (getattr(ctx.me_ps, "hand", None) or [])}
            pre_near = any(
                card.evolvesFrom and c2 is not None and card.evolvesFrom == c2.name
                for c2 in (_CARDS.get(x) for x in
                           [v.id for v in field] + [h for h in hand_ids if h]))
            if not pre_near:
                if len(field) <= 1:
                    s -= 500
                elif len(field) <= 2:
                    s -= 350
        return s

    def decide_acquire(self, ctx):
        """Search / draw / recover: most useful cards; route field energy attach
        targets to our best attacker."""
        opt = ctx.sel.option

        def score(i):
            o = opt[i]
            # energy-attach TARGET that is my field Pokémon (cardId=None)
            if (o.cardId is None and o.playerIndex == ctx.mi
                    and o.area in (AreaType.ACTIVE, AreaType.BENCH)):
                pk = ctx.field_pk(o)
                if pk is not None:
                    return 10000 + _attacker_score(PokemonView(pk, self.roles.get(pk.id)))
            # recover-from-discard slot (cardId=None)
            if o.cardId is None and o.area == AreaType.DISCARD and o.index is not None:
                disc = getattr(ctx.me_ps, "discard", None) or []
                if o.index < len(disc):
                    c = _CARDS.get(getattr(disc[o.index], "id", None))
                    if c is not None:
                        if c.cardType == CardType.POKEMON:
                            return 5000 + (c.hp or 0)
                        return 1000
            return self._card_need(ctx, _CARDS.get(self._opt_card_id(ctx, o)))
        return sorted(range(len(opt)), key=score, reverse=True)

    def decide_discard(self, ctx):
        """Discard / return-to-deck: least useful first.

        Read the raw ``o.cardId`` and every option here was blind: a discard menu names
        cards by HAND/DISCARD *reference* (cardId=None, the card lives at
        ``players[pi].hand[index]``), so this keyed every option to
        ``_card_usefulness(None)`` = 0 -- one flat tie, arbitrary pick. Measured before
        the fix: 638/638 options blind across 119 menus, 92% of them resolvable, and 74%
        of the menus held a genuine choice. **What we throw away was being chosen at
        random**, including the deck's win condition.

        Resolving alone is NOT enough, and shipping it alone is a known regression: doing
        exactly that on the legacy engine cost -2.3/-3.8/-4.3pt across three decks, because
        faithfully following a WRONG ranking is worse than ignoring it (blind-random at
        least discarded the win condition only sometimes). Hence _card_need, whose per-deck
        card_roles say what this deck actually wants to keep."""
        opt = ctx.sel.option
        return sorted(range(len(opt)),
                      key=lambda i: self._card_need(ctx, _CARDS.get(self._opt_card_id(ctx, opt[i]))))

    # ----- SUB-selection dispatch (reuses the deciders, spec §8.4) --------- #
    def choose_sub(self, ctx):
        sel = ctx.sel
        c = sel.context
        opt = sel.option

        # YES / NO
        if sel.type == SelectType.YES_NO or all(
                o.type in (OptionType.YES, OptionType.NO) for o in opt):
            yes = next((i for i, o in enumerate(opt) if o.type == OptionType.YES), None)
            no = next((i for i, o in enumerate(opt) if o.type == OptionType.NO), None)
            if c == SelectContext.MULLIGAN:
                return [no if no is not None else 0]     # keep hand
            if c == SelectContext.IS_FIRST:
                return [yes if yes is not None else 0]   # go first
            return [yes if yes is not None else 0]       # default: activate / heads

        # setup / promote active
        if c in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE):
            return self.decide_active(ctx, mode="setup")
        if c in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH,
                 SelectContext.TO_FIELD):
            ranked = sorted(range(len(opt)),
                            key=lambda i: -self._bench_score(self._opt_pk_id(ctx, opt[i])))
            # Returning FEWER than maxCount is how the engine declines: _mk truncates at
            # maxCount but only pads up to minCount, so a short list stands as-is.
            # SETUP_BENCH_POKEMON is excluded from the GLOBAL guard (at setup a body is a
            # body and nothing has been conceded yet) but included for a policy that set
            # _bench_sub_guard: a dedicated-matchup diet must hold from the first bench.
            if (c == SelectContext.TO_BENCH and _SPARE_EX_BENCH_SUB) \
                    or ((c in (SelectContext.TO_BENCH, SelectContext.SETUP_BENCH_POKEMON))
                        and getattr(self, "_bench_sub_guard", False)):
                keep, taken = [], []
                for i in ranked:
                    cid = self._opt_pk_id(ctx, opt[i])
                    if self._is_spare_ex_sub(ctx, cid, taken):
                        continue
                    keep.append(i); taken.append(cid)
                # Decline only down to the legal floor; below it defer to the full
                # ranking so _mk pads with the BEST body, not with option[0].
                if len(keep) >= (sel.minCount or 0):
                    return keep
            return ranked

        # gust (pull opponent's benched Pokémon active)
        if c == SelectContext.SWITCH and any(
                o.playerIndex is not None and o.playerIndex != ctx.mi for o in opt):
            return self.decide_target(ctx, "gust")

        # opponent-targeting damage / effect
        if c in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
                 SelectContext.DAMAGE_COUNTER_ANY):
            return self.decide_target(ctx, "spread")
        if c == SelectContext.EFFECT_TARGET:
            return self.decide_target(ctx, "effect")

        # enemy energy removal
        if c == SelectContext.DISCARD_ENERGY and any(
                o.playerIndex is not None and o.playerIndex != ctx.mi for o in opt):
            return self.decide_target(ctx, "energy_strip")

        # Scoop Up Cyclone target: a TO_HAND select whose options are all OUR OWN in-play
        # Pokemon (field_pk resolves) is "pick one of your Pokemon up", NOT an acquire from
        # deck/discard. decide_acquire would pick by gain-value (backwards). Pick the body
        # _scoop_best_save chose -- the damaged, energy-invested key attacker we are saving.
        if c == SelectContext.TO_HAND and self.profile.get("save_cards"):
            views = [(i, ctx.field_pk(opt[i])) for i in range(len(opt))]
            if views and all(pk is not None and getattr(o, "playerIndex", ctx.mi) in (ctx.mi, None)
                             for (i, pk), o in zip(views, opt)):
                save = self._scoop_best_save(ctx)
                if save is not None:
                    for i, pk in views:
                        if pk is not None and pk.id == save.id \
                                and len(pk.energies or []) == save.energy_count:
                            return [i]
                    # fall back to the most-damaged of the offered bodies
                    return [max(range(len(opt)),
                                key=lambda i: (views[i][1].maxHp - views[i][1].hp)
                                if views[i][1] else -1)]

        # acquisition (search / draw / attach-from / recover)
        if c in (SelectContext.TO_HAND, SelectContext.LOOK, SelectContext.EVOLVES_FROM,
                 SelectContext.EVOLVES_TO, SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO):
            return self.decide_acquire(ctx)

        # Academy at Night stack: "put a card from your hand ON TOP" (TO_DECK from HAND).
        # decide_discard would put our WORST card on top; the Seek combo wants a PAYLOAD
        # there so Slowking copies its attack next. Pick a payload from the offered hand.
        if c == SelectContext.TO_DECK and self.profile.get("seek_payloads"):
            pay = set(self.profile["seek_payloads"])
            for i in range(len(opt)):
                hc = ctx.hand_card(opt[i])
                if hc is not None and hc.cardId in pay:
                    return [i]

        # discard / return-to-deck (DISCARD_CARD_OR_ATTACHED_CARD: same semantics;
        # never observed in the current pool — routed pre-emptively, tier-2)
        if c in (SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM,
                 SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD,
                 SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
            return self.decide_discard(ctx)

        # attack CHOICE sub-select (SelectContext.ATTACK = 35): e.g. copy attacks
        # (N's Zoroark Night Joker) open a menu of the copyable attacks. The old
        # fallback picked option[0] — for ns_zoroark that was Powerful Rage
        # (display 0) instead of Virtuous Flame 170, i.e. a near-zero attack every
        # turn. Pick the highest-value attack; L2 overrides for doctrine.
        if c == SelectContext.ATTACK:
            return self.decide_attack_choice(ctx)

        # ---- fleet-audit fixes (2782 fallback hits / ~470 games) --------------
        # own-side self-switch destination (Switch/Surfer/attack effects; hit in
        # 45 decks): the fallback promoted bench[0] — an arbitrary wall. Use the
        # same best-attacker ordering as promotion. (Opp-side = gust, handled above.)
        if c == SelectContext.SWITCH:
            return self.decide_active(ctx, mode="switch")

        # heal / move-damage-counters SOURCE (Munkidori Adrena-Brain, Dwebble...):
        # pick our MOST-DAMAGED body, not option[0] (= always the active).
        if c in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
            def _dmg_taken(i):
                pk = ctx.field_pk(opt[i])
                if pk is None:
                    return -1
                return (pk.maxHp or 0) - (pk.hp or 0)
            return sorted(range(len(opt)), key=_dmg_taken, reverse=True)

        # evolve-target sub-select (17 decks): prefer the active, then the most-
        # energized bench body (the committed attackers), not option[0].
        if c == SelectContext.EVOLVE:
            def _evo(i):
                pk = ctx.field_pk(opt[i])
                if pk is None:
                    return -1
                o = opt[i]
                area = o.inPlayArea if o.inPlayArea is not None else o.area
                return (100 if area == AreaType.ACTIVE else 0) + len(pk.energies or [])
            return sorted(range(len(opt)), key=_evo, reverse=True)

        # energy-move SOURCE (Solar Transfer / Happy Switch / Backfire...): the
        # fallback ping-ponged active.energy[0] <-> bench[0] (~160 wasted fires a
        # game for mega_venusaur). Take energy from the LEAST valuable holder
        # (walls), so moves converge on the attacker.
        if c in (SelectContext.SWITCH_ENERGY, SelectContext.SWITCH_ENERGY_CARD,
                 SelectContext.TO_HAND_ENERGY, SelectContext.DETACH_FROM,
                 SelectContext.TO_DECK_ENERGY):
            def _src(i):
                pk = ctx.field_pk(opt[i])
                if pk is None:
                    return 0
                return -_attacker_score(PokemonView(pk, self.roles.get(pk.id)))
            return sorted(range(len(opt)), key=_src, reverse=True)

        # ---- tier-2 pre-emptive routes (never observed in the current pool;
        # they reuse tested scorers only, no new logic) --------------------------
        # devolve the opponent's STRONGEST target (own-side devolve stays on the
        # tripwire fallback — picking our best attacker there would be harmful).
        if c in (SelectContext.DEVOLVE, SelectContext.MORE_DEVOLVE) and any(
                o.playerIndex is not None and o.playerIndex != ctx.mi for o in opt):
            return self.decide_target(ctx, "effect")

        # sending a card to prizes: give up the LEAST useful card.
        if c == SelectContext.TO_PRIZE:
            return list(reversed(self.decide_acquire(ctx)))

        # attack-disable: disable THEIR biggest attack; if the menu is our own
        # active's attacks (opponent effect, we pick the victim), sacrifice our
        # smallest.
        if c == SelectContext.DISABLE_ATTACK:
            atk = [i for i, o in enumerate(opt) if o.type == OptionType.ATTACK]
            if atk:
                ours = set()
                a = ctx.me.active
                if a is not None and a.card is not None:
                    ours = set(a.card.attacks or [])
                mine = all(opt[i].attackId in ours for i in atk)
                pick = (min if mine else max)(atk, key=lambda i: self._opt_atk_dmg(ctx, i))
                return [pick] + [i for i in range(len(opt)) if i != pick]

        # counts -> decide_count (base: the largest number, usually beneficial)
        if sel.type == SelectType.COUNT:
            return self.decide_count(ctx)

        # ---- tier-1 TRIPWIRE: any select that reaches this line is UNHANDLED and
        # resolves as option[0]. Count it (per policy instance, keyed by context)
        # so P1 telemetry surfaces newly-triggered contexts automatically — the
        # ns_zoroark copy-menu bug class self-reports instead of hiding.
        h = getattr(self, "_fallback_hits", None)
        if h is None:
            h = self._fallback_hits = Counter()
        h[int(c)] += 1
        return list(range(len(opt)))

    def decide_count(self, ctx):
        """COUNT selects: default to the largest (draw counts, discard-for-damage
        maximization). Precision combos override (pipeline v2.3)."""
        opt = ctx.sel.option
        return [max(range(len(opt)), key=lambda i: (opt[i].number or 0))]

    def decide_attack_choice(self, ctx):
        """Pick from an attack-choice menu (copy effects): best real damage."""
        opt = ctx.sel.option
        atk = [i for i, o in enumerate(opt) if o.type == OptionType.ATTACK]
        if not atk:
            return list(range(len(opt)))
        best = max(atk, key=lambda i: self._opt_atk_dmg(ctx, i))
        return [best] + [i for i in range(len(opt)) if i != best]

    # ----- shared utilities ------------------------------------------------ #
    def _opt_atk_dmg(self, ctx, i):
        a = _ATTACKS.get(ctx.sel.option[i].attackId)
        if a is None:
            return 0
        mko = _RE_EXACTKO.search(a.text or "")
        if mko:
            opp = ctx.opp.active
            n = int(mko.group(1))
            if opp is not None and (opp.max_hp - opp.hp) // 10 == n:
                return opp.hp + 50                # condition holds: instant KO
            return a.damage or 0                  # else the attack does nothing extra
        v = _atk_value(a)
        # v2.4: damage-state conditional clauses evaluated on the LIVE board
        t = a.text or ""
        opp = ctx.opp.active
        opp_dmgd = opp is not None and opp.hp < opp.max_hp
        me_a = ctx.me.active
        self_dmgd = me_a is not None and me_a.hp < me_a.max_hp
        m = _RE_OPPDMG_BASE.search(t)
        if m and opp_dmgd:
            return int(m.group(1))                # Huge-Bite class: 260 -> 30
        if _RE_OPPNODMG_NOTHING.search(t) and not opp_dmgd:
            return 0                              # Bared-Fangs class
        m = _RE_OPPDMG_BONUS.search(t)
        if m and opp_dmgd:
            v += int(m.group(1))                  # Mortal-Crunch class 200->400
        m = _RE_SELFDMG_BONUS.search(t)
        if m and self_dmgd:
            v += int(m.group(1))
        m = _RE_SELFNODMG_BONUS.search(t)
        if m and not self_dmgd:
            v += int(m.group(1))
        m = _RE_EACH.search(a.text or "")
        if m and not _RE_UPTO.search(a.text or ""):
            n = _ctx_each_count(ctx, a.text)
            if n is not None:                     # live board beats the static x3
                v = (a.damage or 0) + int(m.group(1)) * n
                sn = _RE_SNIPE.search(a.text or "")
                if sn:
                    v += int(sn.group(1))
        return v

    def _best_attack_opt(self, ctx):
        return max(ctx.attacks, key=lambda i: self._opt_atk_dmg(ctx, i))

    def _setup_score(self, cid):
        """Rank a body for the ACTIVE spot (setup / KO-replacement).

        Damage-ranked by default. `setup_by_role: true` in the profile makes it rank by
        card_roles tier first (damage as the tie-break) for THAT deck only -- for a deck
        whose Basics are "the line" vs "the draw engine", printed damage is the wrong
        question: alakazam's Dunsparce (20) outranks Abra (10), so we open/bench the
        draw engine and the Alakazam line lands a full turn late (Abra on board turn 1.8
        vs the live agents' 0.7, and Alakazam NEVER arrives in 13% of our games vs 0% of
        theirs). Opt-in because it is NOT universally right -- tried and reverted on
        mega_lucario, where it moved the opening 30% -> 31% and risked promoting an
        80 HP Riolu on KO-replacement.
        """
        if cid is None:
            return 0
        base = _best_dmg(_CARDS.get(cid))
        if self._setup_by_role:
            return (self._tier_value(cid) or 0) + base
        return base

    def _bench_score(self, cid):
        """Rank a body for the BENCH -- deliberately a separate hook from _setup_score.

        The two questions are not the same. "Who should stand in front and attack?" is
        about damage; "who do I want developing behind?" is about what the body DOES.
        A support Pokemon whose value is an Ability (e.g. Lunatone's Lunar Cycle draw
        engine) is near-worthless as an Active but is the first thing you want benched,
        and scoring both with one damage-driven function forces a bad compromise: bump
        it and the engine promotes it into the firing line instead."""
        return self._setup_score(cid)

    def _mk(self, indices, sel):
        n = len(sel.option)
        out = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < n and i not in out:
                out.append(i)
            if len(out) >= sel.maxCount:
                break
        i = 0
        while len(out) < sel.minCount and i < n:
            if i not in out:
                out.append(i)
            i += 1
        return out

    # ----- role auto-inference (spec §8.7) --------------------------------- #
    def infer_roles(self, deck):
        roles = {}
        pre = self._pre_evo_ids(deck)
        pot = {}
        for cid in set(deck):
            c = _CARDS.get(cid)
            if c and c.cardType == CardType.POKEMON:
                pot[cid] = max(_best_dmg(c), 0)
        primary = None
        if pot:
            primary = max(pot, key=lambda k: (pot[k], _CARDS[k].megaEx,
                                              _CARDS[k].ex, _CARDS[k].stage2))
        manual_main = set(self.profile.get("main_attackers", ()))
        for cid in set(deck):
            roles[cid] = self._infer_one(cid, pre, primary, manual_main)
        self._apply_profile_roles(roles)               # manual override wins (hybrid)
        mp = self._manual_primary(
            roles, [c for c, r in roles.items() if r.get("role") == "attacker"])
        if mp is not None:                             # a declared primary is THE primary
            self._enforce_primary(roles, mp)
        return roles

    def _enforce_primary(self, roles, primary):
        """Make ``primary`` the single primary attacker; demote all other
        role==attacker cards to backup (handles 0-damage copiers too)."""
        for c, r in roles.items():
            if r.get("role") == "attacker":
                r["tier"] = "primary" if c == primary else "backup"
        if primary in roles:
            roles[primary].setdefault("energy_need", _main_cost(_CARDS.get(primary)))

    def _apply_profile_roles(self, roles):
        """Merge per-deck profile role overrides onto ``roles`` (AUGMENT, not
        replace). JSON keys arrive as strings; normalise to int cardIds."""
        for k, v in (self.profile.get("roles") or {}).items():
            try:
                cid = int(k)
            except (TypeError, ValueError):
                continue
            roles.setdefault(cid, {}).update(v)

    def _manual_primary(self, roles, poke):
        """The deck's explicitly-declared main attacker, if any: a card in
        ``main_attackers`` or one whose profile role sets tier=primary / payoff.
        Returns a single cardId (in ``poke``) or None."""
        manual = set(self.profile.get("main_attackers", ()))
        for k, v in (self.profile.get("roles") or {}).items():
            if v.get("tier") == "primary" or v.get("payoff"):
                try:
                    manual.add(int(k))
                except (TypeError, ValueError):
                    pass
        return next((c for c in poke if c in manual), None)

    # ---- archetype role-vocabulary detectors (used by L1 infer_roles) ------ #
    @staticmethod
    def _is_counter_mover(card):
        """Ability that MOVES/PLACES damage counters — the spread engine
        (Munkidori Adrena-Brain, Dusknoir Cursed Blast, Froslass Freezing Shroud)."""
        for s in card.skills or []:
            t = (s.text or "").lower()
            if "damage counter" in t and any(k in t for k in ("move", "put", "place")):
                return True
        return False

    @staticmethod
    def _is_spreader(card):
        """Attack that deals damage to Benched Pokémon (spread)."""
        for aid in card.attacks or []:
            a = _ATTACKS.get(aid)
            t = (a.text or "").lower() if a else ""
            if "bench" in t and "damage" in t:
                return True
        return False

    @staticmethod
    def _is_lock(card):
        """A card that RESTRICTS the opponent's actions (ability / attack / play /
        retreat lock — Neutralization Zone etc.), i.e. a control lock piece."""
        blob = " ".join(
            [(s.text or "") for s in (card.skills or [])]
            + [(_ATTACKS[a].text or "") for a in (card.attacks or []) if _ATTACKS.get(a)]
        ).lower()
        return any(k in blob for k in (
            "no abilities", "can't use", "cannot use", "can't attack", "cannot attack",
            "can't retreat", "cannot retreat", "can't play", "cannot play",
            "can't be", "cannot be"))

    def _pre_evo_ids(self, deck):
        names = {}
        for cid in set(deck):
            c = _CARDS.get(cid)
            if c:
                names.setdefault(c.name, cid)
        pre = set()
        for cid in set(deck):
            c = _CARDS.get(cid)
            if c and c.evolvesFrom and c.evolvesFrom in names:
                pre.add(names[c.evolvesFrom])
        return pre

    def _infer_one(self, cid, pre, primary, manual_main):
        c = _CARDS.get(cid)
        if not c:
            return {"role": "tech"}
        if c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            return {"role": "energy", "etype": c.energyType}
        if c.cardType == CardType.POKEMON:
            if cid in manual_main or cid == primary:
                return {"role": "attacker", "tier": "primary",
                        "energy_need": _cheapest_cost(c)}
            if self._is_engine_ability(c):
                return {"role": "engine"}
            if (c.hp or 0) >= _HIGH_HP_WALL and self._is_wall(c):
                return {"role": "wall"}
            if _best_dmg(c) > 0:
                if cid in pre and _best_dmg(c) < 60:
                    return {"role": "evolution_piece"}
                return {"role": "attacker", "tier": "backup"}
            if cid in pre:
                return {"role": "evolution_piece"}
            return {"role": "tech"}
        # trainers
        nm = c.name
        if _has(nm, _GUST):
            return {"role": "disruption", "subrole": "gust"}
        if _has(nm, _DRAW_SUPPORTERS):
            return {"role": "draw"}
        if _has(nm, _RARE_CANDY):
            return {"role": "accelerator", "subrole": "evolution"}
        if _has(nm, _SEARCH_ITEMS):
            return {"role": "search"}
        if _has(nm, _RECOVERY):
            return {"role": "recovery"}
        if _has(nm, _ENERGY_ACCEL):
            return {"role": "accelerator"}
        if self._in_bucket(c, _SWITCH_CARDS):
            return {"role": "pivot"}
        return {"role": "tech"}

    @staticmethod
    def _is_engine_ability(card):
        for s in card.skills or []:
            t = (s.text or "").lower()
            if any(k in t for k in ("draw", "search your deck", "attach", "look at the top")):
                return True
        return False

    @staticmethod
    def _is_wall(card):
        for s in card.skills or []:
            t = (s.text or "").lower()
            if any(k in t for k in ("less damage", "reduce", "prevent all", "no damage",
                                    "takes no damage")):
                return True
        for aid in card.attacks or []:
            a = _ATTACKS.get(aid)
            t = (a.text or "").lower() if a else ""
            if "less damage" in t or "prevent" in t:
                return True
        return False


# --------------------------------------------------------------------------- #
# L1 — archetype policies (7 top-level; spec §3, §3.3)                          #
# --------------------------------------------------------------------------- #
class AggroPolicy(BasePolicy):
    """L0 is already a competent develop-and-attack floor; aggro only sharpens
    prize tempo — gust a KO-able BENCHED target to bank a prize a turn early.
    (A first cut that trimmed the bench and chased the 'fastest' attacker
    mis-routed energy to weak cheap attackers and lost to the plain floor.)"""
    archetype = "aggro"
    draw_threshold = 4

    def infer_roles(self, deck):
        # Role vocab: fast_attacker (a cheap, real attacker). Metadata only — L0's
        # energy routing already serves aggro well (a speed-only override that
        # chased the 'fastest' card mis-routed energy and lost, so we don't touch
        # energy here).
        roles = super().infer_roles(deck)
        fast = [c for c, r in roles.items() if r.get("role") == "attacker"
                and _cheapest_cost(_CARDS[c]) <= 2 and _best_dmg(_CARDS[c]) >= 50]
        if fast:
            roles[max(fast, key=lambda c: _best_dmg(_CARDS[c]))]["fast_attacker"] = True
        return roles

    def decide_trainer(self, ctx):
        if not ctx.state.supporterPlayed and ctx.opp.bench and ctx.my_active_dmg > 0:
            if any(v.hp <= ctx.my_active_dmg for v in ctx.opp.bench):   # gust enables a KO
                opt = ctx.sel.option
                for i in ctx.plays:
                    c = ctx.hand_card(opt[i])
                    if c and _has(c.name, _GUST):
                        return [i]
        return super().decide_trainer(ctx)


class BeatdownPolicy(BasePolicy):
    """Midrange core. Its distinguishing metadata is a CORRECT single primary
    (re-picked below, excluding pure-engine pokemon). Energy routing is LEFT TO L0
    (concentrate on the primary): a high-power mirror A/B (160g/deck, n=1280)
    showed the old "2-cap primary then spill to backup" rule SIGNIFICANTLY loses to
    the plain floor (46.6%, z=-2.4) — splitting energy across two attackers is
    weaker than fully loading one, since prizes come from KOs. The 2-cap was a
    marnie-Grimmsnarl-specific idea (belongs in L2), not a general beatdown rule."""
    archetype = "beatdown"

    def infer_roles(self, deck):
        # Role vocab: primary(+energy_cap) / backup. Re-pick the primary as the
        # biggest RELIABLE attacker that is NOT a pure engine (a Dunsparce-style
        # draw pokemon with a modest attack must not be tagged primary — that was
        # the generic infer_roles mis-inference). Enforce a SINGLE primary.
        roles = super().infer_roles(deck)            # base enforced a declared primary
        attackers = [c for c, r in roles.items() if r.get("role") == "attacker"]
        if self._manual_primary(roles, attackers) is not None:
            return roles                             # trust the declared primary
        # No declared primary: re-pick the biggest RELIABLE (non-pure-engine)
        # attacker so a Dunsparce-style draw pokemon is never tagged primary.
        poke = [c for c in attackers if _best_dmg(_CARDS[c]) > 0]

        def pure_engine(cid):
            c = _CARDS[cid]
            return roles[cid].get("role") == "engine" or (
                _best_dmg(c) < 60 and not (c.ex or c.megaEx) and not c.stage2)
        cands = [c for c in poke if not pure_engine(c)] or poke or attackers
        if cands:
            primary = max(cands, key=lambda c: (_best_dmg(_CARDS[c]), _CARDS[c].megaEx,
                                                _CARDS[c].ex, _CARDS[c].stage2))
            self._enforce_primary(roles, primary)
        return roles

    def decide_energy_target(self, ctx):
        # Beatdown distinctiveness (fires often — every attach): once the primary
        # can already fire its BEST attack, STOP over-loading it (wasted energy)
        # and develop a backup toward its attack. Only diverts AFTER the primary
        # is fully loaded (cap = main-attack cost), so it never under-powers the
        # primary — the failure mode of the old cheapest-cost cap.
        if ctx.state.energyAttached:
            return super().decide_energy_target(ctx)
        energy_atts = self._energy_attach_opts(ctx)
        if not energy_atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option

        def score(i):
            v = self._target_view(ctx, opt[i])
            if v is None or _attacker_score(v) < 0:
                return -10
            s = _attacker_score(v)
            if v.id in self.primary_ids:
                if v.energy_count >= self._energy_cap(v):
                    return s * 0.01              # primary fully loaded -> spill
                return 1000 + s                  # load primary to its BEST attack first
            if v.energy_count < _main_cost(v.card):
                return 500 + s                   # then arm a backup toward its attack
            return s * 0.1
        return [max(energy_atts, key=score)]


class RampPolicy(BasePolicy):
    """Accelerate energy and OVERLOAD a single big / over-cost attacker (opposite
    of beatdown's 2-cap). Role vocab: payoff (overload target) / accel (proactive)."""
    archetype = "ramp"

    def infer_roles(self, deck):
        roles = super().infer_roles(deck)            # base enforced a declared primary
        atts = [cid for cid, r in roles.items() if r.get("role") == "attacker"]
        payoff = self._manual_primary(roles, atts)
        if payoff is None and atts:
            payoff = max(atts, key=lambda c: (_best_dmg(_CARDS[c]),
                                              _cheapest_cost(_CARDS[c]), _CARDS[c].megaEx))
        for cid in atts:                             # single payoff = the overload target
            roles[cid]["payoff"] = (cid == payoff)
            if cid == payoff:
                roles[cid]["overload"] = True
        if payoff is not None:
            self._enforce_primary(roles, payoff)     # payoff is also the primary
        for r in roles.values():
            if r.get("role") == "accelerator":
                r["proactive"] = True
        return roles

    def main_ladder(self):
        """Ramp OWNS the ladder so `step_overload` outranks the generic attach.

        Measured 2026-07-20: RampPolicy's whole thesis ("accelerate and OVERLOAD one big
        attacker, the opposite of beatdown's 2-cap") existed only as a role flag. There is
        no ramp `decide_energy_target` -- BeatdownPolicy has the capping one, ramp never
        got its mirror image -- and under the archetype mixin `decide_energy_target`
        resolves to **FocusL2** anyway, which stops feeding a body once it reaches its
        cost. That shadow is exactly why the mixin alone moved ramp +0.51pt over 7,500
        games. A ladder STEP is the one hook FocusL2 does not own (it defines no
        `main_ladder`), so the doctrine goes here."""
        return [self.step_lethal, self.step_ability, self.step_evolve,
                self.step_bench, self.step_trainer, self.step_overload,
                self.step_attach, self.step_attack, self.step_retreat, self.step_end]

    def step_overload(self, ctx):
        """Feed the payoff body PAST its cheapest cost, which the generic attach won't.

        Only when the payoff is already in play and can still use more: its damage scales
        with energy, or its big attack costs more than it currently holds. Otherwise fall
        through so the generic logic keeps developing the board."""
        if ctx.state.energyAttached:
            return None
        opts = self._energy_attach_opts(ctx)
        if not opts:
            return None
        payoff = {cid for cid, r in self.roles.items()
                  if r.get("payoff") or r.get("overload")}
        if not payoff:
            return None
        best = None
        for i in opts:
            v = self._target_view(ctx, ctx.sel.option[i])
            if v is None or v.card is None or v.id not in payoff:
                continue
            c = _CARDS.get(v.id)
            if c is None:
                continue
            want = max((len(a.energies or []) for a in
                        (_ATTACKS.get(x) for x in (c.attacks or [])) if a), default=0)
            scales = any((_ATTACKS.get(x).text or "").lower().count("for each") and
                         "energy" in (_ATTACKS.get(x).text or "").lower()
                         for x in (c.attacks or []) if _ATTACKS.get(x))
            if scales or v.energy_count < want:
                if best is None or v.energy_count > best[1]:
                    best = (i, v.energy_count)
        return [best[0]] if best else None

    def decide_trainer(self, ctx):
        # Proactively accelerate — but only with ITEM accel. Playing a SUPPORTER
        # accel (Crispin/Cyrano) proactively burns the once-per-turn supporter and
        # starves draw; a matchup eval showed the accel-anything version cost ramp
        # ~3% vs the field, so supporters are left to L0's draw/gust logic.
        opt = ctx.sel.option
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c and c.cardType in (CardType.ITEM, CardType.TOOL) and _has(c.name, _ENERGY_ACCEL):
                return [i]
        return super().decide_trainer(ctx)

    # NOTE: energy routing AND attack timing are left to L0. A single-big-attack
    # payoff simply cannot attack until it is loaded, so L0's "load the best
    # attacker, swing when able" IS the ramp plan — a divergence probe showed a
    # ramp-specific policy differs from L0 on <0.5% of decisions, and forcing
    # divergence (payoff-only energy; a "hold the weak attack" patience rule) was
    # neutral-or-harmful in A/B. Ramp's genuine edge over L0 is only the proactive
    # accel play (decide_trainer above); the payoff/overload tags stay as metadata.


class SpreadPolicy(BasePolicy):
    """Damage-placement decks: fire damage-move abilities every turn and aim
    damage to CONVERT KOs (finish the closest-to-dead), enabling multi-prize turns.
    Role vocab: mover (damage-counter ability) / spreader (bench-hitting attack)."""
    archetype = "spread"

    def infer_roles(self, deck):
        roles = super().infer_roles(deck)
        for cid, r in roles.items():
            c = _CARDS.get(cid)
            if c and c.cardType == CardType.POKEMON:
                if self._is_counter_mover(c):
                    r["mover"] = True
                if self._is_spreader(c):
                    r["spreader"] = True
        self._apply_profile_roles(roles)
        return roles

    def decide_ability(self, ctx):
        # prefer firing a damage-counter mover ability (the spread engine) first
        if not ctx.abilities:
            return None
        opt = ctx.sel.option

        def mv(i):
            pk = ctx.field_pk(opt[i])
            r = self.roles.get(pk.id) if pk is not None else None
            return 1 if (r and r.get("mover")) else 0
        return [max(ctx.abilities, key=mv)]

    def decide_target(self, ctx, kind):
        if kind in ("spread", "effect"):
            opt = ctx.sel.option

            def sc(i):
                pk = ctx.opp_pokemon_at(opt[i])
                if pk is None:
                    return (-1, 0)
                koable = 1 if pk.hp <= (ctx.sel.remainDamageCounter or 0) * 10 else 0
                return (koable, 10000 - pk.hp)     # finish something, else lowest HP
            return sorted(range(len(opt)), key=sc, reverse=True)
        return super().decide_target(ctx, kind)


class ToolboxPolicy(BasePolicy):
    """Many interchangeable attackers: develop / load the one best vs the current
    opponent (weakness + attacker quality), rather than a fixed primary.
    Role vocab: attacker_pool[etype,cost] / flex_energy (multi-type accel)."""
    archetype = "toolbox"
    _FLEX = ("Prism Energy", "Rainbow", "Crispin", "Tera Orb", "Energy Search",
             "Energy Switch")

    def infer_roles(self, deck):
        roles = super().infer_roles(deck)
        for cid, r in roles.items():
            c = _CARDS.get(cid)
            if not c:
                continue
            if c.cardType == CardType.POKEMON and r.get("role") == "attacker":
                r["pool"] = True
                r["etype"] = c.energyType
                r["cost"] = _cheapest_cost(c)
            elif _has(c.name, self._FLEX):
                r["flex_energy"] = True
        self._apply_profile_roles(roles)
        return roles

    def _matchup_score(self, ctx, view):
        if view is None:
            return -1
        s = _attacker_score(view)
        if s < 0:
            return -1
        opp = ctx.opp.active
        if opp is not None and view.card and opp.card:
            # hit the opponent's weakness (huge — ~2x damage / a clean OHKO)
            if opp.card.weakness is not None and view.card.energyType == opp.card.weakness:
                s += 300
            # avoid sending in an attacker that is weak to the opponent's type
            if view.card.weakness is not None and view.card.weakness == opp.card.energyType:
                s -= 150
        return s

    def decide_energy_target(self, ctx):
        if ctx.state.energyAttached:
            return None
        energy_atts = self._energy_attach_opts(ctx)
        if not energy_atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option

        def score(i):
            v = self._target_view(ctx, opt[i])
            s = self._matchup_score(ctx, v)
            if s < 0:
                return -10
            if v.energy_count < _cheapest_cost(v.card):
                s += 150
            return s
        return [max(energy_atts, key=score)]

    def decide_active(self, ctx, mode="setup"):
        # _target_view only resolves FIELD slots (ctx.field_pk). A setup promotion points
        # at a HAND index, so v was None and the fallback scored _best_dmg(_CARDS.get(None))
        # = 0 for every option -- one flat tie, arbitrary starter. Measured: 40 of 283
        # promotions fully tied, 35 of them resolvable to distinct Pokemon. _opt_pk_id
        # already resolves field-slot AND hand-index promotions; use it for the fallback.
        opt = ctx.sel.option

        def sc(i):
            v = self._target_view(ctx, opt[i])
            if v is not None:
                return self._matchup_score(ctx, v)
            return _best_dmg(_CARDS.get(self._opt_pk_id(ctx, opt[i])))
        return sorted(range(len(opt)), key=sc, reverse=True)


class ControlPolicy(BasePolicy):
    """Disruption / stall: play denial proactively (before attacking), only
    attack for meaningful chip, and pivot to walls more freely."""
    archetype = "control"
    attack_min_dmg = 30

    _DENIAL = ("Crushing Hammer", "Enhanced Hammer", "Xerosic", "Eri",
               "Team Rocket's Petrel", "Judge", "Iono")

    def infer_roles(self, deck):
        # Role vocab: wall (generic) / lock / denial / recovery (generic) / chip.
        roles = super().infer_roles(deck)
        for cid, r in roles.items():
            c = _CARDS.get(cid)
            if not c:
                continue
            if _has(c.name, self._DENIAL):
                r["denial"] = True
            if self._is_lock(c):
                r["lock"] = True
            if r.get("tier") == "primary" and 0 < _best_dmg(c) < 100:
                r["chip"] = True                    # slow win-con attacker
        self._apply_profile_roles(roles)
        return roles

    def main_ladder(self):
        return [self.step_lethal, self.step_ability, self.step_evolve,
                self.step_disrupt, self.step_bench, self.step_trainer,
                self.step_attach, self.step_attack, self.step_retreat, self.step_end]

    def step_disrupt(self, ctx):
        return self.decide_disrupt(ctx)

    def decide_disrupt(self, ctx):
        opt = ctx.sel.option
        # 1) energy / hand denial (keep the opponent off-balance)
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if not c:
                continue
            if c.cardType == CardType.SUPPORTER and ctx.state.supporterPlayed:
                continue
            if _has(c.name, self._DENIAL):
                return [i]
        # 2) establish a lock (stadium/item) if its effect isn't already up
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c and self.roles.get(c.cardId, {}).get("lock"):
                if c.cardType == CardType.STADIUM and ctx.state.stadium:
                    continue                        # a stadium is already in play
                return [i]
        return None


class ComboPolicy(BasePolicy):
    """Assemble a multi-piece engine, then execute.

    A GENERIC assemble-gate is deferred to L2: it needs a per-deck ``combo_online``
    predicate and CORRECT primary/piece roles. With generic (often mis-inferred)
    roles, withholding attacks starved decks and lost badly to the plain floor
    (alakazam 17%, rockets_mewtwo 27% in A/B). So L1 combo currently inherits the
    L0 floor plus a mild dig (search when the hand is thin). Real combo behaviour
    (``combo_online`` ASSEMBLE→EXECUTE) is implemented per deck in L2 with the
    right pieces declared in the profile.
    """
    archetype = "combo"

    def infer_roles(self, deck):
        # Role vocab: payoff (execute attacker) / enabler (search+draw to complete)
        # / piece (assemble engine — mostly declared per-deck in the profile, since
        # generic detection is unreliable; L2 reads these + a combo_online predicate).
        roles = super().infer_roles(deck)
        atts = [c for c, r in roles.items() if r.get("role") == "attacker"]
        payoff = self._manual_primary(roles, atts)
        if payoff is None and atts:
            payoff = max(atts, key=lambda c: (_best_dmg(_CARDS[c]),
                                              _cheapest_cost(_CARDS[c]), _CARDS[c].megaEx))
        if payoff is not None:
            roles[payoff]["payoff"] = True
            self._enforce_primary(roles, payoff)
        for r in roles.values():
            if r.get("role") in ("search", "draw"):
                r["enabler"] = True
        self._apply_profile_roles(roles)             # re-assert manual piece/payoff
        return roles

    def combo_online(self, ctx):
        for v in ctx.me.inplay():
            if v.id in self.primary_ids and v.loaded:
                return True
        return ctx.my_active_dmg >= 80

    def decide_trainer(self, ctx):
        if ctx.me.hand_count <= self.draw_threshold:      # dig only when thin
            opt = ctx.sel.option
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c and _has(c.name, _SEARCH_ITEMS):
                    return [i]
        return super().decide_trainer(ctx)

    # ---- L2 shared: FAST ASSEMBLY (search the combo pieces first) ---------- #
    # The real L2 value is completing the combo a turn EARLIER, not withholding.
    # L0 grabs the generically "most useful" card on a search; a combo deck must
    # grab ITS specific missing pieces. Each deck defines _is_combo_piece.
    def _is_combo_piece(self, cid):
        return False

    def decide_acquire(self, ctx):
        # The `cardId is not None` guard made this entire doctrine DEAD, and dead exactly
        # where it was meant to fire: a deck SEARCH names its options by reference
        # (area=DECK, index=i, cardId=None -- see _opt_card_id), so `front` was always
        # empty and every combo deck fell back to L0's generic "most useful card".
        # Measured: 0 pieces fronted across 1918 menus; resolved, 4648 across 878 (46%
        # of menus). Affects all 10 archetype=combo decks, alakazam among them.
        # Every _is_combo_piece override is None-safe (set membership / _CARDS.get guard).
        opt = ctx.sel.option
        order = super().decide_acquire(ctx)
        front = [i for i in order if self._is_combo_piece(self._opt_card_id(ctx, opt[i]))]
        if not front:
            return order
        back = [i for i in order if i not in front]
        return front + back        # combo pieces first, L0 order within each group

    # ---- L2 shared: PROTECT the combo from opponent disruption -------------- #
    def decide_active(self, ctx, mode="setup"):
        """When forced to promote a new Active (after a KO), DON'T expose a fragile,
        not-yet-ready payoff/engine — if a READY attacker (that isn't the unready
        payoff) is available, promote that and keep the payoff safe on the bench to
        assemble. Only reorders (still legal); never stalls when nothing else can
        attack (falls back to L0's best-attacker order)."""
        order = super().decide_active(ctx, mode)
        if mode != "promote":
            return order
        opt = ctx.sel.option

        def is_ready_safe(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return False
            v = PokemonView(pk, self.roles.get(pk.id))
            unready_payoff = (v.id in self.primary_ids) and not v.loaded
            return bool(v.ready) and not unready_payoff
        ready_safe = [i for i in order if is_ready_safe(i)]
        if ready_safe:
            return ready_safe + [i for i in order if i not in ready_safe]
        return order

    # ---- L2 shared execution contract (spec docs/combo_l2.md) -------------- #
    def _expected_dmg(self, ctx):
        """Real payoff damage (DB lists 0 for scaling attacks). L2 overrides."""
        return ctx.my_active_dmg

    def _execute_or_build(self, ctx):
        """Attack when able, using the deck's REAL scaling damage for lethal
        detection (L0 under-rates 0-damage scaling attacks like Powerful Hand /
        Syrup Storm and can miss a lethal).

        IMPORTANT (measured): WITHHOLDING attacks to "wait for the combo" is
        CATASTROPHIC in this AI environment — a gate that skipped attacks until
        ``combo_online`` scored **−8.4% (z −4.3) vs the field** (hydrapple −28!).
        The opponents punish passivity, so combo decks must pressure ASAP. Thus
        ``combo_online`` is kept only as completion/opponent-read METADATA and is
        NOT used to skip attacks."""
        if not ctx.attacks:
            return None
        opp = ctx.opp.active
        if opp is not None and self._expected_dmg(ctx) >= opp.hp:
            return [self._best_attack_opt(ctx)]            # lethal via real scaling dmg
        return BasePolicy.decide_attack(self, ctx)         # else attack when meaningful


# --------------------------------------------------------------------------- #
# L2 — per-deck combo policies (spec docs/combo_l2.md)                          #
# --------------------------------------------------------------------------- #
def _active_is(ctx, cid):
    a = ctx.me.active
    return a is not None and a.id == cid


class AlakazamL2(ComboPolicy):
    """Powerful Hand (743) = 20 x hand size to the opponent's Active (counter
    placement — ignores reduction). L2 rules: Boss-nuke gust (pull a PH-KO-able
    bench target), Enhanced Hammer only vs special energy, and the blind-P0
    derived rules below (Psychic-to-line energy routing + chain, Enriching
    type-awareness, evolve-last sequencing). We attack every turn — withholding
    and blanket hand-preservation were both measured harmful."""
    _ALAKAZAM = 743
    _FUEL_LOW = 14                                   # deck-count gate for draw-fuel evolves (anti-deckout)
    _PIECES = frozenset({741, 742, 743, 66, 305})   # Abra/Kadabra/Alakazam + draw engine

    def _is_combo_piece(self, cid):
        return cid in self._PIECES

    def _expected_dmg(self, ctx):
        return 20 * ctx.me.hand_count if _active_is(ctx, self._ALAKAZAM) else 0

    def combo_online(self, ctx):
        return _active_is(ctx, self._ALAKAZAM) and ctx.me.hand_count >= 6

    def _ph_dmg(self, ctx):
        """Real Powerful Hand damage to the Active = 20 x hand size."""
        return 20 * ctx.me.hand_count if _active_is(ctx, self._ALAKAZAM) else ctx.my_active_dmg

    def decide_trainer(self, ctx):
        opt = ctx.sel.option

        def play_id(cid):
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c and c.cardId == cid:
                    return [i]
            return None

        # (3) Boss's Orders targets the NUKE: Powerful Hand hits the Active, and L0
        # doesn't know its real 20xhand damage (so L0 never gusts for Alakazam).
        # Pull a benched opponent that Powerful Hand can KO into the Active.
        if not ctx.state.supporterPlayed and _active_is(ctx, self._ALAKAZAM):
            if self.ko_targets_with(ctx, self._ph_dmg(ctx), bench_only=True):
                for i in ctx.plays:
                    c = ctx.hand_card(opt[i])
                    if c and _has(c.name, _GUST):
                        return [i]
        # (4) Enhanced Hammer ONLY when the opponent actually has a Special Energy to
        # strip (dead vs basic-energy decks; otherwise better held for hand size).
        if self.opp_energy_pokemon(ctx, special_only=True):
            r = play_id(1081)
            if r:
                return r
        # ---- P1-loop round 1 fixes: 4 dead cards + 36% deckout losses ---------
        disc_ids = [c.id for c in (getattr(ctx.me_ps, "discard", None) or [])]
        inplay_ids = {v.id for v in ctx.me.inplay()}
        hand_ids = {c.id for c in (ctx.me_ps.hand or [])}
        # Sacred Ash (1129): anti-deckout — recycle Pokémon into the deck when low.
        if ctx.me.deck_count <= 8 and sum(
                1 for d in disc_ids
                if _CARDS.get(d) and _CARDS[d].cardType == CardType.POKEMON) >= 3:
            r = play_id(1129)
            if r:
                return r
        no_alaka = (self._ALAKAZAM not in inplay_ids
                    and self._ALAKAZAM not in hand_ids)
        if no_alaka:
            # rebuild (items first — no supporter cost): Night Stretcher recovers a
            # discarded Alakazam; Lana's Aid = one-card line rebuild (supporter).
            if self._ALAKAZAM in disc_ids:
                r = play_id(1097)
                if r:
                    return r
                if not ctx.state.supporterPlayed:
                    r = play_id(1184)
                    if r:
                        return r
            # Dawn (1231): possession-based line tutor (fetch Basic+St1+St2 = the
            # whole line). The generic hand<=5 draw gate NEVER fires in this deck
            # (hand is 12-18), so without this the card is dead (0.00 plays/game).
            if not ctx.state.supporterPlayed:
                r = play_id(1231)
                if r:
                    return r
        # DECK-LOW mode (P1-loop rounds 2-3: deckout ~40-55% of losses): the engine
        # keeps DIGGING (Poke Pad/Poffin ~2.6 plays/game each) even with a nearly-
        # empty deck — self-milling into a loss. When the deck is low, stop optional
        # thinning: allow only Rare Candy; skip L0's search/draw entirely. (Round 3
        # raised the threshold 6->10; at 6 the mandatory turn-draws + Psychic Draw
        # evolutions still emptied the deck before the mode could matter.)
        if ctx.me.deck_count <= 10:
            return play_id(1079)
        return super().decide_trainer(ctx)

    def decide_evolve(self, ctx):
        # Chain fix (P1-loop round 2: postKO_attack_rate 0.59): L0 evolves by max
        # listed damage, so Dudunsparce (90) out-prioritises Alakazam (nominal 30)
        # and the 2nd Alakazam never comes online. Evolve the LINE first.
        if not ctx.evolves:
            return None
        opt = ctx.sel.option
        # DECK-LOW fuel gate (anti-self-deckout vs stallers, 2026-07): once an
        # Alakazam is online, evolving a SPARE Kadabra/Alakazam only triggers
        # Psychic Draw (+2/3 forced cards) = self-mill. The deckout is ~50% of the
        # loss vs crustle_stall / cubchoo_control (long games); PH is already ~278
        # overkill, so below the threshold skip those draw-triggering evolves. A
        # deck-COUNT gate (not the rejected hand-size gate) only bites the stall
        # matchups where the game drags the deck low with Alakazam already up.
        ev = ctx.evolves
        if (self._ALAKAZAM in {v.id for v in ctx.me.inplay()}
                and ctx.me.deck_count <= self._FUEL_LOW):
            drawless = [i for i in ctx.evolves
                        if getattr(ctx.hand_card(opt[i]), "cardId", 0)
                        not in (742, self._ALAKAZAM)]
            if not drawless:
                return None                          # only self-milling evolves left
            ev = drawless
        pref = {self._ALAKAZAM: 5, 742: 3, 66: 1}
        best = max(ev,
                   key=lambda i: pref.get(getattr(ctx.hand_card(opt[i]), "cardId", 0), 0))
        if pref.get(getattr(ctx.hand_card(opt[best]), "cardId", 0), 0) > 0:
            return [best]
        return super().decide_evolve(ctx)

    def decide_target(self, ctx, kind):
        # gust target chosen by what Powerful Hand (20xhand) can KO, not the Active's
        # listed damage (which L0 uses and which is 0/nominal for Powerful Hand).
        if kind == "gust" and _active_is(ctx, self._ALAKAZAM):
            ph = self._ph_dmg(ctx)
            opt = ctx.sel.option

            def sc(i):
                pk = ctx.opp_pokemon_at(opt[i])
                if pk is None:
                    return (-1, 0)
                return (1 if pk.hp <= ph else 0, _target_score(pk))
            return sorted(range(len(opt)), key=sc, reverse=True)
        return super().decide_target(ctx, kind)

    # HAND-PRESERVATION was tried and REJECTED (2026-07-11): stopping surplus energy
    # attach (Powerful Hand needs only 1) to keep the hand big for a bigger nuke
    # measured ≤ L0 in every form (full: −4.3%; energy-only: −1.1/−1.8% over 2 runs).
    # Root cause: L0's "keep attaching to the best attacker" was loading a BACKUP
    # attacker; suppressing it leaves nothing ready when Alakazam is KO'd.

    # ---- blind-P0 derived rules (docs/p0_alakazam_blind.json, 2026-07-11) ---- #
    _LINE_PRIO = {743: 30, 742: 20, 741: 10}   # most-evolved line member first
    _DRAW_LINE = frozenset({305, 66})           # Dunsparce -> Dudunsparce: shuffles itself away

    @staticmethod
    def _psy(pk):
        """PSYCHIC-providing energy on a Pokémon. H3 (probe-verified): Enriching
        Energy provides {C} only and can NOT pay Powerful Hand's [P] — count real
        Psychic, not attached cards."""
        return sum(1 for e in (pk.energies or []) if e == EnergyType.PSYCHIC)

    def main_ladder(self):
        # H9: Psychic Draw triggers ON EVOLVE (+2/+3 cards) — evolve LAST, after
        # all hand-spending plays and right before the attack, so the drawn cards
        # are still in hand when Powerful Hand counts them.
        return [self.step_lethal, self.step_ability, self.step_bench,
                self.step_trainer, self.step_attach, self.step_evolve,
                self.step_attack, self.step_retreat, self.step_end]

    def decide_ability(self, ctx):
        # H6 (blind-P0), calibrated by measurement: Run Away Draw (Dudunsparce 66)
        # trades a 140HP body (+its energy) for 3 cards — which is +60 Powerful
        # Hand damage, so it IS this deck's fuel. Block it only when the trade is
        # bad: board thin (<=2 bodies — the live archaludon loss was a 6x churn
        # spiral into a sweep) or the body carries energy (it's the armed backup).
        # A stricter hand<=4 gate was tried and REGRESSED the field score 68->58
        # (starved the nuke), so hand size does NOT gate it.
        if not ctx.abilities:
            return None
        opt = ctx.sel.option
        safe = []
        for i in ctx.abilities:
            pk = ctx.field_pk(opt[i])
            if pk is not None and pk.id == 66:
                if (len(ctx.me.inplay()) <= 2 or (pk.energies or [])
                        or ctx.me.deck_count <= 10):
                    continue
            safe.append(i)
        if not safe:
            return None
        old = ctx.abilities
        ctx.abilities = safe                 # base suicide guard still applies
        try:
            return super().decide_ability(ctx)
        finally:
            ctx.abilities = old

    def decide_energy_target(self, ctx):
        # H8: L0 reads Land Crush 90 > Powerful Hand 0 and feeds Dudunsparce,
        # starving the line. Route PSYCHIC to the most-evolved line member still
        # lacking its 1 {P} (energy carries up through evolution — probe-verified),
        # then pre-power a SECOND Alakazam (chain, cap 2). Enriching (draw 4 on
        # attach = +80 nuke) is dumped on a colorless user, never on the line.
        if ctx.state.energyAttached:
            return super().decide_energy_target(ctx)
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option
        powered_alaka = sum(1 for v in ctx.me.inplay()
                            if v.id == self._ALAKAZAM and self._psy(v.pk) >= 1)

        def score(i):
            pk = ctx.field_pk(opt[i])
            card = ctx.hand_card(opt[i])
            if pk is None or card is None:
                return -10
            enriching = card.cardId == 13
            if pk.id in self._LINE_PRIO:
                if enriching:
                    return 1                          # {C} can't pay [P] (H3)
                if self._psy(pk) < 1:
                    if pk.id == self._ALAKAZAM and powered_alaka >= 2:
                        return 2                      # chain capped at 2
                    return 100 + self._LINE_PRIO[pk.id]
                return 2                              # already has its 1 {P}
            # NEVER feed the Dunsparce line (305 -> 66). Run Away Draw shuffles that body
            # **and all attached cards** back into the deck, so energy put there is simply
            # deleted -- and worse, decide_ability then REFUSES to fire the draw because
            # "the body carries energy (it's the armed backup)". Measured: 576 Run Away
            # Draw offers in 80 games, only 18% allowed, and **52% blocked by exactly this
            # self-inflicted energy** (22% of all our attaches landed on Dunsparce). The
            # deck's whole damage is 20 x hand size and Powerful Hand costs a single {P},
            # so a Land-Crush backup is worth far less than 3 cards (+60 damage) a turn.
            # Scored below the already-powered line (2) so surplus energy stalls in hand
            # rather than disarming our own draw engine.
            if pk.id in self._DRAW_LINE:
                return 1 if enriching else 0
            return 50 if enriching else 20            # Enriching dump = +4 hand
        return [max(atts, key=score)]

    def decide_attack(self, ctx):
        return self._execute_or_build(ctx)


class DoubladeL2(ComboPolicy):
    """Weaponized Swords (1066) = 60 x steel-line cards revealed from hand."""
    _DOUBLADE = 1066
    _STEEL_LINE = {1065, 1066, 1067}

    def _is_combo_piece(self, cid):
        return cid in self._STEEL_LINE or cid == 547   # steel line (for hand) + Genesect

    def _steel_in_hand(self, ctx):
        hand = ctx.me_ps.hand or []
        return sum(1 for c in hand if c.id in self._STEEL_LINE)

    def _expected_dmg(self, ctx):
        return 60 * self._steel_in_hand(ctx) if _active_is(ctx, self._DOUBLADE) else 0

    def combo_online(self, ctx):
        return _active_is(ctx, self._DOUBLADE) and self._steel_in_hand(ctx) >= 3

    def decide_attack(self, ctx):
        return self._execute_or_build(ctx)

    # blind-P0 catastrophics (docs/p0_doublade.json): Weaponized Swords AMMO is the
    # steel-line cards HELD IN HAND, so L0's discard/shuffle defaults throw the win
    # away. Never discard steel-line ammo (or Rare Candy); don't shuffle a firing
    # hand back into the deck.
    def decide_discard(self, ctx):
        opt = ctx.sel.option

        def cost(i):
            c = self._opt_card_id(ctx, opt[i])
            if c in self._STEEL_LINE or c == 1079:       # ammo / Rare Candy: never
                return _KEEP
            return self._card_need(ctx, _CARDS.get(c)) if c is not None else 0
        return sorted(range(len(opt)), key=cost)

    def decide_trainer(self, ctx):
        # gate Lillie's Determination (1227, shuffle hand->deck) once we already
        # hold a firing hand (>=3 ammo).
        if self._steel_in_hand(ctx) >= 3:
            skip = {i for i in ctx.plays
                    if (c := ctx.hand_card(ctx.sel.option[i])) and c.cardId == 1227}
            if skip:
                old = ctx.plays
                ctx.plays = [i for i in ctx.plays if i not in skip]
                try:
                    return super().decide_trainer(ctx)
                finally:
                    ctx.plays = old
        return super().decide_trainer(ctx)


class HydrappleL2(ComboPolicy):
    """Grass ramp. The PRIMARY attacker is the BASIC Teal Mask Ogerpon ex (96) -- Myriad
    Leaf Shower = 30 + 30 x energy on BOTH Actives -- NOT the Stage-2 Hydrapple ex (150,
    Syrup Storm = 30 + 30 x {G} on Hydrapple), which is the slow late finisher. The current
    competitive list runs Ogerpon x4 vs Hydrapple x2 and wins most turns with Ogerpon; our
    probe confirms Ogerpon attacks ~200/40 games vs Hydrapple ~5-36. The old pilot scored
    ONLY Hydrapple, so its build/execute logic ignored the real win condition."""
    _HYDRAPPLE = 150
    _OGERPON = 96
    _MEGANIUM = 710
    _TAPU = 920

    def _is_combo_piece(self, cid):
        c = _CARDS.get(cid)
        # attackers (Ogerpon/Hydrapple/Tapu Bulu) + Meganium (doubler) + Celebi + basic {G}
        return cid in {150, 96, 920, 710, 655} or (
            c is not None and c.cardType == CardType.BASIC_ENERGY
            and c.energyType == EnergyType.GRASS)

    def decide_active(self, ctx, mode="setup"):
        # Situational attacker choice on PROMOTION (matchup fit; never withholds -- only
        # reorders which body to bring up after a KO). Human levers, from the matchup
        # research: vs a big WALL Ogerpon can't reach, bring up the scaling Hydrapple ex
        # (Syrup Storm 330-660); when BEHIND on prizes, bring up the single-prize Tapu Bulu
        # (Wood Hammer 220) so a return KO costs the opponent 2, not us feeding a 2-prize ex.
        order = super().decide_active(ctx, mode)
        if not _HYDRA_SMART or mode != "promote" or not order:
            return order
        opt = ctx.sel.option
        oa = ctx.opp.active
        behind = ctx.me.prizes_left > ctx.opp.prizes_left
        mega = any(v.id == self._MEGANIUM for v in ctx.me.inplay())

        def prefer(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return 0
            v = PokemonView(pk, self.roles.get(pk.id))
            g = v.energy_count * (2 if mega else 1)
            # wall-break: high-HP active Ogerpon can't KO -> a loaded Hydrapple ex
            if (v.id == self._HYDRAPPLE and oa is not None
                    and oa.hp >= 250 and 30 + 30 * g >= oa.hp):
                return 3
            # prize race: we're behind -> a ready single-prize Tapu Bulu that can KO
            if v.id == self._TAPU and behind and v.ready and oa is not None and 220 >= oa.hp:
                return 2
            return 0
        best = max(range(len(order)), key=lambda k: prefer(order[k]))
        if prefer(order[best]) > 0:
            j = order[best]
            return [j] + [i for i in order if i != j]
        return order

    def _eff_grass(self, ctx):
        g = sum(1 for v in ctx.me.inplay() for e in v.energy if e == EnergyType.GRASS)
        if any(v.id == self._MEGANIUM for v in ctx.me.inplay()):
            g *= 2                                          # Wild Growth: each {G} -> {G}{G}
        return g

    def _expected_dmg(self, ctx):
        if _active_is(ctx, self._HYDRAPPLE):
            return 30 + 30 * self._eff_grass(ctx)           # Syrup Storm: {G} on Hydrapple
        if _active_is(ctx, self._OGERPON):
            mine = ctx.me.active.energy_count if ctx.me.active else 0
            opp = ctx.opp.active.energy_count if (ctx.opp and ctx.opp.active) else 0
            return 30 + 30 * (mine + opp)                   # Myriad Leaf Shower: both Actives
        if _active_is(ctx, self._TAPU):
            return 220                                      # Wood Hammer (single-prize body)
        return 0

    def combo_online(self, ctx):
        # Ogerpon (Basic) is online as soon as it holds a few energy -- do NOT withhold it
        # like the slow Stage-2; Hydrapple wants >=4 {G} for a real Syrup Storm.
        if _active_is(ctx, self._OGERPON):
            return (ctx.me.active.energy_count if ctx.me.active else 0) >= 2
        return _active_is(ctx, self._HYDRAPPLE) and self._eff_grass(ctx) >= 4

    _BRIAR = 1201

    def decide_trainer(self, ctx):
        # BRIAR closer: when the opponent is at EXACTLY 2 prizes and our Tera attacker
        # (Teal Mask Ogerpon ex, the primary) can KO their Active this turn, Briar takes
        # the last 2 prizes in ONE KO = game. The base engine never played it (0/30);
        # it directly answers the deck-out loss vs walls -- close before we mill out.
        if (not ctx.state.supporterPlayed
                and getattr(ctx.opp, "prizes_left", None) == 2
                and _active_is(ctx, self._OGERPON)):
            oa = ctx.opp.active
            if oa is not None and self._expected_dmg(ctx) >= oa.hp:
                for i in ctx.plays:
                    c = ctx.hand_card(ctx.sel.option[i])
                    if c is not None and c.cardId == self._BRIAR:
                        return [i]
        return super().decide_trainer(ctx)

    def decide_attack(self, ctx):
        return self._execute_or_build(ctx)


class RocketsMewtwoL2(ComboPolicy):
    """Erasure Ball (431), gated by Power Saver (>=4 Team Rocket's Pokémon in play).
    The engine won't offer the attack early, so L2 mainly builds the tribe fast."""
    _MEWTWO = 431

    @staticmethod
    def _is_tr(cid):
        c = _CARDS.get(cid)
        return bool(c and "team rocket's" in _norm_name(c.name))

    def _is_combo_piece(self, cid):
        return self._is_tr(cid)                       # any Team Rocket's card unlocks Power Saver

    def _tr_count(self, ctx):
        return sum(1 for v in ctx.me.inplay() if self._is_tr(v.id))

    def combo_online(self, ctx):
        return self._tr_count(ctx) >= 4

    _ARTICUNO = 414

    def decide_bench(self, ctx):
        # Develop Team Rocket's Pokémon first (unlock Power Saver sooner), and put
        # Team Rocket's Articuno (Repelling Veil: prevents attack EFFECTS on our
        # Basic TR Pokémon) down FIRST — it protects the whole assembly from
        # disruption attacks while we build toward 4 TR Pokémon + Erasure Ball.
        if len(ctx.me.bench) < self.bench_target:
            opt = ctx.sel.option
            have_veil = any(v.id == self._ARTICUNO for v in ctx.me.inplay())

            def rank(i):
                c = ctx.hand_card(opt[i])
                if not (c and c.cardType == CardType.POKEMON and c.basic):
                    return -1
                if c.cardId == self._ARTICUNO and not have_veil:
                    return 2                          # protection wall first
                return 1 if self._is_tr(c.cardId) else 0
            cands = [i for i in ctx.plays if rank(i) > 0]
            if cands:
                return [max(cands, key=rank)]
        return super().decide_bench(ctx)

    def decide_attack(self, ctx):
        # Power Saver is engine-enforced (no attack offered until >=4 TR Pokémon),
        # so no withholding is needed; attack when able. Value is the fast tribe
        # development (decide_bench) + lethal awareness.
        return BasePolicy.decide_attack(self, ctx)


class MamoswineL2(ComboPolicy):
    """Rumbling March (283) = 180 + 40 x benched Stage-2. 180 base is fine, so NO
    withholding — just track real damage and bench Stage-2 lines for the bonus."""
    _MAMOSWINE = 283

    def _is_combo_piece(self, cid):
        c = _CARDS.get(cid)                            # Mamoswine line + any Stage-2 (bonus)
        return cid in {281, 282, 283} or (c is not None and c.cardType == CardType.POKEMON and c.stage2)

    def _benched_stage2(self, ctx):
        return sum(1 for v in ctx.me.bench if v.card and v.card.stage2)

    # _expected_dmg override REMOVED (workstream 1, 2026-07-17): it hand-computed
    # `180 + 40 * benched Stage-2` because _ctx_each_count could not read "for each
    # Stage 2 Pokemon on your Bench" and fell back to counting the WHOLE bench. Now that
    # the generic model filters the bench by stage, `ctx.my_active_dmg` computes exactly
    # this. Verified on live boards: generic == the old bespoke value on **every** active
    # this deck ever had (Mamoswine ex 7/7, and Swinub/Piloswine/Torchic/Combusken/Abra
    # 4/4, 6/6, 4/4, 1/1, 5/5), i.e. the override was pure duplication.

    def decide_attack(self, ctx):
        # attack whenever loaded (L0), but use the REAL scaling damage for lethal
        if ctx.attacks:
            opp = ctx.opp.active
            if opp is not None and self._expected_dmg(ctx) >= opp.hp:
                return [self._best_attack_opt(ctx)]
        return BasePolicy.decide_attack(self, ctx)


class _NeverChosenMixin:
    """Shared helpers for the three cards engine_v2 was measured never to play.

    Fleet scan of 3,000,000 v41 pool rows: of 232 distinct cards ever OFFERED as a play,
    exactly two were never chosen (Waitress 5,807 offers, Klinklang 1,697) -- and Briar
    joins them as soon as ogerpon_mono enters the pool (43 offers, 0 plays). A card the
    heuristic never plays can only ever appear in the training data as a NEGATIVE, so the
    LM cannot learn the line from imitation no matter how long it trains.
    """

    def _hand_idx(self, ctx, cid):
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is not None and c.cardId == cid:
                return i
        return None

    def _live_dmg(self, ctx, attack_i):
        """Damage this attack would really land, via lm/hidden. None when unavailable."""
        obs = getattr(self, "_raw_obs", None)
        if not isinstance(obs, dict) or not obs.get("search_begin_input"):
            return None
        try:
            from lm import damage as _dmg, hidden as _hid
            dec = _hid.read(obs)
            if dec is None:
                return None
            cur = obs["current"]; yi = cur["yourIndex"]
            me = (cur["players"][yi].get("active") or [None])[0]
            op = (cur["players"][1 - yi].get("active") or [None])[0]
            if not me or not op:
                return None
            v, _k = _dmg.final_damage(obs, dec, me["serial"], op["serial"],
                                      ctx.sel.option[attack_i].attackId, yi)
            return v
        except Exception:
            return None


class BriarL2(AggroPolicy, _NeverChosenMixin):
    """Play Briar only on the turn it WINS THE GAME.

    Briar: "only if your opponent has exactly 2 Prize cards remaining. During this turn, if
    your opponent's Active is Knocked Out by damage from an attack used by your TERA
    Pokemon, take 1 more Prize card." Competitive lists (NAIC 2026 Hydrapple) run it as a
    finisher: it turns a 2-prize KO into 3, so a player sitting on 3 Prizes wins on the spot.

    A first attempt played it on ANY knockout and measured -1.09pt +- 1.52 over 640 paired
    games -- it was spending the turn's Supporter, worth a Judge or a Lillie's, to gain one
    Prize that did not end anything. The condition that was missing is the whole card: only
    play it when the extra Prize is the LAST one.
    """
    _BRIAR = 1201
    ladder = ("rule_briar",)

    def rule_briar(self, ctx):
        if ctx.state.supporterPlayed or not ctx.plays or not ctx.attacks:
            return None
        idx = self._hand_idx(ctx, self._BRIAR)
        if idx is None:
            return None
        me_a, opp = ctx.me.active, ctx.opp.active
        if me_a is None or opp is None:
            return None
        if not getattr(me_a.card, "tera", False):      # Briar only pays for a Tera attacker
            return None
        take = _prize_value(opp.pk) + 1
        if take < ctx.me.prizes_left:                  # the extra Prize must END it
            return None
        best = max((self._live_dmg(ctx, i) or 0) for i in ctx.attacks)
        return [idx] if best >= opp.hp > 0 else None


class WaitressL2(BasePolicy, _NeverChosenMixin):
    """Waitress only when one Basic Energy UNLOCKS A BIGGER ATTACK on the Active.

    "Look at the top 6 cards of your deck and attach a Basic Energy card you find there to 1
    of your Pokemon." It is a second attach in a turn, and the engine never chose it in 5,807
    offers because it is neither draw nor search.

    A first rule fired whenever anything on board was unready -- 13 times per 40 games, and
    -1.56pt +- 1.31, because most of those turns the extra energy unlocked nothing and the
    Supporter would have been a Lillie's. The condition that makes the card worth a Supporter
    is narrower: the ACTIVE must be exactly ONE energy short of an attack that hits harder
    than anything it can already use. In this deck that is the real ladder -- Mega Abomasnow
    ex goes from Hammer-lanche at [WW] to Frost Barrier 200 at [WWW].

    Shortfall comes from lm/hidden.insufficient_energy (a port of the engine's own
    GameUtil.h:InsufficientEnergyCount, diffed against it over 272,128 pairs) and the damage
    from lm/damage.final_damage, because Hammer-lanche's printed damage is 0 and says nothing.
    """
    _WAITRESS = 1235
    ladder = ("rule_waitress",)

    def choose_sub(self, ctx):
        """Put the Waitress energy on the ACTIVE.

        Measured: 42.3% of the attaches landed on the BENCH, which throws away the entire
        premise -- the rule fires because the ACTIVE is one energy short of a bigger attack.
        ATTACH_TO falls through to decide_acquire, which ranks bodies generically and does
        not know why this energy was fetched.
        """
        from cg.api import SelectContext
        # ATTACH_FROM is the POKEMON pick (`card:c723@ACTIVE0` / `card:c721@BENCH0`);
        # ATTACH_TO is the energy card pick out of LOOKING. The names read backwards, and
        # overriding the wrong one changed nothing at all -- the A/B came back byte-identical.
        if ctx.sel.context == SelectContext.ATTACH_FROM:
            for i, o in enumerate(ctx.sel.option):
                if o.playerIndex == ctx.mi and o.area == AreaType.ACTIVE:
                    return [i]
        return BasePolicy.choose_sub(self, ctx)

    def rule_waitress(self, ctx):
        if ctx.state.supporterPlayed or not ctx.plays or not ctx.state.energyAttached:
            return None                                # the free attach is still available
        idx = self._hand_idx(ctx, self._WAITRESS)
        if idx is None:
            return None
        obs = getattr(self, "_raw_obs", None)
        if not isinstance(obs, dict) or not obs.get("search_begin_input"):
            return None
        try:
            from lm import damage as _dmg, hidden as _hid, vocab as _v
            dec = _hid.read(obs)
            if dec is None:
                return None
            cur = obs["current"]; yi = cur["yourIndex"]
            me = (cur["players"][yi].get("active") or [None])[0]
            op = (cur["players"][1 - yi].get("active") or [None])[0]
            if not me or not op:
                return None
            card = _v._CARDS.get(me["id"])
            now = nxt = 0
            for aid in ((card.attacks if card else None) or []):
                short = _hid.insufficient_energy(dec, obs, me["serial"], aid)
                if short is None or short > 1:
                    continue
                v, _k = _dmg.final_damage(obs, dec, me["serial"], op["serial"], aid, yi)
                if v is None:
                    continue
                if short == 0:
                    now = max(now, v)
                else:
                    nxt = max(nxt, v)
            return [idx] if nxt > now else None
        except Exception:
            return None


class KlinklangL2(ComboPolicy, _NeverChosenMixin):
    """Klinklang's Emergency Rotation is a FREE body -- take it whenever it is offered.

    "Once during your turn, if this Pokemon is in your hand and your opponent has any Stage 2
    Pokemon in play, you may put this Pokemon onto your Bench." A 140 HP Stage 2 that costs
    no Supporter, no Item and no evolution step, and attacks for [CC] 130. It is worth ONE
    prize, so the spare-ex concern that governs benching Megas does not apply here
    ([[spare-ex-bench-guard]]).

    The engine passed on it 1,697 times because it reads as a Stage 2 in hand with no
    matching Stage 1 on board -- an unplayable evolution -- and its own legality never gets
    re-examined.
    """
    _KLINKLANG = 167
    ladder = ("rule_klinklang",)

    def rule_klinklang(self, ctx):
        if not ctx.plays or len(ctx.me.bench) >= (ctx.me_ps.benchMax or 5):
            return None
        idx = self._hand_idx(ctx, self._KLINKLANG)
        return [idx] if idx is not None else None


class DudunsparceBoxL2(BeatdownPolicy):
    """dudunsparce_box — pay the retreat that arms Gale Thrust.

    Mega Lopunny ex's Gale Thrust is 60, or **230** if it moved from the Bench to the Active
    Spot THAT turn. Measured on the config-only build: 79 of 83 Gale Thrusts (95.2%) landed
    60, because BasePolicy.decide_retreat returns early whenever the active can attack -- so
    Lopunny was promoted once, then swung at 26% power every turn afterwards. Air Balloon x3
    (-2 retreat) is in the live list precisely to make that pivot free, and nothing in the
    generic pilot knows to spend it.

    Two rules, both narrow:
      pivot    retreat into a benched, energised Lopunny when the retreat strands NOTHING
               (active carries no energy -- the Air-Balloon case), and remember the turn.
      finish   while that window is open, take Gale Thrust over Spiky Hopper. `_opt_atk_dmg`
               reads the printed 60 and 160 and would otherwise pick the weaker attack; the
               window is not in the observation (Card::turnState.benchToActive is not
               exported), so the pilot has to remember its own move.
    """
    _LOPUNNY, _FROSLASS = 849, 861
    _GALE, _SPIKY = 1225, 1226
    _AIR_BALLOON = 1174

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pivot_turn = None

    def _bench_lopunny(self, ctx):
        return next((v for v in ctx.me.bench
                     if v.id == self._LOPUNNY and v.energy_count >= 1), None)

    def _window_open(self, ctx):
        a = ctx.me.active
        return (a is not None and a.id == self._LOPUNNY
                and self._pivot_turn == ctx.state.turn)

    def _bench_froslass(self, ctx):
        return next((v for v in ctx.me.bench
                     if v.id == self._FROSLASS and v.energy_count >= 1), None)

    def decide_retreat(self, ctx):
        if ctx.retreat_idx is not None and not ctx.state.retreated:
            a = ctx.me.active
            # `energy_count == 0` is the whole safety condition: the engine only offers
            # retreat when it is payable, and with nothing attached there is nothing to
            # strand. That is exactly the Air-Balloon board this deck plays toward.
            if (a is not None and a.id != self._LOPUNNY and a.energy_count == 0
                    and self._bench_lopunny(ctx) is not None):
                self._pivot_turn = ctx.state.turn
                return [ctx.retreat_idx]
            # A Lopunny stuck in front with one energy can only Gale Thrust for 60 -- its
            # window closed the turn it arrived. Get out for an armed Froslass, unless the
            # defender is a wall Froslass cannot hurt.
            if (a is not None and a.id == self._LOPUNNY and self._pivot_turn != ctx.state.turn
                    and a.energy_count <= 1 and self._bench_froslass(ctx) is not None):
                return [ctx.retreat_idx]
        return BeatdownPolicy.decide_retreat(self, ctx)

    def decide_active(self, ctx, mode="setup"):
        # the pivot only pays if the promotion actually picks Lopunny
        if self._pivot_turn == ctx.state.turn:
            for i, o in enumerate(ctx.sel.option):
                if self._opt_pk_id(ctx, o) == self._LOPUNNY:
                    return [i]
        # OTHERWISE PREFER FROSLASS. Measured over 42 games: Gale Thrust was used 226 times
        # and its window was open on 6.2% of them, i.e. the deck's main attack was a 60 while
        # Resentful Refrain -- one Water for 50 per card in the opponent's hand -- averaged
        # 240 over its 31 uses. Lopunny parked in front is the losing line; it is a FINISHER
        # reached by the pivot, not the body that stands there.
        for i, o in enumerate(ctx.sel.option):
            if self._opt_pk_id(ctx, o) == self._FROSLASS:
                return [i]
        return BeatdownPolicy.decide_active(self, ctx, mode)

    def decide_attack(self, ctx):
        if ctx.attacks and self._window_open(ctx):
            for i in ctx.attacks:
                if ctx.sel.option[i].attackId == self._GALE:
                    return [i]
        return BeatdownPolicy.decide_attack(self, ctx)


class MetagrossL2(ComboPolicy):
    """Steven's Metagross ex (641) energy engine powering a Metal Stomp 200 toolbox.
    200 is fine any time -> no withholding; L2 just prioritises the engine setup."""
    _METAGROSS = 641

    def _is_combo_piece(self, cid):
        return cid in {639, 640, 641}                  # Steven's Beldum/Metang/Metagross engine

    def combo_online(self, ctx):
        return any(v.id == self._METAGROSS for v in ctx.me.inplay())

    def decide_attack(self, ctx):
        return BasePolicy.decide_attack(self, ctx)        # attack when loaded (L0)


class MegaLucarioL2(AggroPolicy):
    """mega_lucario L2 — built from the ZERO-SHOT blind P0 (docs/p0_mega_lucario.json)
    + P1 baseline telemetry (confirmed: Wally dead 0.00, Carmine-holding-678 x13,
    active overload x108, Aura Jab fuel 0.45 discard-F). Fixes F1-F5:"""
    _LUCARIO, _RIOLU = 678, 677
    _MEGA_BRAVE = 983
    _AURA_JAB = 982
    _CAPE = 1159
    # Wally's used to require a LOADED backup Mega on the bench ("Wally STRIPS all its
    # energy"). But the energy goes to HAND and is re-attachable 1/turn, and this deck runs
    # only 8 Pokemon, so that condition blocked **85%** of the heals that qualified on
    # damage. A/B vs the (now live-accurate) alakazam, 150 games each: Wally's 0.06 ->
    # **0.49**/game and win **46 -> 50%** vs alakazam; 0.17 -> **0.58** and **34 -> 40%**
    # vs alakazam_nz_fez. Sign-consistent on both. Denying a 3-prize KO beats one attack.
    _WALLY_NEEDS_BACKUP = False

    def _is_spare_evolution(self, ctx, o):
        """Would this EVOLVE park a second multi-prize body on our BENCH for nothing?

        _is_spare_ex is the same doctrine for Basics, but it is called from decide_bench,
        which only ever considers `card.basic` -- so it cannot see a Stage1/Stage2 ex,
        which arrives by EVOLVING a benched pre-evolution. Measured on the shipped engine
        (engine_v2): with _SPARE_EX_GUARD already ON, mega_lucario still parked **4.75
        prizes** of spare Mega Lucario ex (Stage1, 3 prizes) on its bench -- the opponent
        only needs 6. Turning a benched Riolu (1 prize) into a Mega (3) hands them +2 for
        a body that is not attacking.

        This is NOT the pure loss the Basic case is, so the rule has to be careful:
          * Evolving our ACTIVE is the entire game plan -- never touched here.
          * The FIRST copy is never spare, wherever it lands.
          * A pre-evolution can be evolved on any LATER turn, so waiting costs little --
            EXCEPT when we need a ready body now, which is why CAN_KO_ME_NOW opts out.
          * A thin board needs any body at all (same carve-out as _is_spare_ex).
        So gate on the PRIZE MATH: only refuse when the evolution would put our board's
        total conceded prizes at or past what the opponent still needs to win -- i.e. when
        our own board becomes their whole win condition.

        A/B on the SHIPPED engine, mega_lucario, **150 games per matchup** (bench prizes
        conceded / attacks per game / wins), guard OFF -> ON:
            archaludon     6.07 -> 5.35   3.09 -> 3.11   77% -> 79%
            dragapult      8.23 -> 6.23   5.11 -> 5.36   76% -> 77%
            crustle_stall  9.13 -> 7.22   8.21 -> 9.35   20% -> 27%
        Sign-consistent on all three: the liability drops and attacks do NOT pay for it.

        This lives on MegaLucarioL2 and NOT on BasePolicy on purpose. On BasePolicy it
        fired on **all 45** engine_v2 decks, and only the Lucario family is validated.
        A 60-games-each fleet A/B looked like it caught regressions (mega_latias attacks
        -41%, cynthia_garchomp -27%) but that run was UNDERPOWERED and proves nothing:
        re-running with the guard scoped -- i.e. IDENTICAL code for those decks -- moved
        attacks/game by up to **1.33** and wins by **14/60** on its own. So fleet safety
        is UNKNOWN, not disproven; it is simply unmeasured, and shipping an unmeasured
        behaviour change to 44 other decks is not worth it. (mega_lucario_hg, same
        archetype and same L2, does benefit: bench prizes 3.17 -> 1.98.) A deck that
        wants this opts in by overriding BasePolicy._is_spare_evolution and measuring at
        a sample that can actually resolve the effect -- 150+ games per matchup.

        DO NOT re-judge this at 30 games/matchup -- at that size the same comparison
        reported "-14% attacks" and "crustle 13->5 wins", both pure noise, and even the
        OFF baseline's own mechanism counts moved run to run (bench 6.13 vs 5.47).
        """
        if not _SPARE_EX_GUARD:
            return False
        card = ctx.hand_card(o)
        if card is None or not (getattr(card, "ex", False) or getattr(card, "megaEx", False)):
            return False
        if o.inPlayArea != AreaType.BENCH:
            return False                        # evolving the ACTIVE is the plan
        inplay = ctx.me.inplay()
        if len(inplay) < 2:
            return False                        # thin board: any body is worth it
        copies = [v for v in inplay if v.id == card.cardId]
        if not copies:
            return False                        # our FIRST copy -- always want it
        # --- "do we need the backup NOW?" -- the whole reason this is not a pure loss.
        # A pre-evolution can be evolved on any LATER turn, so waiting costs one turn;
        # being wrong costs us an attacker. Arm the backup the moment every copy we
        # already have stops being reliable.
        #
        # Judge the copies we HAVE, not the Active slot: a bench evolve does not put a
        # body in front, so "our Active isn't one of these, so make one" is a non-sequitur
        # -- and on THIS deck the Mega is BUILT ON THE BENCH and promoted later (measured,
        # guard off: 2.20 bench evolves/game vs 0.55 onto the Active), so "bench evolve ==
        # spare" is only true of the copies BEYOND the one we are building.
        if ctx.opp_threat in ("CAN_KO_ME_NOW", "CAN_KO_ME_NEXT"):
            return False                        # our Active is on the clock
        if all(v.max_hp and v.hp * 2 <= v.max_hp for v in copies):
            return False                        # every copy is past half: it dies soon
        target = ctx.field_pk(o)
        if target is None:
            return False
        add = self._prize_value(card) - self._prize_value(_CARDS.get(target.id))
        if add <= 0:
            return False                        # not actually raising our liability
        conceded = sum(self._prize_value(_CARDS.get(v.id)) for v in inplay)
        opp_needs = ctx.prize.opp if ctx.prize is not None else 6
        return (conceded + add) >= opp_needs

    def _setup_score(self, cid):
        """Promote the body that STARTS THE LINE, not the biggest stand-alone Basic.

        The base hook is pure _best_dmg, blind to what a body BECOMES. On a list running
        the Solrock/Lunatone draw package, Solrock (Cosmic Beam 70) and Lunatone (Power
        Gem 50) out-damage Riolu (Accelerating Stab 30) -- so the engine ranked its ONLY
        path to Mega Lucario ex THIRD and opened on a body that cannot start the clock.

        Measured through the SHIPPED engine (engine_v2 -- see the warning below), 90
        games vs archaludon/dragapult/crustle_stall:
            opening Active Riolu   40% -> **60%**
            first attack on turn   8.2 -> **5.6**   (hg, whose only Basic IS Riolu: 2.8)
            attacked at all      84%  -> 90%
        mega_lucario_hg is unaffected (still 100% Riolu) -- nothing there outranks it.

        WARNING, and the reason this fix was briefly reverted: `tools/generate_agents.py`
        writes agents/<deck>.py as `_engine.act(..., policy=policies.X)` -- the LEGACY
        engine -- so load_agent()/tools/evaluate.py DO NOT EXERCISE THIS FILE AT ALL,
        while mega_lucario actually ships through engine_v2 (see the first line of
        tools/build_engine_v2_submission.py). Judged on legacy numbers this fix looked
        like a no-op (30% -> 31%); on the engine that really ships it is worth 20pt of
        Riolu starts and 2.6 turns of tempo. Measure engine_v2 changes with an
        engine_v2 harness (scratchpad/v2_harness.py), never with evaluate.py.
        """
        return (self._tier_value(cid) or 0) + super()._setup_score(cid)
    _HARIYAMA = 674
    _F9_ON = True
    _WALL_ON = True
    _XERO_ON = True
    _XERO_MIN = 6
    _K_XERO = 1197
    ladder = ("rule_cape", "rule_wall_gust")

    def rule_cape(self, ctx):
        # F9: proactively equip Hero's Cape (+100 HP) to the active Mega FIRST —
        # 340->440 survives the Psychic-weakness one-shot KO (170x2=340), the
        # dominant live loss driver (WR 15% vs Psychic attackers). Attach ASAP,
        # before the Mega is threatened, rather than late via the generic tool step.
        # F9-fix (live v27/v28 whiff: Cape drawn but never attached vs Alakazam): a
        # Pokemon Tool CARRIES THROUGH EVOLUTION, so also equip the active RIOLU —
        # the Mega is then +100 HP the instant it evolves, closing the gap where the
        # Cape was drawn while the pre-evolution (Riolu) was still active.
        if not self._F9_ON:
            return None
        a = ctx.me.active
        if a is None or a.id not in (self._LUCARIO, self._RIOLU):
            return None
        if getattr(a.pk, "tools", None):
            return None                             # already has a tool
        opt = ctx.sel.option
        for i in list(ctx.attaches) + list(ctx.plays):
            c = ctx.hand_card(opt[i])
            if c and c.cardId == self._CAPE:
                pk = ctx.field_pk(opt[i])
                if pk is None or pk is a.pk:         # attach to the active body
                    return [i]
        return None

    @staticmethod
    def _wall_blocks(attacker_view, opp_active_view):
        """True if opp_active's ability PREVENTS ALL our attacker's attack damage.
        Crustle 'Mysterious Rock Inn' blocks damage from opponent {ex} Pokemon;
        Cornerstone Ogerpon 'Cornerstone Stance' blocks Pokemon that HAVE an
        Ability. So a NON-ex attacker (Hariyama/Riolu) breaks Crustle, and an
        ability-less attacker (Mega Lucario ex) breaks Cornerstone (complementary).
        Detection is attacker-aware so we don't mis-flag a breakable wall."""
        if opp_active_view is None:
            return False
        oc = _CARDS.get(opp_active_view.id)
        ac = _CARDS.get(attacker_view.id) if attacker_view is not None else None
        if oc is None:
            return False
        for s in (oc.skills or []):
            t = (getattr(s, "text", "") or "").lower()
            if "prevent all damage" not in t:
                continue
            if "{ex}" in t or "pokémon ex" in t or "pokemon ex" in t or "} pok" in t:
                if ac is not None and (getattr(ac, "ex", False) or getattr(ac, "megaEx", False)):
                    return True
            elif "have an ability" in t or "that have an" in t:
                if ac is not None and (ac.skills or []):
                    return True
            else:
                return True                          # unconditional prevention
        return False

    # NOTE: a wall-aware decide_energy_target (redirect the turn's attach off a Mega
    # whose damage a Crustle-class wall prevents, onto a body it does not block) was
    # written to make Hariyama's Wild Press live, and REVERTED -- it does not do that.
    # Measured on the SHIPPED engine (engine_v2, 25 games vs crustle_stall): the redirect
    # gate fired 61/138 live attaches, but energy went 678 x103 / 677 x24 / 675 x5 /
    # 676 x4 / 673 x2 and **674 Hariyama x0** -- Hariyama is simply not in play at the
    # moment the wall is up (in play 47% of GAMES != in play at attach time), and Wild
    # Press needs 3 attaches against 1/turn while the Mega wants 2. Wild Press fired
    # 0/40 games. Hariyama is dead for CARD-ECONOMY reasons, not pilot reasons; the
    # answer is to cut the line, not to teach the pilot to fuel it.

    def rule_wall_gust(self, ctx):
        if not self._WALL_ON:
            return None
        # Damage-prevention wall detection (live crustle loss: 11 attacks, 0 damage,
        # 0-0 timeout). If the opp Active hard-walls our current attacker, don't dump
        # the self-locking Mega Brave into it — Boss's Orders a KO-able BENCHED target
        # we CAN damage (not itself a wall) and bank a prize instead.
        if ctx.state.supporterPlayed:
            return None
        a = ctx.me.active
        oa = ctx.opp.active
        if a is None or oa is None or not self._wall_blocks(a, oa):
            return None
        dmg = ctx.my_active_dmg
        if dmg <= 0 or not ctx.opp.bench:
            return None
        if not any(b is not None and 0 < b.hp <= dmg and not self._wall_blocks(a, b)
                   for b in ctx.opp.bench):
            return None
        opt = ctx.sel.option
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c and _has(c.name, _GUST):
                return [i]
        return None

    @staticmethod
    def _f(pk):
        # works for both a raw Pokemon (.energies) and a PokemonView (.energy)
        e = getattr(pk, "energies", None)
        if e is None:
            e = getattr(pk, "energy", None)
        return len(e or [])

    # F1 (H5): active Lucario is 2-capped (Mega Brave cost); surplus manual attach
    # pre-loads the BENCH line (twin-tower relay doctrine + post-KO continuity).
    def decide_energy_target(self, ctx):
        if ctx.state.energyAttached:
            return super().decide_energy_target(ctx)
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option
        active = ctx.me.active

        def score(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return -10
            is_active = active is not None and pk is active.pk
            n = self._f(pk)
            if pk.id == self._LUCARIO:
                if is_active:
                    return 100 - n if n < 2 else 1      # cap 2
                return 80 - n if n < 2 else 1           # then bench Lucario to 2
            if pk.id == self._RIOLU:
                return 40 - n if n < 1 else 1           # seed future line
            return 0
        return [max(atts, key=score)]

    def decide_trainer(self, ctx):
        opt = ctx.sel.option
        active = ctx.me.active

        def play_id(cid):
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c and c.cardId == cid:
                    return [i]
            return None

        hand_ids = [c.id for c in (ctx.me_ps.hand or [])]
        # F8 (leaderboard-informed): Night Stretcher (1097) recovers a KO'd
        # attacker from discard when the Riolu/Mega line is thin — the fix for the
        # "backup dry" loss mode (14/25 top-25 leaderboard decks run recovery).
        # No-op for the bare list (no Night Stretcher in it).
        bodies = sum(1 for v in (([active] if active else []) + list(ctx.me.bench))
                     if v is not None and v.id in (self._LUCARIO, self._RIOLU))
        if getattr(self, '_F8_ON', True) and bodies <= 1:
            disc = [getattr(c, "id", None) for c in (getattr(ctx.me_ps, "discard", None) or [])]
            if self._LUCARIO in disc or self._RIOLU in disc:
                r = play_id(1097)
                if r:
                    return r
        # F5a+ (rebuild): stack the OHKO boosters to convert a KO on the opp Active.
        # Premium Power Pro (1141, +30 to Active) and Black Belt's Training (1211, +40
        # to Active {ex}) both add damage "before applying Weakness and Resistance".
        # Play the MINIMAL set that turns THE ATTACK WE CAN ACTUALLY USE lethal.
        #
        # This was gated on `d >= 200` ("only boost the big attack, never waste a
        # booster on Aura Jab"). That gate killed the card. `my_active_dmg` is the max
        # over CURRENTLY AVAILABLE attacks, and **Mega Brave self-locks** ("during your
        # next turn this Pokemon can't use Mega Brave"), so on most turns only Aura Jab
        # (130) is available, d=130, and the gate refused to look. Measured on the
        # SHIPPED engine (engine_v2, 24 games each): Mega Brave fires ~1/game while Aura
        # Jab fires 1.6-7.6/game, and PPP was played **0.12 / 0.04 / 0.00 per game** vs
        # archaludon / dragapult / crustle_stall -- 3 dead slots. (The last session
        # reported this mechanism as firing; that was measured through load_agent() ==
        # the LEGACY engine and was never true of the bundle we ship.)
        #
        # The `d < oa.hp <= d + avail` window below is already the real test -- it only
        # spends a booster when the booster CONVERTS a KO -- so the extra damage gate is
        # redundant and strictly harmful.
        oa = ctx.opp.active
        d = ctx.my_active_dmg
        if (oa is not None and active is not None and active.id == self._LUCARIO
                and d > 0):
            oc = _CARDS.get(oa.id)
            opp_ex = bool(oc and (getattr(oc, "ex", False) or getattr(oc, "megaEx", False)))
            if oc is not None and oc.weakness == 6:     # opp weak to Fighting -> already lethal
                d *= 2
            have_ppp = 1141 in hand_ids
            have_bb = opp_ex and (1211 in hand_ids)
            avail = (30 if have_ppp else 0) + (40 if have_bb else 0)
            if d < oa.hp <= d + avail:                  # boosters CONVERT the KO
                if have_ppp:                            # item first (no supporter cost)
                    r = play_id(1141)
                    if r:
                        return r
                if have_bb and not ctx.state.supporterPlayed:
                    r = play_id(1211)
                    if r:
                        return r
        # F4 (H1): Wally's Compassion (1229, was DEAD) — full-heal the mega ONLY when
        # it is heavily damaged AND it cannot just KO back this turn AND a loaded
        # backup exists (Wally STRIPS all its energy — probe-verified).
        # The "it can just KO back" carve-out has to be PRIZE-AWARE. Measured vs
        # alakazam: of the 142 menus that qualified on damage, **93 (65%) were blocked
        # by this** -- and the KO it deferred to was a **1-prize** Alakazam (HP140) while
        # the Mega it let die is **3 prizes**. They need two Mega KOs to win; we need SIX
        # Alakazams. Wally's -- a 4-of, and the deck's whole answer to that math -- fired
        # **0.03/game**. So only skip the heal when the KO is worth at least what we
        # concede, or when it actually closes the game.
        ko_pays = False
        if oa is not None and ctx.my_active_dmg >= oa.hp:
            ko_pays = (self._prize_value(_CARDS.get(oa.id))
                       >= self._prize_value(_CARDS.get(active.id))
                       or (ctx.prize is not None and ctx.prize.can_close))
        if (not ctx.state.supporterPlayed and active is not None
                and active.id == self._LUCARIO
                and (active.max_hp - active.hp) >= 170
                and not ko_pays
                and (not self._WALLY_NEEDS_BACKUP
                     or any(v.id == self._LUCARIO and self._f(v) >= 2 for v in ctx.me.bench))):
            r = play_id(1229)
            if r:
                return r
        # Xerosic (1197, was DEAD 0.15/game): strip the opponent's hand to 3 to
        # deny Powerful-Hand fuel (Alakazam's 20 dmg x hand size — the mechanism
        # that lets crustle beat alakazam ~78%). Only when the supporter slot is
        # free AND we are NOT taking a gust-KO this turn (that prize comes first)
        # AND the opponent holds a real hand worth stripping (>= _XERO_MIN).
        # The gust-KO carve-out has to mean "we ARE taking that prize", i.e. a gust card
        # is actually IN HAND. It used to fire on "a KO-able benched body EXISTS", which
        # against alakazam -- whose bench is Abra/Kadabra/Dunsparce -- is true nearly
        # always, so Xerosic was blocked **22 of 29** chances (76%) and only 4 survived.
        # Both are supporters, so they compete for the same slot; without Boss's Orders
        # in hand there is no prize to lose and nothing to defer to.
        if (self._XERO_ON and not ctx.state.supporterPlayed
                and ctx.opp.hand_count >= self._XERO_MIN):
            have_gust = (not _XERO_GUST_NEEDS_CARD) or any(
                (c := ctx.hand_card(opt[i])) and _has(c.name, _GUST) for i in ctx.plays)
            gust_ko = (have_gust and ctx.opp.bench and ctx.my_active_dmg > 0
                       and any(v.hp <= ctx.my_active_dmg for v in ctx.opp.bench))
            if not gust_ko:
                r = play_id(self._K_XERO)
                if r:
                    return r
        # F2 (H2): Carmine (1192) discards the WHOLE hand; 678 is unrecoverable
        # (8-Pokemon deck, zero recovery). Never fire it while holding a Lucario.
        if self._LUCARIO in hand_ids:
            skip = {i for i in ctx.plays
                    if (c := ctx.hand_card(opt[i])) and c.cardId == 1192}
            if skip:
                old = ctx.plays
                ctx.plays = [i for i in ctx.plays if i not in skip]
                try:
                    return super().decide_trainer(ctx)
                finally:
                    ctx.plays = old
        return super().decide_trainer(ctx)

    # F3 (H6): discard costs (Ultra Ball etc.) should SEED Aura Jab — prefer
    # tossing Basic {F} (id 6) when the hand holds spares; never toss a Lucario.
    def decide_discard(self, ctx):
        opt = ctx.sel.option
        hand_f = sum(1 for c in (ctx.me_ps.hand or []) if c.id == 6)

        def cost(i):
            cid = self._opt_card_id(ctx, opt[i])
            if cid == self._LUCARIO:
                return _KEEP                      # never
            if cid == 6 and hand_f >= 2:
                return _SHED                      # fuel: F into discard = Aura Jab ammo
            c = _CARDS.get(cid) if cid is not None else None
            return self._card_need(ctx, c) if c else 0
        return sorted(range(len(opt)), key=cost)

    # F6: KO-efficient attack choice. The greedy max-damage floor always fires
    # Mega Brave (983, 270) when affordable, but Mega Brave SELF-LOCKS ("can't use
    # next turn") — live history showed ~45% of Mega Braves overkilled <=130 HP
    # targets that Aura Jab (982, 130 + attach 3 {F}) would KO, wasting the big
    # attack AND leaving us unable to KO the next threat (opp prizes/game 2.9->3.2
    # in the v2.4 regression). Reserve Mega Brave for targets Aura Jab can't KO;
    # otherwise Aura Jab KOs the Active while accelerating energy and keeping
    # Mega Brave available.
    def decide_attack(self, ctx):
        if not ctx.attacks:
            return None
        opt = ctx.sel.option
        oa = ctx.opp.active
        by_aid = {}
        for i in ctx.attacks:
            aid = opt[i].attackId
            if aid is not None:
                by_aid.setdefault(aid, i)
        aura = by_aid.get(self._AURA_JAB)
        brave = by_aid.get(self._MEGA_BRAVE)
        # Wall guard: if the opp Active hard-walls our attacker (Crustle/Cornerstone),
        # every attack does 0 — never SELF-LOCK with Mega Brave for nothing. Prefer
        # Aura Jab (re-attaches 3 {F} = accel toward a breaker, no self-lock).
        a = ctx.me.active
        if self._WALL_ON and oa is not None and a is not None and self._wall_blocks(a, oa):
            if aura is not None:
                return [aura]
            return None
        # only intervene when BOTH are on the menu (i.e. Mega Brave is affordable
        # and we would otherwise overkill with it)
        if oa is not None and aura is not None and brave is not None:
            lc = _CARDS.get(self._LUCARIO)
            oc = _CARDS.get(oa.id)
            wmul = 2 if (oc is not None and lc is not None
                         and oc.weakness == lc.energyType) else 1
            if 130 * wmul >= oa.hp:
                return [aura]        # Aura Jab KOs -> save Mega Brave + accelerate
        return super().decide_attack(ctx)

    def decide_target(self, ctx, kind):
        # Gust away from walls: never drag a damage-immune wall (Crustle/Cornerstone)
        # active — target a KO-able BENCHED body we can actually damage.
        if kind == "gust" and self._WALL_ON:
            a = ctx.me.active
            opt = ctx.sel.option
            dmg = ctx.my_active_dmg

            def sc(i):
                pk = ctx.opp_pokemon_at(opt[i])
                if pk is None:
                    return (-1, 0)
                blocked = a is not None and self._wall_blocks(a, pk)
                koable = 1 if (not blocked and dmg > 0 and pk.hp <= dmg) else 0
                return (koable, -1 if blocked else _target_score(pk))
            return sorted(range(len(opt)), key=sc, reverse=True)
        return super().decide_target(ctx, kind)


class MarnieGrimmsnarlL2(SpreadPolicy):
    """marnie_grimmsnarl (spread/wall). Ported from the proven legacy plan:
    walls (Munkidori/Snorunt/Froslass — ALL type-dead in mono-Dark) buy time while
    Grimmsnarl ex assembles; Punk Up loads the board 2-capped; Adrena-Brain (the
    SpreadPolicy mover) converts chip into KOs; the ONLY retreat is the swap into
    a loaded Grimmsnarl."""
    _GRIMM, _MUNKI = 648, 112
    _LINE = {646, 647, 648}            # Impidimp -> Morgrem -> Grimmsnarl
    _WALLS = {112, 103, 104}           # Munkidori / Snorunt / Froslass

    @staticmethod
    def _dark(pk):
        return sum(1 for e in (pk.energies or []) if e == EnergyType.DARKNESS)

    def decide_retreat(self, ctx):
        # single-purpose swap: retreat ONLY into a bench Grimmsnarl with >=2 Dark
        if ctx.retreat_idx is None or ctx.state.retreated or ctx.attacks:
            return None
        a = ctx.me.active
        if (a is not None and a.id != self._GRIMM
                and any(v.id == self._GRIMM and self._dark(v.pk) >= 2
                        for v in ctx.me.bench)):
            return [ctx.retreat_idx]
        return None                                   # walls otherwise STAY put

    def decide_active(self, ctx, mode="setup"):
        # wall ordering (legacy wscore): loaded Grimmsnarl > Munkidori > Snorunt/
        # Froslass > unloaded line last (never expose the assembling payoff)
        opt = ctx.sel.option

        def wscore(i):
            pk = ctx.field_pk(opt[i])
            cid = self._opt_pk_id(ctx, opt[i])
            if cid is None:
                return 0
            if cid == self._GRIMM:
                return 100 if (pk is not None and self._dark(pk) >= 2) else 10
            if cid == self._MUNKI:
                return 50
            if cid in (103, 104):
                return 40
            if cid in self._LINE:
                return 10
            return 30
        return sorted(range(len(opt)), key=wscore, reverse=True)

    def decide_energy_target(self, ctx):
        # Dark feed plan (user-directed, live-proven): pre-Grimmsnarl feed the walls
        # (bench Munkidori to 1, then active Munkidori to 2 = swap fare); Grimmsnarl
        # is 2-CAPPED (Shadow Bullet needs 2; more just feeds a gust loss), spill to
        # the Marnie's line.
        if ctx.state.energyAttached:
            return super().decide_energy_target(ctx)
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option
        active = ctx.me.active
        grimm_active = active is not None and active.id == self._GRIMM

        def score(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return -10
            d = self._dark(pk)
            is_active = active is not None and pk is active.pk
            if pk.id == self._GRIMM:
                return (60 - d) if d < 2 else 1
            if pk.id == self._MUNKI:
                if not grimm_active:
                    if not is_active and d < 1:
                        return 80
                    if is_active and d < 2:
                        return 70
                return 5 if d < 1 else 0
            if pk.id in self._LINE:
                return 20 if d < 2 else 1
            return 2
        return [max(atts, key=score)]

    def decide_acquire(self, ctx):
        # Punk Up distribution (ATTACH_FROM select carries MY field Pokémon):
        # each Grimmsnarl to 2 Dark, then the rest of the Marnie's line, then walls.
        opt = ctx.sel.option
        mine = [o for o in opt if o.cardId is None and o.playerIndex == ctx.mi
                and o.area in (AreaType.ACTIVE, AreaType.BENCH)]
        if mine:
            def pscore(i):
                pk = ctx.field_pk(opt[i])
                if pk is None:
                    return -1
                d = self._dark(pk)
                if pk.id == self._GRIMM and d < 2:
                    return 100 - d
                if pk.id in self._LINE and d < 2:
                    return 50 - d
                if pk.id == self._MUNKI and d < 1:
                    return 40
                return 0
            return sorted(range(len(opt)), key=pscore, reverse=True)
        return super().decide_acquire(ctx)


class CynthiaGarchompL2(BeatdownPolicy):
    """cynthia_garchomp L2 (docs/p0_cynthia_garchomp.json). Cynthia's Garchomp ex
    (381): Corkscrew Dive (1{F}, 100, draw to 6 — the SUSTAINABLE repeatable attack,
    keeps energy) vs Draconic Buster (2{F}, 260, DISCARDS ALL energy — burst only).
    Roserade (342) Cheer On to Glory buffs all Cynthia's attacks +30. L2 rules:"""
    _GARCHOMP, _ROSERADE = 381, 342
    _CORKSCREW, _DRACONIC = 531, 532

    def _roserade(self, ctx):
        return sum(1 for v in ctx.me.inplay() if v.id == self._ROSERADE)

    @staticmethod
    def _fig(pk):
        e = getattr(pk, "energies", None)
        if e is None:
            e = getattr(pk, "energy", None)
        return sum(1 for x in (e or []) if x == EnergyType.FIGHTING)

    def _expected_dmg(self, ctx):
        if not _active_is(ctx, self._GARCHOMP):
            return ctx.my_active_dmg
        buff = 30 * self._roserade(ctx)
        f = self._fig(ctx.me.active.pk)
        if f >= 2:
            return 260 + buff
        if f >= 1:
            return 100 + buff
        return 0

    def decide_attack(self, ctx):
        # H1: Draconic Buster self-nukes energy; use the sustainable Corkscrew when
        # it already KOs (keeps energy + refuels hand), reserve Draconic for a KO
        # Corkscrew cannot reach, and default non-lethal to Corkscrew.
        if not ctx.attacks:
            return None
        opt = ctx.sel.option
        cork = next((i for i in ctx.attacks if opt[i].attackId == self._CORKSCREW), None)
        drac = next((i for i in ctx.attacks if opt[i].attackId == self._DRACONIC), None)
        if cork is None and drac is None:
            return super().decide_attack(ctx)
        buff = 30 * self._roserade(ctx)
        opp = ctx.opp.active
        if opp is not None:
            if cork is not None and 100 + buff >= opp.hp:
                return [cork]                         # KO while keeping energy
            if drac is not None and 260 + buff >= opp.hp:
                return [drac]                         # burst KO only when needed
        return [cork] if cork is not None else [drac]  # non-lethal: sustainable

    def decide_energy_target(self, ctx):
        # H8 chain: keep the active Garchomp at 2 {F} (Draconic-capable), then pre-load
        # a SECOND Garchomp so a KO'd / Draconic-emptied primary has a ready successor.
        if ctx.state.energyAttached:
            return super().decide_energy_target(ctx)
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option
        active = ctx.me.active

        def score(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return -10
            f = self._fig(pk)
            if pk.id == self._GARCHOMP:
                is_active = active is not None and pk is active.pk
                if is_active:
                    return (100 - f) if f < 2 else 5   # cap active at 2
                return (60 - f) if f < 2 else 1        # then arm a bench Garchomp
            return -5
        return [max(atts, key=score)]


class MegaFeraligatrL2(BeatdownPolicy):
    """mega_feraligatr L2 (docs/p0_mega_feraligatr.json). H1 catastrophic: Feraligatr
    (49) Torrential Heart puts 5 damage counters on ITSELF (self-damage 50) to boost
    Giant Wave 160->280; L0 fires it every turn and self-KOs. Gate it to lethal-only.
    Mega Feraligatr ex (939) Mortal Crunch 200 -> 400 vs a pre-damaged Active."""
    _FERALIGATR, _MEGA = 49, 939

    def _expected_dmg(self, ctx):
        a = ctx.me.active
        if a is None:
            return ctx.my_active_dmg
        if a.id == self._MEGA:
            opp = ctx.opp.active
            return 400 if (opp is not None and (opp.max_hp - opp.hp) > 0) else 200
        return ctx.my_active_dmg

    def decide_ability(self, ctx):
        # only fire Torrential Heart (self-damage) when it CONVERTS a lethal Giant
        # Wave this turn and the body survives the 50 self-damage.
        opt = ctx.sel.option
        keep = []
        for i in ctx.abilities:
            pk = ctx.field_pk(opt[i])
            if pk is not None and pk.id == self._FERALIGATR:
                a = ctx.me.active; opp = ctx.opp.active
                lethal = (a is not None and a.id == self._FERALIGATR
                          and opp is not None and 160 < opp.hp <= 280 and a.hp > 60)
                safe = (a is not None and a.id == self._FERALIGATR and a.hp > 120)
                if not (lethal or safe):
                    continue
            keep.append(i)
        if not keep:
            return None
        old = ctx.abilities; ctx.abilities = keep
        try:
            return super().decide_ability(ctx)
        finally:
            ctx.abilities = old


class OmatsuriL2(SpreadPolicy):
    """omatsuri L2 (docs/p0_omatsuri.json). Dipplin (93) Do the Wave (attack 115,
    display 0) = 20 x bench, and with Festival Grounds (1245) in play Festival Lead
    attacks TWICE = 40 x bench. Display-0 blinds L0: it routes the scarce 5 Grass to
    Thwackey/Seaking and mis-evolves. L2 fixes: real-damage perception, route Grass
    to the Dipplin line, evolve the payoff line, widen the bench."""
    _DIPPLIN, _FESTIVAL, _DTW = 93, 1245, 115
    _LINE = {92, 93}
    bench_target = 5

    def _festival(self, ctx):
        return bool(ctx.state.stadium) and any(c.id == self._FESTIVAL for c in ctx.state.stadium)

    def _expected_dmg(self, ctx):
        if not _active_is(ctx, self._DIPPLIN):
            return ctx.my_active_dmg
        return 20 * len(ctx.me.bench) * (2 if self._festival(ctx) else 1)

    def decide_attack(self, ctx):
        if ctx.attacks:
            dtw = next((i for i in ctx.attacks if ctx.sel.option[i].attackId == self._DTW), None)
            if dtw is not None:
                opp = ctx.opp.active
                d = self._expected_dmg(ctx)
                if (opp is not None and d >= opp.hp) or d >= 40:
                    return [dtw]                      # the real payoff (display 0)
        return super().decide_attack(ctx)

    def decide_evolve(self, ctx):
        opt = ctx.sel.option
        dip = next((i for i in ctx.evolves
                    if getattr(ctx.hand_card(opt[i]), "cardId", 0) == self._DIPPLIN), None)
        if dip is not None:
            return [dip]                              # payoff line first
        return super().decide_evolve(ctx)

    def decide_energy_target(self, ctx):
        if ctx.state.energyAttached:
            return super().decide_energy_target(ctx)
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option

        def score(i):
            pk = ctx.field_pk(opt[i])
            return 100 if (pk is not None and pk.id in self._LINE) else 5
        return [max(atts, key=score)]

    def decide_trainer(self, ctx):
        # keep the Festival Grounds engine up; don't play our own Forest of Vitality
        # over it. If no stadium is up, prefer Festival Grounds specifically.
        if not ctx.state.stadium:
            opt = ctx.sel.option
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c and c.cardId == self._FESTIVAL:
                    return [i]
        return super().decide_trainer(ctx)


class ZoroarkL2(BeatdownPolicy):
    """ns_zoroark L2 (docs/p0_ns_zoroark.json). H1 catastrophic: Trade (293) draws
    2 / discards 1, and with up to 4 Zoroarks firing it every turn the deck mills
    ~8 cards/turn -> deckout (24/40 games in probe). Gate Trade to real need +
    deck safety. Night Joker (attack 403, display 0) copies the MAX-display benched
    N's attack paid with Zoroark's own {D}{D}; feed the shared real-damage view."""
    _ZOROARK, _NIGHT_JOKER, _LILLIE = 293, 403, 1227
    _ZORUA, _PLAN = 292, 1221             # N's Plan: move 2 energy bench -> active
    _CASTLE, _PPUP, _MUNKI = 1253, 1113, 112
    _DEAD = frozenset({906, 303, 112})   # Zekrom/Reshiram/Munkidori: costs unpayable by {D}
    deck_low = 8                          # HH1/HH7: stop Ultra Ball/Poffin/Cyrano digging early

    # ---- ROTATION line (pipeline v2; forced-probe verified E1-E4): N's Castle
    # zeroes N's retreat cost -> keep TWO armed Zoroarks and rotate the locked one
    # out for free, so Rampaging Thunder 250 fires EVERY turn (probe: consecutive-
    # turn 250s x5, retreat free 4/4 under Castle). Trade discards feed N's PP Up
    # (attach D from discard to a benched N's) — the discard is FUEL, not loss.
    ladder = ("rule_castle", "rule_ppup", "rule_rotate")

    def _castle_up(self, ctx):
        return bool(ctx.state.stadium) and any(
            x.id == self._CASTLE for x in ctx.state.stadium)

    def _armed(self, ctx):
        return sum(1 for v in ctx.me.inplay()
                   if v.id == self._ZOROARK and self._dk(v.pk) >= 2)

    def rule_castle(self, ctx):
        if self._castle_up(ctx) or ctx.state.stadiumPlayed:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and c.cardId == self._CASTLE:
                return [i]
        return None

    def rule_ppup(self, ctx):
        # N's PP Up: re-attach a discarded D to the body that needs it — the
        # UNARMED benched Zoroark first, then Munkidori (Adrena-Brain needs 1 D).
        if not any(x.id == 7 for x in (ctx.me_ps.discard or [])):
            return None
        # target priority: unarmed benched Zoroark, then pre-evolution Zorua
        # (energy persists through evolution -> arms Zoroark #2 早く), then
        # 1D Munkidori once both Zoroarks armed.
        need = any(v.id in (self._ZOROARK, self._ZORUA) and self._dk(v.pk) < 2
                   for v in ctx.me.bench)
        if not need and self._armed(ctx) >= 2:
            need = any(v.id == self._MUNKI and self._dk(v.pk) < 1 for v in ctx.me.bench)
        if not need:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and c.cardId == self._PPUP:
                return [i]
        return None

    def rule_rotate(self, ctx):
        # active Zoroark armed but CANNOT attack (post-250 lock, sleep...):
        # swap in the second armed Zoroark — free under Castle, else a Switch.
        a = ctx.me.active
        if (a is None or a.id != self._ZOROARK or ctx.attacks
                or self._dk(a.pk) < 2):
            return None
        if not any(v.id == self._ZOROARK and self._dk(v.pk) >= 2
                   for v in ctx.me.bench):
            return None
        if (self._castle_up(ctx) and ctx.retreat_idx is not None
                and not ctx.state.retreated):
            return [ctx.retreat_idx]
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and c.cardId == 1123:               # Switch
                return [i]
        return None

    def decide_discard(self, ctx):
        # Trade/Ultra Ball discard choice: a discarded basic D is PP UP FUEL
        # (combo edge, individually-negative without rule_ppup) — shed it first
        # once the hand can spare it; protect the line cards.
        # Keyed on the raw opt[i].cardId, every rule below was DEAD: a discard menu names
        # hand cards by reference (cardId=None), so `cid == 7` and `cid in protect` never
        # matched once -- measured 427/427 options blind, 421 of them resolvable.
        opt = ctx.sel.option
        hand_d = sum(1 for x in (ctx.me_ps.hand or []) if x.id == 7)
        fuel_ok = hand_d >= 2 or self._armed(ctx) >= 2
        protect = {self._ZOROARK, self._ZORUA, self._CASTLE, self._PPUP, self._PLAN}

        def key(i):
            cid = self._opt_card_id(ctx, opt[i])
            if cid == 7 and fuel_ok:
                return _SHED
            base = self._card_need(ctx, _CARDS.get(cid))
            if cid in protect:
                base += 500
            return base
        return sorted(range(len(opt)), key=key)

    @staticmethod
    def _dk(pk):
        e = getattr(pk, "energies", None) or getattr(pk, "energy", None) or []
        return sum(1 for x in e if x == EnergyType.DARKNESS)

    def _copy_dmg(self, ctx):
        best = 0
        for v in ctx.me.bench:
            c = _CARDS.get(v.id)
            if c:
                best = max(best, _best_dmg(c))
        return best

    def _expected_dmg(self, ctx):
        if _active_is(ctx, self._ZOROARK):
            return self._copy_dmg(ctx) or ctx.my_active_dmg
        return ctx.my_active_dmg

    def decide_ability(self, ctx):
        opt = ctx.sel.option                          # HH1: gate the Trade self-mill
        keep = []
        for i in ctx.abilities:
            pk = ctx.field_pk(opt[i])
            if pk is not None and pk.id == self._ZOROARK:
                # the hand>=8 cap self-limits the 4-Zoroark chain to ~1 fire; the
                # deck<=8 floor stops only the terminal mill (Lillie refuels at 16;
                # legacy digs with NO floor — starving Trade costs more than it saves).
                if ctx.me.hand_count >= 8 or ctx.me.deck_count <= 8:
                    continue
            keep.append(i)
        if not keep:
            return None
        old = ctx.abilities; ctx.abilities = keep
        try:
            return super().decide_ability(ctx)
        finally:
            ctx.abilities = old

    # NOTE (iter-2 round-3): forcing the Sustain doctrine by never benching N's
    # Zekrom REGRESSED (-20.5 -> -22.7 at n=367) — Zekrom's 250/500 reach matters.
    # Round-4 found the REAL bug: the copy target IS agent-controllable via the
    # SelectContext.ATTACK sub-menu (35), which the old L0 fallback answered with
    # option[0] = Powerful Rage (0 dmg). See decide_attack_choice below.

    def decide_attack_choice(self, ctx):
        # Copy-menu doctrine: Virtuous Flame 170 (no drawback) is the floor; take
        # Rampaging Thunder 250 for a KO 170 can't reach, OR whenever the ROTATION
        # is online (2nd armed Zoroark + Castle free retreat) — then the self-lock
        # costs nothing and 250/turn is strictly better.
        opt = ctx.sel.option
        atk = {opt[i].attackId: i for i, o in enumerate(opt)
               if o.type == OptionType.ATTACK}
        rt, vf = atk.get(1306), atk.get(421)
        opp = ctx.opp.active
        hp = opp.hp if opp is not None else 9999
        rotation = (self._castle_up(ctx) and any(
            v.id == self._ZOROARK and self._dk(v.pk) >= 2 for v in ctx.me.bench))
        if rt is not None and (170 < hp <= 250 or (rotation and hp > 170)):
            return [rt]
        if vf is not None:
            return [vf]
        return super().decide_attack_choice(ctx)

    def decide_energy_target(self, ctx):
        # HH2: {D} arms the copy-attacker Zoroarks first (both bodies — the
        # rotation needs TWO); once both are armed, 1 D on Munkidori turns on
        # Adrena-Brain (move 3 damage counters/turn). Zekrom/Reshiram stay dead.
        if ctx.state.energyAttached:
            return None
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option
        armed = self._armed(ctx)

        def sc(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return -100
            if pk.id == self._ZOROARK:
                is_active = ctx.me.active is not None and pk is ctx.me.active.pk
                d = self._dk(pk)
                if d >= 2:
                    return 30                         # already armed: overflow only
                return (300 if is_active else 200) - d
            if pk.id == self._ZORUA and armed >= 1:
                # arm the pre-evolution body: energy persists through evolution,
                # so this is the fastest path to the SECOND armed Zoroark
                # (two_armed conformance was 37%/T13 — the rotation bottleneck).
                return 150 - self._dk(pk) if self._dk(pk) < 2 else 10
            if pk.id == self._MUNKI:
                return 100 if (armed >= 2 and self._dk(pk) < 1) else -40
            if pk.id in self._DEAD:
                return -50                            # never feed the type-dead line
            return 5
        return [max(atts, key=sc)]

    def decide_trainer(self, ctx):
        opt = ctx.sel.option
        if not ctx.state.supporterPlayed:
            # HH8: play Boss's Orders when the REAL copy damage KOs a benched target
            # (Zoroark's display 0 makes the base gust gate refuse to fire).
            if _active_is(ctx, self._ZOROARK):
                d = self._copy_dmg(ctx)
                if d > 0 and any(v.hp <= d for v in ctx.opp.bench):
                    for i in ctx.plays:
                        c = ctx.hand_card(opt[i])
                        if c and _has(c.name, _GUST):
                            return [i]
            # Cyrano (search up to 3 ex = Zoroarks): fire while the Zoroark line
            # is thin — the 2nd body is the rotation prerequisite (two_armed
            # conformance driver).
            line = (sum(1 for v in ctx.me.inplay() if v.id in (self._ZOROARK, self._ZORUA))
                    + sum(1 for x in (ctx.me_ps.hand or []) if x.id in (self._ZOROARK, self._ZORUA)))
            if line < 2 and ctx.me.deck_count > self.deck_low:
                for i in ctx.plays:
                    c = ctx.hand_card(opt[i])
                    if c and c.cardId == 1205:
                        return [i]
            # HH6: Lillie's Determination = anti-deckout refuel (shuffle hand -> deck,
            # draw 6). Fire it as the deck thins, before the deck-low guard freezes.
            if ctx.me.deck_count <= 16 and ctx.me.hand_count >= 4:
                for i in ctx.plays:
                    c = ctx.hand_card(opt[i])
                    if c and c.cardId == self._LILLIE:
                        return [i]
            # N's Plan (dead under L0): pull 2 {D} off the bench onto a freshly
            # promoted Zoroark -> re-arm the copy attack the SAME turn (post-KO
            # chain was 0.44). Only when the active is a Zoroark short of {D}{D}
            # and the bench actually holds energy to move.
            a = ctx.me.active
            if (a is not None and a.id == self._ZOROARK and self._dk(a.pk) < 2
                    and any(self._dk(v.pk) > 0 for v in ctx.me.bench)):
                for i in ctx.plays:
                    c = ctx.hand_card(opt[i])
                    if c and c.cardId == self._PLAN:
                        return [i]
        # Switch (dead under L0): a gusted-active provider/Munkidori is a WALL
        # (type-dead, can't attack, may not afford retreat) — swap a benched
        # Zoroark back in to keep the copy chain alive.
        a = ctx.me.active
        if (a is not None and a.id != self._ZOROARK
                and any(v.id == self._ZOROARK for v in ctx.me.bench)):
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c and c.cardId == 1123:            # Switch
                    return [i]
        return super().decide_trainer(ctx)

    def decide_retreat(self, ctx):
        # Base retreat compares bench best_potential_dmg > active's — Zoroark
        # DISPLAYS 0, so the base NEVER retreats a stuck provider into it.
        # Retreat any non-Zoroark active for a benched Zoroark (prefer armed).
        if ctx.retreat_idx is None or ctx.state.retreated:
            return None
        a = ctx.me.active
        if (a is not None and a.id != self._ZOROARK and not ctx.attacks
                and any(v.id == self._ZOROARK for v in ctx.me.bench)):
            return [ctx.retreat_idx]
        return super().decide_retreat(ctx)

    def decide_active(self, ctx, mode="setup"):
        # Post-KO promotion: the L1 generic promotes by display damage, which puts
        # a type-dead Zekrom/Reshiram (250/170 display, unpayable costs) in front
        # and breaks the copy chain. Always promote the Zoroark line: armed Zoroark
        # (2 {D}) > any Zoroark > Zorua > everything else.
        opt = ctx.sel.option

        def rank(i):
            pk = ctx.field_pk(opt[i])
            cid = self._opt_pk_id(ctx, opt[i])
            if cid is None:
                return -1
            if cid == self._ZOROARK:
                return 300 + (self._dk(pk) if pk is not None else 0)
            if cid == self._ZORUA:
                return 200
            if cid in self._DEAD:
                return 10
            return 50
        return sorted(range(len(opt)), key=rank, reverse=True)

    def decide_target(self, ctx, kind):
        # HH8: gust value uses the REAL copy damage, not Zoroark's display 0.
        if kind == "gust" and _active_is(ctx, self._ZOROARK):
            d = self._copy_dmg(ctx)
            opt = ctx.sel.option

            def gsc(i):
                pk = ctx.opp_pokemon_at(opt[i])
                if pk is None:
                    return (-1, 0)
                return (1 if (d > 0 and pk.hp <= d) else 0, _target_score(pk))
            return sorted(range(len(opt)), key=gsc, reverse=True)
        return super().decide_target(ctx, kind)

    def decide_attack(self, ctx):
        if _active_is(ctx, self._ZOROARK) and ctx.attacks:
            opp = ctx.opp.active
            nj = next((i for i in ctx.attacks
                       if ctx.sel.option[i].attackId == self._NIGHT_JOKER), None)
            if nj is not None:
                d = self._copy_dmg(ctx)
                if (opp is not None and d >= opp.hp) or d >= 20:
                    return [nj]                       # real payoff (display 0)
        return super().decide_attack(ctx)


class EthanHoohL2(BasePolicy):
    """ethan_hooh L2 (docs/p0_ethan_hooh.json). Ethan's Magcargo (356) Lava Burst
    (attack 493, display 0) = 70 x min(5, R discarded) -> up to 350 on-demand KO;
    Ho-Oh ex (357) Shining Feathers 160 grinds while Golden Flame banks R on the
    benched Magcargo. Display-0 blinds L0 to the Lava KO; feed the real-damage view
    and keep manual {R} on the active attacker (bench Magcargo is fed by ability)."""
    _MAGCARGO, _HOOH, _LAVA, _LATIAS, _SLUGMA = 356, 357, 493, 184, 355

    # ---- pipeline v2 GRIND+BANK line (P0.5 probes): Golden Flame (ability, up
    # to 2 R/turn from hand) must feed the DESIGNATED bank bodies, not Slugma/
    # Fezandipiti (probe: targets scattered). Lava Burst self-empties Magcargo ->
    # Melt Away rc0 free retreat out; Latias ex Skyliner = rc0 for our BASICS
    # (Ho-Oh!), so the pivot in/out of the Lava finisher is free. The v1 "hoard
    # then attack" L2 regressed because it had the bank WITHOUT the rotation.
    ladder = ("rule_lava_promote", "rule_pivot_grinder", "rule_melt_retreat")

    @staticmethod
    def _r(pk):
        e = getattr(pk, "energies", None) or getattr(pk, "energy", None) or []
        return sum(1 for x in e if x == EnergyType.FIRE)

    def _expected_dmg(self, ctx):
        a = ctx.me.active
        if a is None:
            return ctx.my_active_dmg
        if a.id == self._MAGCARGO:
            return 70 * min(5, self._r(a.pk))
        if a.id == self._HOOH:
            return 160
        return ctx.my_active_dmg

    def _skyliner(self, ctx):
        return any(v.id == self._LATIAS for v in ctx.me.inplay())

    def rule_lava_promote(self, ctx):
        # on-demand KO: benched Magcargo's Lava reaches what the active can't —
        # pivot it in (free when the active is a Basic under Skyliner; else Switch).
        opp = ctx.opp.active
        a = ctx.me.active
        if opp is None or a is None or a.id == self._MAGCARGO:
            return None
        bank_r = max((self._r(v.pk) for v in ctx.me.bench
                      if v.id == self._MAGCARGO), default=0)
        bank = 70 * min(5, bank_r)
        lethal_pull = bank >= opp.hp and ctx.my_active_dmg < opp.hp
        stuck_front = bank_r >= 3 and ctx.my_active_dmg <= 0 and not ctx.attacks
        if not (lethal_pull or stuck_front):
            return None
        active_basic = a.card is not None and a.card.basic
        if (self._skyliner(ctx) and active_basic and ctx.retreat_idx is not None
                and not ctx.state.retreated):
            return [ctx.retreat_idx]
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and self._in_bucket(c, _SWITCH_CARDS):
                return [i]
        return None

    def rule_pivot_grinder(self, ctx):
        # r6 (post-mortem: first_attack median was 14.5 — the 0R setup active
        # WALLED until the opponent KO'd it): when the active cannot attack and
        # an ARMED body waits on the bench, pivot proactively — free under
        # Skyliner for basics, else burn a Switch.
        a = ctx.me.active
        if a is None or ctx.attacks:
            return None
        armed_bench = any((v.id == self._HOOH and self._r(v.pk) >= 4)
                          or (v.id == self._MAGCARGO and self._r(v.pk) >= 3)
                          for v in ctx.me.bench)
        if not armed_bench:
            return None
        active_basic = a.card is not None and a.card.basic
        if (self._skyliner(ctx) and active_basic and ctx.retreat_idx is not None
                and not ctx.state.retreated):
            return [ctx.retreat_idx]
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and self._in_bucket(c, _SWITCH_CARDS):
                return [i]
        return None

    def rule_melt_retreat(self, ctx):
        # post-Lava: Magcargo is empty (Melt Away rc0) and can't attack — free
        # retreat back to the grinder instead of walling with a 130 HP body.
        a = ctx.me.active
        if (a is not None and a.id == self._MAGCARGO and not ctx.attacks
                and self._r(a.pk) == 0 and ctx.me.bench
                and ctx.retreat_idx is not None and not ctx.state.retreated):
            return [ctx.retreat_idx]
        return None

    def decide_acquire(self, ctx):
        # Golden Flame target (ATTACH_TO): bank on Magcargo (finisher, cap 5),
        # then a benched Ho-Oh (grinder #2, cap 4). Never Slugma/tech bodies.
        opt = ctx.sel.option
        mine = [i for i, o in enumerate(opt)
                if o.cardId is None and o.playerIndex == ctx.mi
                and o.area in (AreaType.ACTIVE, AreaType.BENCH)]
        if mine:
            def sc(i):
                pk = ctx.field_pk(opt[i])
                if pk is None:
                    return -1
                r = self._r(pk)
                if pk.id == self._HOOH:
                    return (300 - r) if r < 4 else 5   # grinder first (legacy FOCUS)
                if pk.id == self._MAGCARGO:
                    return (200 - r) if r < 5 else 5
                return 2
            best = max(mine, key=sc)
            return [best] + [i for i in range(len(opt)) if i != best]
        return super().decide_acquire(ctx)

    def decide_energy_target(self, ctx):
        # manual attach: active Ho-Oh toward Shining Feathers (4R); otherwise the
        # bank bodies. Latias is an ability body (Skyliner), never an attacker.
        if ctx.state.energyAttached:
            return None
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option

        def sc(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return -100
            is_active = ctx.me.active is not None and pk is ctx.me.active.pk
            r = self._r(pk)
            if pk.id == self._HOOH:
                if is_active and r >= 2:
                    return 250 - r
                return (150 - r) if (not is_active and r < 4) else 5
            if pk.id == self._MAGCARGO:
                return (140 - r) if r < 5 else 5
            if pk.id == self._LATIAS:
                return -50
            return 2
        best = max(atts, key=sc)
        if sc(best) <= 5:
            return super().decide_energy_target(ctx)   # early game: cheap tempo
        return [best]

    def decide_active(self, ctx, mode="setup"):
        # promotion: armed Magcargo first when it KOs, else the most-fueled Ho-Oh.
        opt = ctx.sel.option
        opp = ctx.opp.active

        def rank(i):
            pk = ctx.field_pk(opt[i])
            cid = self._opt_pk_id(ctx, opt[i])
            if cid is None:
                return -1
            r = self._r(pk) if pk is not None else 0
            if cid == self._MAGCARGO:
                if r >= 3:
                    return 450 + r                    # primary: Lava 210+ every turn
                return 60 + r
            if cid == self._HOOH:
                # sub-4R Ho-Oh must NOT front: Golden Flame is bench-only, so an
                # early active Ho-Oh is a starved wall (legacy fronts Pinsir/
                # Latias and charges Ho-Oh on the bench — oracle histogram).
                return (470 + r) if r >= 4 else 70 + 10 * r
            if cid == self._LATIAS:
                return 60
            # cheap tanks front while the bench charges
            if pk is not None and pk.energies and len(pk.energies) >= 1:
                return 150
            return 130
        return sorted(range(len(opt)), key=rank, reverse=True)

    def decide_attack(self, ctx):
        # LEGACY-ORACLE fix: fire Lava Burst whenever Magcargo holds >=3 R
        # (210-350). The old lethal-only gate + display-0 meant a loaded active
        # Magcargo PASSED the turn (attacks_flow was stuck at 0.176).
        if _active_is(ctx, self._MAGCARGO) and ctx.attacks:
            lava = next((i for i in ctx.attacks
                         if ctx.sel.option[i].attackId == self._LAVA), None)
            if lava is not None and self._r(ctx.me.active.pk) >= 3:
                return [lava]
        return super().decide_attack(ctx)


class ManectricL2(BasePolicy):
    """manectric L2 (docs/p0_manectric.json). Mega Manectric ex (737) Riotous
    Blasting (attack 1065) = 200 + 130 (discard all Energy) = 330 real. H1
    catastrophic: Scoop Up Cyclone (1093) can bounce our last body -> auto-loss;
    guard it. Display-underrated 330 blinds lethal detection; feed real damage."""
    _MANECTRIC, _RIOTOUS, _SCOOP, _LILLIE = 737, 1065, 1093, 1227
    _PAYOFF = frozenset({737, 868})       # Mega Manectric / Mega Eelektross bodies
    deck_low = 8                          # HH3: deckout throttle

    # ---- pipeline v2 PING-PONG ECONOMY line (P0.5 probes / DB): Mega Manectric
    # is rc0 (rotation is FREE), Dynamotor attaches L FROM THE DISCARD to a
    # BENCHED body, and Riotous Blasting discards this Pokemon's 3 L = Dynamotor
    # fuel. Loop: Riotous 330 -> free retreat out -> Dynamotor recharges the
    # bench -> the twin swings next. Discarded L is fuel, not loss (combo edge,
    # individually negative). Probe showed Dynamotor targets scattered onto
    # Eelektrik/Budew — the ATTACH_TO routing below is the core fix.
    ladder = ("rule_pingpong",)

    @staticmethod
    def _l(pk):
        e = getattr(pk, "energies", None) or getattr(pk, "energy", None) or []
        return sum(1 for x in e if x == EnergyType.LIGHTNING)

    def _expected_dmg(self, ctx):
        a = ctx.me.active
        if a is not None and a.id == self._MANECTRIC:
            # Riotous 330 when we hold >=3 L to pay + discard; else Flash Ray 120.
            return 330 if self._RIOTOUS in {getattr(o, "attackId", None)
                                            for o in ctx.sel.option} else 120
        return ctx.my_active_dmg

    def rule_pingpong(self, ctx):
        # spent active Manectric (post-Riotous, <3 L) + armed twin on the bench:
        # rotate for free (rc0) so the 330 keeps coming every turn.
        a = ctx.me.active
        if (a is None or a.id != self._MANECTRIC or ctx.attacks
                or self._l(a.pk) >= 2):
            return None
        # rc0: rotating out a spent Manectric costs NOTHING — do it whenever any
        # bench body can actually act (line rev after P4 REJECT: the armed-twin-
        # only gate left 0-energy Manectrics walling; payoff_armed median T23).
        if not any((v.energy_count or 0) >= 1 or v.id in self._PAYOFF
                   for v in ctx.me.bench):
            return None
        if ctx.retreat_idx is not None and not ctx.state.retreated:
            return [ctx.retreat_idx]
        return None

    def decide_acquire(self, ctx):
        # Dynamotor target (ATTACH_TO): charge the PAYOFF bodies on the bench
        # (Manectric to 3 L first), never the engine/tech bodies.
        opt = ctx.sel.option
        mine = [i for i, o in enumerate(opt)
                if o.cardId is None and o.playerIndex == ctx.mi
                and o.area in (AreaType.ACTIVE, AreaType.BENCH)]
        if mine:
            def sc(i):
                pk = ctx.field_pk(opt[i])
                if pk is None:
                    return -1
                l = self._l(pk)
                if pk.id == self._MANECTRIC:
                    return (300 - l) if l < 3 else 10
                if pk.id in self._PAYOFF:
                    return (150 - l) if l < 3 else 5
                return 2
            best = max(mine, key=sc)
            return [best] + [i for i in range(len(opt)) if i != best]
        return super().decide_acquire(ctx)

    def decide_discard(self, ctx):
        # discarded basic L is Dynamotor FUEL (combo edge): shed it first once
        # the hand can spare it. (Was keyed on the raw cardId -- 94/94 options blind,
        # so neither the fuel rule nor the protect list ever fired.)
        opt = ctx.sel.option
        hand_l = sum(1 for x in (ctx.me_ps.hand or []) if x.id == 4)
        armed = any(v.id in self._PAYOFF and self._l(v.pk) >= 3
                    for v in ctx.me.inplay())
        def key(i):
            cid = self._opt_card_id(ctx, opt[i])
            if cid == 4 and (hand_l >= 2 or armed):
                return _SHED
            base = self._card_need(ctx, _CARDS.get(cid))
            if cid in (737, 868, 512):
                base += 500
            return base
        return sorted(range(len(opt)), key=key)

    def decide_active(self, ctx, mode="setup"):
        # promotion: the armed payoff body first — never a wall by display value.
        opt = ctx.sel.option

        def rank(i):
            pk = ctx.field_pk(opt[i])
            cid = self._opt_pk_id(ctx, opt[i])
            if cid is None:
                return -1
            l = self._l(pk) if pk is not None else 0
            n_e = len(pk.energies or []) if pk is not None else 0
            if cid == self._MANECTRIC:
                return (400 + l) if l >= 2 else 120 + 30 * l
            if cid in self._PAYOFF:
                return (300 + l) if l >= 2 else 100 + 20 * l
            # promote something that can ACT over an empty payoff wall
            return 150 if n_e >= 1 else 50
        return sorted(range(len(opt)), key=rank, reverse=True)

    def decide_energy_target(self, ctx):
        # HH4/HH5: manual L belongs on a PAYOFF body (charge a Mega Manectric to LLL
        # for Riotous 330), not Eelektrik/Clefairy/Tynamo.
        if ctx.state.energyAttached:
            return None
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option

        def sc(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return -100
            if pk.id in self._PAYOFF:
                return 300 - self._l(pk)
            return 5
        return [max(atts, key=sc)]

    def decide_trainer(self, ctx):
        opt = ctx.sel.option
        # HH3: Lillie's Determination refuels against deckout.
        if (not ctx.state.supporterPlayed and ctx.me.deck_count <= 12
                and ctx.me.hand_count >= 4):
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c and c.cardId == self._LILLIE:
                    return [i]
        # HH1: never Scoop Up our own board down to the last body.
        if len(ctx.me.inplay()) <= 1:
            plays = [i for i in ctx.plays
                     if getattr(ctx.hand_card(opt[i]), "cardId", 0) != self._SCOOP]
            old = ctx.plays; ctx.plays = plays
            try:
                return super().decide_trainer(ctx)
            finally:
                ctx.plays = old
        return super().decide_trainer(ctx)

    def decide_attack(self, ctx):
        if _active_is(ctx, self._MANECTRIC) and ctx.attacks:
            rb = next((i for i in ctx.attacks
                       if ctx.sel.option[i].attackId == self._RIOTOUS), None)
            opp = ctx.opp.active
            rotation = any(v.id in self._PAYOFF and self._l(v.pk) >= 3
                           for v in ctx.me.bench)
            if rb is not None and opp is not None and (330 >= opp.hp or rotation):
                return [rb]                # discard-all is Dynamotor fuel
        return super().decide_attack(ctx)


class MegaVenusaurL2(BasePolicy):
    """mega_venusaur L2 (pipeline v2.1; oracle=_kanga_focus[652]). Line: evolve
    Bulbasaur->Ivysaur->Mega Venusaur ex (380HP, Jungle Dump 240+heal30) and
    CONCENTRATE all {G} on it — Solar Transfer (as-often-as-you-like G move)
    makes every placement recoverable, so arm the LINE early and consolidate.
    Ogerpon (Teal Dance accel + Myriad display-30-real-30+30xE) tanks early.
    v2.1 histogram showed Bulbasaur walling 40% of turns and 0% energy on the
    Mega — promotion-resolution + this routing are the fixes."""
    _MEGA, _IVY, _BULBA, _OGER, _BULU = 652, 651, 650, 96, 920
    _JUNGLE = 941
    ladder = ("rule_promote_focus",)

    @staticmethod
    def _g(pk):
        e = getattr(pk, "energies", None) or getattr(pk, "energy", None) or []
        return sum(1 for x in e if x == EnergyType.GRASS)

    def _mega(self, ctx):
        return next((v for v in ctx.me.inplay() if v.id == self._MEGA), None)

    def _expected_dmg(self, ctx):
        a = ctx.me.active
        if a is None:
            return ctx.my_active_dmg
        if a.id == self._MEGA:
            return 240
        if a.id == self._OGER:
            return 30 + 30 * len(getattr(a.pk, "energies", None) or [])
        return ctx.my_active_dmg

    def decide_ability(self, ctx):
        # Solar Transfer only when it MOVES value (donor has G, Mega short of 4)
        # — unconditional firing ping-ponged one energy ~160x/game (audit).
        opt = ctx.sel.option
        keep = []
        for i in ctx.abilities:
            pk = ctx.field_pk(opt[i])
            if pk is not None and pk.id == self._MEGA:
                m = self._mega(ctx)
                donors = any(v.id != self._MEGA and self._g(v.pk) > 0
                             for v in ctx.me.inplay())
                if not (m is not None and self._g(m.pk) < 4 and donors):
                    continue
            keep.append(i)
        if not keep:
            return None
        old = ctx.abilities; ctx.abilities = keep
        try:
            return super().decide_ability(ctx)
        finally:
            ctx.abilities = old

    def decide_energy_target(self, ctx):
        # concentrate on the Mega line (energy survives evolution; Solar
        # Transfer consolidates later anyway). Ogerpon second (Teal Dance body).
        if ctx.state.energyAttached:
            return None
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option

        def sc(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return -100
            g = self._g(pk)
            if pk.id == self._MEGA:
                return (300 - g) if g < 4 else 10
            if pk.id == self._IVY:
                return 200 - g
            if pk.id == self._BULBA:
                return 150 - g
            if pk.id == self._OGER:
                return (100 - g) if g < 3 else 10
            return 5
        return [max(atts, key=sc)]

    def decide_acquire(self, ctx):
        # Teal Dance target (ATTACH_TO): same concentration order.
        opt = ctx.sel.option
        mine = [i for i, o in enumerate(opt)
                if o.cardId is None and o.playerIndex == ctx.mi
                and o.area in (AreaType.ACTIVE, AreaType.BENCH)]
        if mine:
            def sc(i):
                pk = ctx.field_pk(opt[i])
                if pk is None:
                    return -1
                g = self._g(pk)
                if pk.id == self._MEGA:
                    return (300 - g) if g < 4 else 10
                if pk.id in (self._IVY, self._BULBA):
                    return 180 - g
                if pk.id == self._OGER:
                    return 120 - g
                return 2
            best = max(mine, key=sc)
            return [best] + [i for i in range(len(opt)) if i != best]
        return super().decide_acquire(ctx)

    def rule_promote_focus(self, ctx):
        # armed Mega on the bench + a front that cannot attack -> pivot in
        # (Solar Transfer already drained the outgoing body, so the retreat
        # discard costs nothing of value).
        a = ctx.me.active
        if a is None or a.id == self._MEGA or ctx.attacks:
            return None
        m = next((v for v in ctx.me.bench if v.id == self._MEGA), None)
        if m is None or self._g(m.pk) < 4:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and self._in_bucket(c, _SWITCH_CARDS):
                return [i]
        if ctx.retreat_idx is not None and not ctx.state.retreated:
            return [ctx.retreat_idx]
        return None

    def decide_retreat(self, ctx):
        # the Mega never retreats (380HP + Jungle heal = the tank IS the plan)
        a = ctx.me.active
        if a is not None and a.id == self._MEGA:
            return None
        return super().decide_retreat(ctx)


class FocusL2(BasePolicy):
    """Shared 'focus' doctrine (oracle: legacy _kanga_focus, pipeline v2.1):
    concentrate energy on ONE focus body (priority-ordered), pivot it in when
    loaded, and FORCE its big attacks — display-0 scalers included (the base
    refuses display-0; the ethan Lava lesson generalized)."""
    FOCUS = ()                 # priority-ordered body cids
    BIG = frozenset()          # attack ids to force when offered
    NEED = {}                  # cid -> energies required (default cheapest)
    ladder = ("rule_promote_focus", "rule_lock_rotate")

    def _focus_in_play(self, ctx):
        for cid in self.FOCUS:
            v = next((v for v in ctx.me.inplay() if v.id == cid), None)
            if v is not None:
                return v
        return None

    def _need(self, cid):
        return self.NEED.get(cid, max(1, _cheapest_cost(_CARDS.get(cid))))

    def decide_energy_target(self, ctx):
        if ctx.state.energyAttached:
            return None
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option
        rank = {cid: len(self.FOCUS) - i for i, cid in enumerate(self.FOCUS)}

        def sc(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return -100
            n = len(getattr(pk, "energies", None) or [])
            r = rank.get(pk.id, 0)
            if r and n < self._need(pk.id):
                return 100 * r - n
            return 5 - n
        best = max(atts, key=sc)
        if sc(best) <= 5:
            return super().decide_energy_target(ctx)
        return [best]

    def rule_promote_focus(self, ctx):
        # v2.1 layer fix: an UNARMED focus in front used to BLOCK this rule
        # (a.id in FOCUS bailed out) while the armed twin waited on the bench.
        # Now: pivot whenever the front cannot act and a bench focus is READY
        # (can-act = cheapest attack, not the big-attack NEED — Black Kyurem
        # fronts at 3 for Ice Age 90 tempo while charging Black Frost).
        a = ctx.me.active
        if a is None or ctx.attacks:
            return None
        armed = next((v for v in ctx.me.bench if v.id in self.FOCUS and v.ready),
                     None)
        if armed is None:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and self._in_bucket(c, _SWITCH_CARDS):
                return [i]
        if ctx.retreat_idx is not None and not ctx.state.retreated:
            return [ctx.retreat_idx]
        return None

    def decide_active(self, ctx, mode="setup"):
        # Focus-deck promotion (v2.1 oracle diff: we fronted Chien-Pao/Meowth-ex
        # walls and DONATED the prize race; legacy fronts cheap 1-prize bodies
        # while the focus charges on the bench): armed focus > cheap wall >
        # part-charged focus > multi-prize tech LAST.
        opt = ctx.sel.option

        def score(i):
            cid = self._opt_pk_id(ctx, opt[i])
            if cid is None:
                return -1
            pk = ctx.field_pk(opt[i])
            n = len(getattr(pk, "energies", None) or []) if pk is not None else 0
            if cid in self.FOCUS:
                can_act = (pk is not None
                           and n >= max(1, _cheapest_cost(_CARDS.get(cid))))
                return (500 + n) if can_act else 90 + 15 * n
            c = _CARDS.get(cid)
            prizes = 3 if (c and c.megaEx) else 2 if (c and c.ex) else 1
            return (150 - 40 * (prizes - 1)) + 5 * n
        return sorted(range(len(opt)), key=score, reverse=True)

    def rule_lock_rotate(self, ctx):
        # ARMED focus that cannot attack = a self-lock turn (Eon Blade / Bright
        # Horns class — "can't use attacks next turn"). Rotate out to a body
        # that CAN act instead of walling the lock turn (Zoroark-250 lesson,
        # generalized). Retreat cost is usually free here (Skyliner / rc0).
        a = ctx.me.active
        if a is None or a.id not in self.FOCUS or ctx.attacks:
            return None
        if len(getattr(a.pk, "energies", None) or []) < self._need(a.id):
            return None                    # unarmed, not locked
        if not any(v.ready or len(getattr(v.pk, "energies", None) or [])
                   >= self._need(v.id) for v in ctx.me.bench if v is not None):
            return None
        if ctx.retreat_idx is not None and not ctx.state.retreated:
            return [ctx.retreat_idx]
        return None

    def decide_attack(self, ctx):
        if ctx.attacks:
            big = [i for i in ctx.attacks if ctx.sel.option[i].attackId in self.BIG]
            if big:
                return [max(big, key=lambda i: self._opt_atk_dmg(ctx, i))]
        return super().decide_attack(ctx)


class MegaGardevoirL2(FocusL2):
    """mega_gardevoir (oracle: legacy focus=Latias; v2.1 adds the real Mega
    Symphonia perception). Latias ex Eon Blade 200 (PPC) is the clean focus;
    Mega Gardevoir ex Mega Symphonia (1079, cost 1 P!) = 50 x EVERY {P} on our
    board — arm the board, not just the active."""
    FOCUS = (747, 184, 751)          # Mega Gardevoir > Latias > Xerneas
    BIG = frozenset({243, 1079, 1084})
    NEED = {747: 1, 184: 3, 751: 3}
    _GARDE, _LATIAS = 747, 184

    def _total_p(self, ctx):
        n = 0
        for v in ctx.me.inplay():
            e = getattr(v.pk, "energies", None) or []
            n += sum(1 for x in e if x == EnergyType.PSYCHIC)
        return n

    def _expected_dmg(self, ctx):
        a = ctx.me.active
        if a is None:
            return ctx.my_active_dmg
        if a.id == self._GARDE:
            return 50 * self._total_p(ctx)
        if a.id == self._LATIAS:
            return 200
        return ctx.my_active_dmg

    def decide_energy_target(self, ctx):
        # once a Mega Gardevoir is in play with its 1 P, EVERY P on our board
        # feeds Symphonia — spread beats concentration (unique to this deck).
        g = next((v for v in ctx.me.inplay() if v.id == self._GARDE), None)
        if g is not None and any(x == EnergyType.PSYCHIC
                                 for x in (getattr(g.pk, "energies", None) or [])):
            return super(FocusL2, self).decide_energy_target(ctx)
        return super().decide_energy_target(ctx)


class MegaDiancieL2(FocusL2):
    """mega_diancie (oracle: legacy _kanga_focus[184, 766]). Latias ex Eon
    Blade 200 focus; Mega Diancie ex Garland Ray (1110, PP, display-0
    discard-up-to-2 scaler) forced like the Lava lesson."""
    FOCUS = (184, 766)
    BIG = frozenset({243, 1110})
    NEED = {184: 3, 766: 2}

    def _expected_dmg(self, ctx):
        a = ctx.me.active
        if a is None:
            return ctx.my_active_dmg
        if a.id == 184:
            return 200
        if a.id == 766:
            return 130                    # Garland Ray conservative floor
        return ctx.my_active_dmg



class BlackKyuremL2(FocusL2):
    """black_kyurem (oracle: legacy _kanga_focus[179] min_energy=4): load 4 for
    Black Frost 250, never stall on the cheap attack."""
    FOCUS = (179, 140)         # Fez = secondary sniper
    BIG = frozenset({238, 183})
    NEED = {179: 4, 140: 3}

    def decide_energy_target(self, ctx):
        # after the Kyurem line + Fez are served, 1 energy on Dunsparce turns a
        # hard wall into a LIVE pivot (Trading Places, 1C self-switch).
        r = super().decide_energy_target(ctx)
        if r is not None:
            return r
        atts = self._energy_attach_opts(ctx)
        if not atts or ctx.state.energyAttached:
            return r
        opt = ctx.sel.option
        for i in atts:
            pk = ctx.field_pk(opt[i])
            if pk is not None and pk.id == 305 and not (pk.energies or []):
                return [i]
        return r


class MegaLatiasL2(FocusL2):
    """mega_latias (oracle: legacy _kanga_focus[756,184])."""
    FOCUS = (756, 184)
    BIG = frozenset({1092, 453, 243, 1090})
    NEED = {184: 3}


class SolrockLunatoneMixin:
    """Solrock (676) + Lunatone (675) are a PACKAGE, and the stock engine mis-reads both.

    Measured on the scouted top mega_lucario list (MegaLucarioTRL2), and the traps are a
    property of the CARDS, so they bite any deck running the pair:
      * Lunatone's Lunar Cycle ("if you have Solrock in play, discard a Basic {F} from
        hand -> DRAW 3") is a DRAW ENGINE, but its attack is a feeble 50, so _setup_score
        (which ranks bodies by printed damage) benched it on turn **5.6** where the live
        1059.7 player has it down by turn **1.2** and fires Lunar Cycle 0.78x per turn.
      * Solrock reads as "70 damage for one energy" -- the best rate in the deck -- so the
        engine stands it in the Active spot, but **Cosmic Beam does NOTHING unless
        Lunatone is on the Bench**. Measured: Solrock active 39.6% of turns, Lunatone
        benched in only 47% of those games, **64% of Solrock's attacks whiffed for zero**.

    Mix in AHEAD of the policy base and add "rule_pair_lunatone" to the ladder.
    """
    _SOLROCK, _LUNATONE = 676, 675
    _PAIR_ON = True

    def _has_lunatone_benched(self, ctx):
        return any(v.id == self._LUNATONE for v in ctx.me.bench)

    def _solrock_live(self, ctx):
        """Is Solrock's attack actually functional right now?"""
        return self._has_lunatone_benched(ctx)

    def rule_pair_lunatone(self, ctx):
        """Get the PAIR down immediately -- neither half works without the other.

        Lunar Cycle needs **Solrock in play**; Cosmic Beam needs **Lunatone on the Bench**.
        An earlier version reached only for Lunatone, on the theory that bench_target=5
        would bring Solrock along. It does not reliably: measured on the TR list, 117 of
        the 301 decisions where Lunatone was out but Solrock was NOT had **a Solrock
        sitting in hand, unplayed** -- and on the rebuilt mega_zygarde, Lunatone reached
        57% of decisions while Solrock stalled at 32%, i.e. the draw engine was switched
        off for most of the game it was supposedly online. Reach for whichever half is
        missing."""
        if not self._PAIR_ON or len(ctx.me.bench) >= 5:
            return None
        want = []
        if not self._has_lunatone_benched(ctx):
            want.append(self._LUNATONE)         # the draw engine: first priority
        if not any(v.id == self._SOLROCK for v in ctx.me.inplay()):
            want.append(self._SOLROCK)          # without it Lunar Cycle cannot fire at all
        if not want:
            return None
        opt = ctx.sel.option
        for cid in want:
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c and c.cardId == cid:
                    return [i]
        return None

    def _bench_score(self, cid):
        """BENCH ranking only -- deliberately NOT _setup_score.

        Bumping _setup_score got Lunatone benched sooner but it also feeds decide_active,
        so the engine started PROMOTING Lunatone and the Mega's first appearance slipped
        from turn 5.7 to 12.1. Splitting the hooks lets Lunatone be the first body we want
        BEHIND us without ever making it a candidate to stand in front."""
        s = super()._bench_score(cid)
        if cid == self._LUNATONE:
            s += 600            # the draw engine
        elif cid == self._SOLROCK:
            s += 200            # 70 off one energy, ignores Weakness/Resistance
        return s


class MegaZygardeL2(SolrockLunatoneMixin, FocusL2):
    """mega_zygarde, REBUILT to the real City League list (Kanagawa 12th, jp/58589).

    The old build was a pasted-staple box: Buddy-Buddy Poffin x4 (**dead** -- it fetches
    Basics with HP<=70 and this deck's smallest is Binacle 80; verified 0 search menus in
    31 plays), plus Hilda/Dawn/Pokegear, a lone Fezandipiti ex, and **no Solrock/Lunatone
    at all**. Three real 2026-Standard lists run ZERO Poffin and ZERO Hilda: the dead card
    was the symptom, the missing engine was the disease.

    The real deck is a {F} loop:
      * **Barbaracle (1052) Stone Arms** -- ability: attach a Basic {F} from HAND to any
        {F} Pokemon, once per turn per copy. This is the accelerator, and there are 4.
      * **Lunatone/Solrock** -- Lunar Cycle discards a Basic {F} to DRAW 3.
      * **Tarragon (1238) x4** -- put up to 4 {F} Pokemon / Basic {F} Energy back from the
        DISCARD to hand: it refuels exactly what Lunar Cycle just discarded.
      * **Wally's Compassion (1229)** -- heal the Mega AND return its Energy to hand,
        where Stone Arms re-attaches it.
      * **Mega Zygarde ex Gaia Wave (1525)** = 200 for {F}{F}{F}, -30 taken next turn.

    The old config named **140 (Fezandipiti ex) and its attack 183 (Cruel Arrow)** in
    FOCUS/NEED/BIG -- both cut from the list, so it pointed at cards that no longer exist.
    """
    FOCUS = (1056,)            # Mega Zygarde ex is the only prize-taker in this build
    BIG = frozenset({1525})    # Gaia Wave 200; Nullifying Zero (1526) is a 5-energy coin flip
    NEED = {1056: 3}
    bench_target = 5           # Lunatone + Solrock + the Binacle line all want to be down
    _ZYG, _BARBARACLE, _WALLY = 1056, 1052, 1229
    ladder = ("rule_pair_lunatone", "rule_wally") + FocusL2.ladder

    def rule_wally(self, ctx):
        """Wally's Compassion (1229) -- full-heal the 310HP Mega. Was DEAD: 0 plays in
        183 offers, because decide_trainer has no branch for it.

        mega_lucario's version of this rule demands a LOADED BACKUP MEGA on the bench,
        since Wally's strips all the healed Pokemon's Energy to hand and that deck cannot
        put it back quickly. **This deck can**: Barbaracle's Stone Arms attaches a Basic
        {F} from HAND every turn, and there are 4 of them -- the energy Wally's returns is
        exactly what Stone Arms re-attaches. That is why the human list runs both.
        So gate on Barbaracle in play instead of on a spare Mega.

        Still refuse when we can just KO back this turn: a prize beats a heal."""
        if ctx.state.supporterPlayed:
            return None
        a = ctx.me.active
        if a is None or a.id != self._ZYG:
            return None
        if (a.max_hp - a.hp) < 150:
            return None                    # not worth the Supporter yet
        oa = ctx.opp.active
        if oa is not None and ctx.my_active_dmg >= oa.hp:
            return None                    # closing -> swing, don't heal
        if not any(v.id == self._BARBARACLE for v in ctx.me.inplay()):
            return None                    # nothing to re-attach the stripped energy
        opt = ctx.sel.option
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c and c.cardId == self._WALLY:
                return [i]
        return None

    def decide_retreat(self, ctx):
        a = ctx.me.active
        if a is not None and a.id == self._ZYG:
            return None                    # the 310HP tank never ping-pongs its energy away
        return super().decide_retreat(ctx)


class CubchooL2(ToolboxPolicy):
    """cubchoo_control (oracle: legacy): Cubchoo Snotted Up (716) attack-locks
    the Defending Pokemon — force it every turn; Boss a loaded THREAT into the
    lock first (locking a wall is worthless).

    ACCOMPANYING FLUTE COMBO (1091): "what every lock deck ever wanted". When no good
    lock target exists, Flute force-benches a weak Basic onto the OPPONENT, then Boss's
    Orders drags that 0-energy LIABILITY active and Snotted Up + Gravity Gemstone hard-lock
    it (0 energy can't pay the raised retreat). A 0-energy forced Basic is a STRICTLY
    better lock than their loaded attacker (which could still retreat), so the Boss target
    below prefers the lowest-energy body, and `threat` now also counts 0-energy liabilities."""
    _SNOT, _BOSS, _FLUTE = 716, 1182, 1091

    def _hold(self, ctx, cid):
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and c.cardId == cid:
                return i
        return None

    def decide_trainer(self, ctx):
        opt = ctx.sel.option
        snot_ready = any(opt[i].attackId == self._SNOT for i in ctx.attacks)
        if snot_ready and not ctx.state.supporterPlayed and ctx.opp is not None:
            boss_i = self._hold(ctx, self._BOSS)
            oa = ctx.opp.active
            oa_e = oa.energy_count if oa is not None else 0
            # a benched body worth locking: a loaded threat, OR a 0-energy liability
            # (a Flute-forced Basic that will be retreat-locked once Snotted).
            has_liability = any(v.energy_count == 0 for v in ctx.opp.bench)
            has_threat = any(v.energy_count >= 2 and v.energy_count > oa_e
                             for v in ctx.opp.bench)
            if boss_i is not None and (has_threat or (has_liability and _FLUTE_COMBO)):
                return [boss_i]
            # no lock target yet: MAKE one with Accompanying Flute (needs Boss in hand to
            # follow up this same turn, and room on the opp bench). Only when the current
            # active is worth swapping out (has energy) so we don't waste it on an empty.
            if (_FLUTE_COMBO and boss_i is not None and len(ctx.opp.bench) < 5
                    and oa_e >= 1):
                flute_i = self._hold(ctx, self._FLUTE)
                if flute_i is not None:
                    return [flute_i]
        return super().decide_trainer(ctx)

    def decide_target(self, ctx, kind):
        if kind == "gust" and _FLUTE_COMBO:
            opt = ctx.sel.option
            cand = [(i, ctx.opp_pokemon_at(opt[i])) for i in range(len(opt))]
            cand = [(i, pk) for i, pk in cand if pk is not None]
            if cand:
                # best LOCK target: fewest energy (can't pay retreat once Snotted +
                # Gravity Gemstone), then lowest HP. Locking a 0-energy liability is the
                # hardest lock in the deck.
                cand.sort(key=lambda ip: (len(ip[1].energies or []), ip[1].hp))
                return [i for i, _ in cand]
        return super().decide_target(ctx, kind)

    def decide_attack(self, ctx):
        snot = next((i for i in ctx.attacks
                     if ctx.sel.option[i].attackId == self._SNOT), None)
        if snot is not None:
            return [snot]
        return super().decide_attack(ctx)


class ConfigL2(FocusL2):
    """PIPELINE-GENERATED line config (P0.5: tools/p05_deckconfig.py) — ONE
    generic class for any deck. No hand-written per-deck code: the pipeline
    derives FOCUS/NEED/BIG, stadiums, bench target and discard-fuel from the
    card DB, and P1-P4 tune/accept the config as a bundle."""

    def __init__(self, deck, profile=None):
        super().__init__(deck, profile)
        cfg = (profile or {}).get("line") or {}
        self.FOCUS = tuple(cfg.get("focus", ()))
        self.NEED = {int(k): v for k, v in (cfg.get("need") or {}).items()}
        self.BIG = frozenset(cfg.get("big", ()))
        self._stadiums = tuple(cfg.get("stadiums", ()))
        self._bench_wide = cfg.get("bench_target")
        self._fuel = frozenset(cfg.get("discard_fuel", ()))
        if self._stadiums:
            self.ladder = self.ladder + ("rule_stadium",)
        if self._combo:
            self.ladder = ("rule_combo_pivot",) + self.ladder

    def rule_stadium(self, ctx):
        if ctx.state.stadiumPlayed:
            return None
        if ctx.state.stadium and any(x.id in self._stadiums for x in ctx.state.stadium):
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and c.cardId in self._stadiums:
                return [i]
        return None

    @property
    def bench_target(self):
        return self._bench_wide or BasePolicy.bench_target

    @bench_target.setter
    def bench_target(self, v):                     # keep class-attr compatibility
        self._bench_wide = v

    def _opp_active_counters(self, ctx):
        a = ctx.opp.active
        return (a.max_hp - a.hp) // 10 if a is not None else None

    def _window_reach(self, ctx):
        """Can the opponent's active be brought to EXACTLY n this turn?"""
        n = self._combo.get("n")
        c = self._opp_active_counters(ctx)
        if not n or c is None or c > n:
            return None
        if c == n:
            return 0
        movable = sum(3 for v in ctx.me.inplay()
                      if v.id in self._support_e
                      and len(getattr(v.pk, "energies", None) or []) >= 1)
        stock = sum((v.max_hp - v.hp) // 10 for v in ctx.me.inplay())
        return (n - c) if (n - c) <= min(movable, stock) else None

    def rule_combo_pivot(self, ctx):
        # CONVERSION rule (v2.3 rev-3): when the exact-N window is open or
        # reachable THIS turn, put the armed finisher in front NOW — via a
        # self-switch ability (free), a Switch card, or retreat. Abilities
        # (movers) then run before the attack step: pivot -> set N -> finish.
        fin = self._combo.get("finisher")
        if not fin or self._window_reach(ctx) is None:
            return None
        a = ctx.me.active
        if a is not None and a.id == fin:
            return None
        if not any(v.id == fin and v.ready for v in ctx.me.bench):
            return None
        for i in ctx.abilities:
            pk = ctx.field_pk(ctx.sel.option[i])
            c2 = _CARDS.get(pk.id) if pk is not None else None
            if c2 and any(_RE_AB_SWITCH.search(sk.text or "")
                          for sk in (c2.skills or [])):
                return [i]
        for i in ctx.plays:
            cd = ctx.hand_card(ctx.sel.option[i])
            if cd and self._in_bucket(cd, _SWITCH_CARDS):
                return [i]
        if ctx.retreat_idx is not None and not ctx.state.retreated:
            return [ctx.retreat_idx]
        return None

    def decide_attack(self, ctx):
        # WINDOW DISCIPLINE (v2.3 rev-3): while building toward exactly n on a
        # body we cannot KO outright, never chip its counters past n. Prefer the
        # EXACT setter (damage == 10*(n-c)), then a bench snipe; chip only as
        # the last resort (anti-passivity still holds — we always act).
        n = self._combo.get("n")
        if n and ctx.attacks:
            opp = ctx.opp.active
            if opp is not None:
                c = (opp.max_hp - opp.hp) // 10
                best = self._best_attack_opt(ctx)
                if self._opt_atk_dmg(ctx, best) < opp.hp and c < n:
                    setter = snipe = None
                    for i in ctx.attacks:
                        at = _ATTACKS.get(ctx.sel.option[i].attackId)
                        if at is None:
                            continue
                        if (at.damage or 0) == 10 * (n - c) and not _RE_EACH.search(at.text or ""):
                            setter = i
                        elif _RE_SNIPE.search(at.text or "") and not (at.damage or 0):
                            snipe = i
                    if setter is not None:
                        return [setter]
                    if snipe is not None:
                        return [snipe]
        return super().decide_attack(ctx)

    def decide_count(self, ctx):
        # exact_counters combo (pipeline v2.3, e.g. Terminal Period "exactly N"):
        # when MOVING damage counters to the opponent, move exactly what brings
        # the active to N — never overshoot past the instant-KO window.
        n = self._combo.get("n")
        if n and ctx.sel.context in (SelectContext.DAMAGE_COUNTER_COUNT,
                                     SelectContext.REMOVE_DAMAGE_COUNTER_COUNT):
            c = self._opp_active_counters(ctx)
            if c is not None and c < n:
                want = n - c
                opt = ctx.sel.option
                exact = [i for i in range(len(opt)) if (opt[i].number or 0) <= want]
                if exact:
                    return [max(exact, key=lambda i: opt[i].number or 0)]
        return super().decide_count(ctx)

    def decide_target(self, ctx, kind):
        # counter-move DESTINATION: the opponent's active while it is short of N
        # (never past); once at N, stock a FRESH bench body (Boss + finisher later).
        n = self._combo.get("n")
        if (n and kind in ("spread", "effect")
                and ctx.sel.context == SelectContext.DAMAGE_COUNTER):
            opt = ctx.sel.option
            c = self._opp_active_counters(ctx)

            def sc(i):
                o = opt[i]
                pk = ctx.opp_pokemon_at(o)
                if pk is None:
                    return -1
                is_active = o.area == AreaType.ACTIVE
                k = (pk.maxHp - pk.hp) // 10
                if is_active:
                    return 300 if (c is not None and c < n) else 5
                return 100 - abs(n - 1 - k)       # bench stock toward N
            return sorted(range(len(opt)), key=sc, reverse=True)
        return super().decide_target(ctx, kind)

    def decide_discard(self, ctx):
        # discard-fuel combo edge (PP Up / Wondrous Patch class: an "attach from
        # your discard pile" trainer turns discarded energy into acceleration).
        # (Was keyed on the raw cardId -- 188/188 options blind, so `cid in self._fuel`
        # never matched and this whole override was a no-op.)
        if not self._fuel:
            return super().decide_discard(ctx)
        opt = ctx.sel.option
        hand_fuel = sum(1 for x in (ctx.me_ps.hand or []) if x.id in self._fuel)

        def key(i):
            cid = self._opt_card_id(ctx, opt[i])
            if cid in self._fuel and hand_fuel >= 2:
                return _SHED
            return self._card_need(ctx, _CARDS.get(cid))
        return sorted(range(len(opt)), key=key)



class MegaLucarioTRL2(SolrockLunatoneMixin, MegaLucarioL2):
    """mega_lucario on the SCOUTED top list (Team_Rocket, live 1059.7): 16 Pokemon /
    10 basics -- Riolu->Mega Lucario ex PLUS Solrock+Lunatone and Makuhita->Hariyama.

    P0 measured this list at 39% under the stock engine vs 72% for our 8-Pokemon list,
    and P1 found why: the engine reads Solrock as "70 damage for 1 energy" (the best
    rate in the deck), so Solrock occupies the Active spot 39.6% of the time -- more
    than Mega Lucario -- but **Solrock's attack does NOTHING unless Lunatone is on the
    BENCH**, and generic decide_bench just plays the first Basic it finds, so Lunatone
    was benched in only 47% of Solrock-active games. 64% of Solrock's attacks whiffed
    for zero. Fixes: pair Lunatone with Solrock, and never present a blank Solrock as
    an attacker.
    """
    _MAKUHITA, _HARIYAMA_TR = 673, 674   # _SOLROCK / _LUNATONE come from SolrockLunatoneMixin
    # 16 Pokemon / 10 basics: this list wants the bench FULL (Lunatone + Solrock + Riolu
    # + the Makuhita line). The inherited bench_target=3 stopped development early --
    # measured: on 117 of the 301 decisions where Lunatone was out but Solrock was NOT,
    # a Solrock was sitting in HAND unplayed. Lunatone escaped this only because
    # rule_pair_lunatone bypasses the target, which is exactly why Lunatone reached 77%
    # of turns while Solrock stalled at 52%.
    bench_target = 5
    _PAIR_ON = True
    # MEASURED -4.5pt (43.4 -> 38.9). Keeping the Mega off the Active spot until it can
    # KO looks right from "your three-prize Mega Lucarios wait for the perfect moment to
    # strike", but that quote is about MEGA BRAVE, not about benching the Mega: the JP
    # guides are explicit that you lead with Aura Jab (130 for one energy) precisely
    # BECAUSE it recycles 3 Basic {F} from the discard onto the bench. Shielding the Mega
    # switches off the deck's own accelerator. Left in, defaulted OFF, as a documented
    # dead end -- do not re-derive it.
    _PRIZE_TRADE_ON = False
    _SWITCH_CARD = 1123
    ladder = ("rule_pair_lunatone", "rule_prize_shield", "rule_cape", "rule_wall_gust")

    # ---- prize-trade doctrine -------------------------------------------------
    # Mega Lucario ex is megaEx = it concedes **3 of the opponent's 6 prizes**; every
    # other body in this list concedes 1. The published plan is explicit: "play off a
    # single-prize board of Solrock and Lunatone, forcing awkward prize trades, while
    # your three-prize Mega Lucarios wait for the perfect moment to strike."
    #
    # The engine has the exact opposite instinct. _target_score weights the OPPONENT's
    # bodies by _prize_value*1000, but _my_attacker_score scores OUR bodies by damage
    # and then ADDS +100 for ex/megaEx -- it is rewarded for shoving our 3-prize body
    # into the firing line. Nothing anywhere prices what WE concede. So decide_active
    # always promotes the Mega (setup_score ~1370) over a Solrock (~70), and each
    # trade hands over half the game.
    #
    # Solrock is the intended early attacker for a reason: 70 for ONE energy, and its
    # damage "isn't affected by Weakness or Resistance" -- while the Mega has a 2x
    # Psychic weakness that the live meta (Alakazam) exploits to one-shot it.

    def _dmg_if_promoted(self, ctx, pk):
        """Roughly, what could this body hit the opponent's Active for right now?"""
        if pk is None:
            return 0
        c = _CARDS.get(pk.id)
        if not c:
            return 0
        e = len(pk.energies)
        best = 0
        for a in (c.attacks or []):
            at = _ATTACKS.get(a)
            if not at or not at.damage:
                continue
            if _attack_cost(a) <= e:
                best = max(best, at.damage)
        if pk.id == self._SOLROCK and not self._has_lunatone_benched(ctx):
            return 0                       # blank without Lunatone benched
        oa = ctx.opp.active
        if oa is not None and best and pk.id != self._SOLROCK:
            # Solrock is excluded on purpose: its text says the damage "isn't affected
            # by Weakness or Resistance".
            oc = _CARDS.get(oa.id)
            if oc is not None and oc.weakness == c.energyType:
                best *= 2
        return best

    def _one_prize_ready(self, ctx, i):
        """Is option i a 1-prize body that can actually attack right now?"""
        opt = ctx.sel.option
        cid = self._opt_pk_id(ctx, opt[i])
        c = _CARDS.get(cid)
        if not c or c.ex or c.megaEx:
            return False
        pk = ctx.field_pk(opt[i])
        return pk is not None and self._dmg_if_promoted(ctx, pk) > 0

    def decide_active(self, ctx, mode="setup"):
        order = super().decide_active(ctx, mode)
        if not self._PRIZE_TRADE_ON or not order:
            return order
        opt = ctx.sel.option
        top = opt[order[0]]
        if self._opt_pk_id(ctx, top) != self._LUCARIO:
            return order                      # not about to expose the Mega anyway
        # The Mega goes up only to CLOSE: it KOs the opponent's Active now, or the
        # game is already in reach. Otherwise a 1-prize body takes the hit instead.
        oa = ctx.opp.active
        mega_dmg = self._dmg_if_promoted(ctx, ctx.field_pk(top))
        closing = (oa is not None and mega_dmg >= oa.hp) or ctx.me.prizes_left <= 2
        if closing:
            return order
        alt = [i for i in order if self._one_prize_ready(ctx, i)]
        if not alt:
            return order                      # nothing else can attack -> Mega it is
        return alt + [i for i in order if i not in alt]

    def rule_prize_shield(self, ctx):
        """Switch the 3-prize Mega out when it is exposed and not closing.

        Same doctrine as decide_active, but mid-turn: if the Mega is Active, cannot KO,
        and a functional 1-prize Solrock is benched, spend a Switch to put the cheap
        body in front. Retreat is not an option worth forcing (it costs 2 energy that
        Aura Jab just spent recycling)."""
        if not self._PRIZE_TRADE_ON:
            return None
        a = ctx.me.active
        if a is None or a.id != self._LUCARIO:
            return None
        oa = ctx.opp.active
        if oa is not None and ctx.my_active_dmg >= oa.hp:
            return None                       # closing -> swing, don't run
        if ctx.me.prizes_left <= 2:
            return None                       # endgame: the Mega is the win-con
        sol = next((v for v in ctx.me.bench
                    if v.id == self._SOLROCK and self._has_lunatone_benched(ctx)
                    and len(v.pk.energies) >= 1), None)
        if sol is None:
            return None
        opt = ctx.sel.option
        for i in ctx.plays:
            c = ctx.hand_card(opt[i])
            if c and c.cardId == self._SWITCH_CARD:
                return [i]
        return None

    def decide_acquire(self, ctx):
        """Search (Dusk Ball / Fighting Gong / Poke Pad) should fetch the engine first.

        Fighting Gong searches a Basic {F} Energy *or* a Basic Fighting Pokemon, and the
        guides call it the cleanest way to switch the Solrock+Lunatone package on early.
        If Lunatone is not in play yet, take it over any other card."""
        if self._PAIR_ON and not self._has_lunatone_benched(ctx):
            opt = ctx.sel.option
            for i in range(len(opt)):
                # _opt_card_id (not _opt_pk_id) -- deck-search options carry cardId=None
                # and only resolve through sel.deck[index].
                if self._opt_card_id(ctx, opt[i]) == self._LUNATONE:
                    return [i]
        return super().decide_acquire(ctx)

    def decide_attack(self, ctx):
        # Never swing with a blank Solrock: its 70 is conditional on Lunatone being
        # benched, but _my_attacker_score/decide_attack read the printed damage and
        # would happily attack for zero (64% of its swings did exactly that).
        a = ctx.me.active
        if a is not None and a.id == self._SOLROCK and not self._solrock_live(ctx):
            return None
        return super().decide_attack(ctx)


# --------------------------------------------------------------------------- #
# decide_bench ordering. MEASURED A/B (same process/seeds, n=210/deck):
#            legacy(hand-order) -> scored
#   mega_lucario_tr   44.3 -> 44.3   (no change AT ALL, even though the scored order got
#                                     Lunatone down on turn 3.7 instead of 6.6)
#   mega_lucario_hilda 73.8 -> 74.3  mega_gardevoir 19.5 -> 22.4  cynthia 40.5 -> 44.3
#   marnie_grimmsnarl 61.3 -> 51.2   (-10.1 REGRESSION)
# No beneficiary + a real regression => keep the legacy order. Left switchable because
# _bench_score itself (separate from _setup_score) is sound and may pay off elsewhere.
_BENCH_HAND_ORDER = True
# A/B switch for the spare-ex bench guard (BasePolicy._is_spare_ex): do not park a second
# copy of a multi-prize Basic ex/megaEx on the bench while one is already in play.
_SPARE_EX_GUARD = True
# A/B switch: extend the spare-ex doctrine to the TO_BENCH SUB-select (deck-search
# benching, e.g. Nest Ball / Poffin), where _is_spare_ex never ran. See
# BasePolicy._is_spare_ex_sub.
#
# OFF -- the doctrine is sound but the situation essentially never arises. Measured over
# the 15 decks that run 2+ copies of a benchable Basic ex/megaEx, 40 games each vs
# mega_lucario (600 games): TO_BENCH fired 168 times (0.28/game) and the guard declined
# **2 times**. Wiring verified (mega_diancie 1, mega_zygarde 1), so this is a frequency
# result, not a plumbing bug.
#
# The reason is structural, not a threshold to tune: deck-search benching happens EARLY,
# when the board is still empty -- so neither "a copy is already in play" nor "board >= 2"
# holds. By the time spare copies are a liability, the deck is no longer searching.
# metagross is the clean case: most TO_BENCH of any deck (38) and 3+2 benchable ex in the
# list, yet 0 declines. At 2 events / 600 games no A/B can resolve this above noise.
_SPARE_EX_BENCH_SUB = False
# A/B switch: does MegaLucarioL2's Xerosic gust-KO carve-out require a gust card IN HAND?
# Off == the old "a KO-able benched body exists" test, which blocked Xerosic 76% vs
# alakazam (its bench is all small bodies).
_XERO_GUST_NEEDS_CARD = True

# A/B switch for the deck-search resolution fix (see _opt_card_id). True = legacy blind pick.
_SEARCH_BLIND = False
_RESOLVE_HAND = True
# A/B switch for the generic Switch-item free pivot (BasePolicy._switch_pivot). Off ==
# the legacy behaviour where the base had no branch and Switch was a dead slot in 11 decks.
_SWITCH_PIVOT = os.environ.get("ENGINE_NO_SWITCH_PIVOT") != "1"
# A/B switch for the hand-disruption rule (BasePolicy._disrupt_play). Off == legacy, where
# no rule played Xerosic's Machinations-class cards (dead in 7 decks).
_HAND_DISRUPT = os.environ.get("ENGINE_NO_DISRUPT") != "1"
# A/B for the Round-1 batch of new item rules (heal / deck-recover / energy-denial /
# hand-reset). Each is per-deck opt-in; this flag disables the whole batch for A/B.
_ITEM_RULES = os.environ.get("ENGINE_NO_ITEM_RULES") != "1"
# A/B for the residual idle-turn play of otherwise-dead cards (BasePolicy._residual_play).
_RESIDUAL = os.environ.get("ENGINE_NO_RESIDUAL") != "1"
# A/B for CubchooL2's Accompanying Flute lock combo (Flute play + lowest-energy gust target).
_FLUTE_COMBO = os.environ.get("ENGINE_NO_FLUTE_COMBO") != "1"
# HydrappleL2's situational promotion reorder (wall-break Hydrapple ex / single-prize Tapu
# Bulu). Default OFF: A/B measured -3.0% (dragapult -10, ns_zoroark -9) -- reordering off
# the efficient Ogerpon-primary line to force a Stage-2 or fixed-220 body hurt the good
# matchups more than it helped the bad ones. Opt-in only (ENGINE_HYDRA_SMART=1).
_HYDRA_SMART = os.environ.get("ENGINE_HYDRA_SMART") == "1"
# A/B for SlowkingComboL2 keeping Slowking on the BENCH (greeter fronts) vs fronting it.
_SLOW_GREETER = os.environ.get("ENGINE_NO_SLOW_GREETER") != "1"
class SlowkingComboL2(ConfigL2):
    """slowking: RUN THE SEEK INSPIRATION COMBO (user directive: execute the combo,
    winrate secondary). Slowking (163) Seek Inspiration (213, {P}{C}) discards the top
    card and, if it is a non-Rule-Box Pokemon, uses one of ITS attacks. Academy at Night
    (1248) stacks a payload (Kyurem 144 / Metagross 276 / Zeraora 377 -- all non-Rule-Box)
    on top each turn; Seek then fires it for {P}{C}. The shipped ConfigL2 fronted Mega
    Kangaskhan and attacked with Slowking ZERO times (Seek 0/40 games). This pilot fronts
    Slowking, uses Academy to stack a payload, and forces Seek."""
    _SLOWKING, _SEEK, _ACADEMY = 163, 213, 1248

    def _payloads(self):
        return frozenset(self.profile.get("seek_payloads") or ())

    _SWITCH = 1123

    def decide_trainer(self, ctx):
        # PROTECT: pull a damaged Slowking OUT of the Active spot with Switch (free) before
        # it is Knocked Out -- denies the opponent a prize and PRESERVES its 3 attached
        # Psychic energy (re-attaching that costs the whole deck several turns). Slowking is
        # 120 HP / retreat 3, so Switch is the only escape (Penny/Jet Energy are rotated out
        # of this pool). Only when a greeter is benched to take its place.
        if _SLOW_GREETER:
            a = ctx.me.active
            if (a is not None and a.id == self._SLOWKING and a.hp <= 60
                    and any(v.id in self._GREETERS or v.id in (162, 183) for v in ctx.me.bench)):
                for i in ctx.plays:
                    c = ctx.hand_card(ctx.sel.option[i])
                    if c is not None and c.cardId == self._SWITCH:
                        return [i]
        return super().decide_trainer(ctx)

    def decide_ability(self, ctx):
        # Academy at Night: put a payload on top so Seek copies it THIS turn. Only while
        # Slowking is Active (so Seek is the same-turn follow-up) and a payload is in hand.
        if (self._payloads() and ctx.me.active is not None
                and ctx.me.active.id == self._SLOWKING):
            if {h.id for h in (ctx.me_ps.hand or [])} & self._payloads():
                for i in ctx.abilities:
                    pk = ctx.field_pk(ctx.sel.option[i])
                    c = _CARDS.get(pk.id) if pk is not None else None
                    if (pk is not None and pk.id == self._ACADEMY) or (c and any(
                            "on top of" in (sk.text or "").lower()
                            for sk in (c.skills or []))):
                        return [i]
        return super().decide_ability(ctx)

    def decide_attack(self, ctx):
        # Force Seek Inspiration whenever Slowking is Active and Seek is legal -- it is the
        # deck's win condition, even though its DISPLAY damage is 0 (the payload's is not).
        if ctx.me.active is not None and ctx.me.active.id == self._SLOWKING:
            for i in ctx.attacks:
                if ctx.sel.option[i].attackId == self._SEEK:
                    return [i]
        return super().decide_attack(ctx)

    # GREETERS to sit Active while Slowking (120 HP, one-shot bait) CHARGES on the bench.
    # Latias ex (210 HP) is best -- it also gives Skyliner (our Basics retreat free), so
    # swapping Slowking in on the attack turn costs nothing. Then Meowth ex (170).
    _GREETERS = (184, 1071)

    def _slow_ready(self, ctx):
        return next((v for v in ctx.me.inplay()
                     if v.id == self._SLOWKING and v.energy_count >= 2), None)

    def decide_active(self, ctx, mode="setup"):
        opt = ctx.sel.option
        def pick(ids):
            for i in range(len(opt)):
                pk = ctx.field_pk(opt[i]) if hasattr(ctx, "field_pk") else None
                cid = pk.id if pk is not None else self._opt_pk_id(ctx, opt[i])
                if cid in ids:
                    return [i]
            return None
        if not _SLOW_GREETER:
            r = pick({self._SLOWKING})     # legacy: always front Slowking
            if r:
                return r
            return super().decide_active(ctx, mode)
        # PROMOTING (after a KO / our own retreat): bring Slowking up ONLY if it is charged
        # and ready to Seek this turn; otherwise front a durable greeter and keep charging.
        if mode != "setup" and self._slow_ready(ctx):
            r = pick({self._SLOWKING})
            if r:
                return r
        for tier in (self._GREETERS, (162, 183)):   # Latias/Meowth, then Slowpoke/Smoochum
            r = pick(set(tier))
            if r:
                return r
        return super().decide_active(ctx, mode)

    def decide_retreat(self, ctx):
        # Pivot a READY benched Slowking into the Active spot to Seek: retreat the greeter
        # (free under Latias Skyliner). Only when a payload is in hand so Academy can stack
        # it and the Seek actually lands -- never expose Slowking for nothing.
        if not _SLOW_GREETER or ctx.state.retreated or ctx.me.active is None \
                or ctx.me.active.id == self._SLOWKING:
            return None
        slow = next((v for v in ctx.me.bench
                     if v.id == self._SLOWKING and v.energy_count >= 2), None)
        if slow is None:
            return None
        if not ({h.id for h in (ctx.me_ps.hand or [])} & self._payloads()):
            return None
        if ctx.retreat_idx is not None:
            return [ctx.retreat_idx]
        return super().decide_retreat(ctx)


class HybridSlowkingL2(SlowkingComboL2):
    """MODERN Slowking = Mega Kangaskhan ex CLOCK + Seek Inspiration as a SECONDARY line
    (the current competitive Slowking+Kangaskhan hybrid; the pure-combo build was a weak
    POR-era niche and its power card, Metagross CRI-61 300, is out of this pool). Kangaskhan
    is the reliable win condition; Slowking-Seek fires in the window BEFORE Kangaskhan is
    charged (an early Kyurem Trifrost spread) and whenever Kangaskhan is gone -- the human
    'early Trifrost, then close with Kangaskhan' line. Inherits the Academy stack, Seek-
    force, payload targeting and Switch-protect from SlowkingComboL2."""
    _KANGA = 756

    def decide_active(self, ctx, mode="setup"):
        opt = ctx.sel.option
        def pick(ids):
            for i in range(len(opt)):
                pk = ctx.field_pk(opt[i]) if hasattr(ctx, "field_pk") else None
                cid = pk.id if pk is not None else self._opt_pk_id(ctx, opt[i])
                if cid in ids:
                    return [i]
            return None
        kanga = pick({self._KANGA})
        kanga_ready = any(v.id == self._KANGA and v.ready for v in ctx.me.inplay())
        slow_ready = self._slow_ready(ctx) is not None
        # Kangaskhan is the clock: front it once it can attack, or while Slowking isn't yet
        # charged. Only front Slowking when Kangaskhan is NOT ready AND Slowking IS -> take
        # the early Seek window instead of standing around.
        if kanga and (kanga_ready or not slow_ready):
            return kanga
        if slow_ready:
            r = pick({self._SLOWKING})
            if r:
                return r
        if kanga:
            return kanga
        return ConfigL2.decide_active(self, ctx, mode)

    def decide_retreat(self, ctx):
        # No forced Slowking pivot (Kangaskhan is primary); keep the ping-pong-safe base.
        return ConfigL2.decide_retreat(self, ctx)

class SlowkingLiveL2(ConfigL2):
    """slowking, written from what the #1 and #2 LADDER AGENTS actually do with this list.

    Not inferred -- read off 49 of their games (2,977 decisions) with tools/replay_profile.py.
    The three role assignments below are all INVERTED in the pilot this replaces, which is why
    the scouted decklist measured as a regression: we were running the right cards with the
    wrong plan.

        their attacks        Seek Inspiration 36 | Delightful Kiss 28 | Destined Fight 8
                             Trifrost 7 | Gutsy Swing 4 | Super Psy Bolt 2
                             **Rapid-Fire Combo 1** -- Mega Kangaskhan attacks ONCE in 49 games
        their Active         Slowking 38.4% | Mega Kangaskhan 35.7% | Smoochum 13.2%

    1. **Mega Kangaskhan ex is a WALL, not an attacker.** 300 HP in the Active Spot running
       Run Errand (draw 2) every turn. It attacks essentially never because it holds no
       energy -- and that is a CHOICE, not an accident: the list runs 8-10 energy total and
       all of it belongs to Slowking. Keeping energy off Kangaskhan is what makes it a wall,
       so the fix is in `decide_energy_target`, not in `decide_attack`.

    2. **Smoochum ATTACKS.** Delightful Kiss costs NOTHING (`energies=[]`) and fetches two
       basic {P} from the deck onto the Bench. It is 30% of their attacks and it is the whole
       reason an 8-energy list functions. Our engine would never choose it: the attack shows
       0 damage, the same display trap that hid Seek Inspiration for months.

    3. **Slowking always Seeks.** Super Psy Bolt (120) is used twice in 49 games. Seek is
       {P}{C} and copies Conkeldurr's Gutsy Swing (250) or Annihilape's Impact Blow (160)
       when one is on top -- and mills the deck toward one when it is not.
    """

    _SLOWKING, _SEEK, _PSYBOLT = 163, 213, 214
    _KANGA, _RUN_ERRAND = 756, None
    _SMOO, _KISS = 183, 242
    _ACADEMY, _CIPHER = 1248, 1188
    _GREETERS = (184, 1071)

    def _pay(self):
        return frozenset(self.profile.get("seek_payloads") or ())

    def _energy_on(self, ctx, cid):
        return sum(v.energy_count for v in ctx.me.inplay() if v.id == cid)

    def _starving(self, ctx):
        """Slowking needs {P}{C}; anything less and the deck has no attack worth making."""
        return self._energy_on(ctx, self._SLOWKING) < 2

    def decide_energy_target(self, ctx):
        # Energy belongs to Slowking. Feeding Kangaskhan turns the wall into a 3-energy
        # attacker and starves the Seek engine -- the shipped profile listed Kangaskhan first
        # in main_attackers, so every attach went to it.
        if ctx.attaches and not ctx.state.energyAttached:
            energy_atts = self._energy_attach_opts(ctx)
            for i in energy_atts:
                v = self._target_view(ctx, ctx.sel.option[i])
                if v is not None and v.card is not None and v.card.cardId == self._SLOWKING:
                    return [i]
        return super().decide_energy_target(ctx)

    def decide_ability(self, ctx):
        # Run Errand: free two cards every turn Kangaskhan is Active. This is what the wall
        # is FOR, and it is the deck's whole draw engine.
        a = ctx.me.active
        if a is not None and a.id == self._KANGA:
            for i in ctx.abilities:
                pk = ctx.field_pk(ctx.sel.option[i])
                if pk is not None and pk.id == self._KANGA:
                    return [i]
        # Academy at Night: put a payload on top for Slowking to Seek next.
        if self._pay() and any(v.id == self._SLOWKING for v in ctx.me.inplay()) \
                and ({h.id for h in (ctx.me_ps.hand or [])} & self._pay()):
            for i in ctx.abilities:
                pk = ctx.field_pk(ctx.sel.option[i])
                cid = pk.id if pk is not None else self._opt_card_id(ctx, ctx.sel.option[i])
                if cid == self._ACADEMY:
                    self._sk_placing = True
                    return [i]
        return super().decide_ability(ctx)

    def choose_sub(self, ctx):
        # Only the follow-up of a placer we just chose: SelectContext.TO_DECK also covers
        # ordinary "shuffle this back" costs, and answering those with a payload buries our
        # own ammo.
        if getattr(self, "_sk_placing", False) and ctx.sel.context == SelectContext.TO_DECK:
            self._sk_placing = False
            opt = ctx.sel.option
            for i in range(len(opt)):
                if self._opt_card_id(ctx, opt[i]) in self._pay():
                    return [i]
        return super().choose_sub(ctx)

    def decide_active(self, ctx, mode="setup"):
        opt = ctx.sel.option

        def pick(ids):
            for i in range(len(opt)):
                pk = ctx.field_pk(opt[i])
                cid = pk.id if pk is not None else self._opt_card_id(ctx, opt[i])
                if cid in ids:
                    return [i]
            return None
        # Charged Slowking first (it is the attacker), then Smoochum while energy is short
        # (its attack is free and fetches two), then Kangaskhan to wall and draw.
        if not self._starving(ctx):
            r = pick({self._SLOWKING})
            if r:
                return r
        else:
            r = pick({self._SMOO})
            if r:
                return r
        for tier in ({self._KANGA}, {self._SLOWKING}, set(self._GREETERS), {162}):
            r = pick(tier)
            if r:
                return r
        return super().decide_active(ctx, mode)

    def decide_attack(self, ctx):
        a = ctx.me.active
        if a is None:
            return super().decide_attack(ctx)
        if a.id == self._SMOO:
            # 0 damage, 0 cost, two energy onto the bench. _best_attack scores by damage and
            # would decline it forever.
            for i in ctx.attacks:
                if ctx.sel.option[i].attackId == self._KISS:
                    return [i]
        if a.id == self._SLOWKING:
            for i in ctx.attacks:
                if ctx.sel.option[i].attackId == self._SEEK:
                    return [i]
        return super().decide_attack(ctx)


class DuskNoirL2(ConfigL2):
    """dragapult_dusknoir, with the findings of the Phantom Dive forensics written down.

    WHAT THE FORENSICS SAID (4 opponents x 300 games, tools/dusk_ogerpon_audit.py and the
    setup table in tools/gate_protagonist.py). The chain to a Phantom Dive is

        Dragapult ex in play  ->  a body can pay {R}{P}  ->  that body is ACTIVE  ->  dive

    and each arrow loses games. The measured drops name which arrow to fix:

      * in play -> can pay   : -7..-9pt on three opponents, but -20pt against ogerpon_mono,
        whose Crushing Hammers take 1.58 energies per game off the line. ENERGY.
      * can pay -> is ACTIVE : -20..-25pt on EVERY opponent, and two full turns of clock
        (payable at our turn ~4.5, active at ~6.5-7.5). PROMOTION. This is the big one.
      * is ACTIVE -> dive    : 125/125, 178/178, 199/207, 199/200. Already perfect; forcing
        it measured +0.33 +- 0.87. Nothing to win here, and the rule that tried was dropped.

    So the promotion arrow gets the work. The structural cause is visible in `main_ladder`:
    `step_attack` runs BEFORE `step_retreat`, and `FocusL2.rule_promote_focus` bails out at
    `if ctx.attacks` -- so an Active that can throw ANY chip attack (Drakloak's Jet Headbutt,
    Budew's Itchy Pollen; measured 184 and 153 uses against ogerpon vs 85 Phantom Dives)
    permanently outranks a 200-damage Dragapult ex sitting on the bench. Against ogerpon_mono
    a payable Dragapult ex sat benched for 118 turns and we took a promote/retreat option on
    72 of them.

    Every rule here is OFF by default and switched on by `line.dusk` in tuning.json, so this
    class with no config is byte-identical to ConfigL2 and the A/B is paired.
    """

    DREEPY, DRAKLOAK, PULT = 119, 120, 121
    DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
    BUDEW = 235
    PHANTOM_DIVE = 154
    LINE = (DREEPY, DRAKLOAK, PULT)
    # The ogerpon_mono matchup, as one meshed plan (`ogre` in DUSK_RULES / line.dusk).
    # Verified card facts the rules lean on:
    #   Teal Mask Ogerpon ex: 210 HP, Basic, retreat 1; Teal Dance attaches {G} from hand
    #   + draws; Myriad Leaf Shower = 30 + 30 x (energy on BOTH Actives).
    #   Their list: 4x Ogerpon and NOTHING else -- every KO we take is 2 prizes, three
    #   clean KOs is the game. ~23 Items (4 Hammer, 4 Tera Orb, 4 Bug Catching Set,
    #   3 Jumbo Ice Cream = heal 80 if 3+ energy, 1 Hero's Cape +100), 2 Lively Stadium
    #   (+30 HP to all BASICS -> 240), Boss x2, Judge/Lillie x8.
    # Ours: Phantom Dive 200 + 6 bench counters; Jet Headbutt 70 for ONE colorless (the
    # charging Dragapult still hits); Adrena-Brain moves up to 3 counters ACROSS (heal our
    # diver AND finish their 10); Watchtower bounces their stadium; Crushing Hammer x3 and
    # Handheld Fan keep Myriad Leaf Shower and Jumbo Ice Cream below their thresholds.
    OGERPON = 96
    MUNKIDORI, FEZ, MEOWTH = 112, 140, 1071
    HAMMER, WATCHTOWER, LIVELY, FAN, CRISPIN = 1120, 1256, 1251, 1161, 1198
    STRETCHER, RUINS, JAMMING = 1097, 1260, 1246
    OUR_STADIUMS = (WATCHTOWER, RUINS, JAMMING)
    FIRE_E, PSY_E, DARK_E = 2, 5, 7      # basic-energy card ids == their energy types
    # A line card is dead in hand without the body it evolves from. This is the engine-side
    # twin of plan_filter's `search_bottom`, which measured +6.55 +- 1.13 on the encoder and
    # +10.62 +- 1.89 on the bare 4B -- the largest single gain of the project.
    PREREQ = {DRAKLOAK: DREEPY, PULT: DRAKLOAK, DUSCLOPS: DUSKULL, DUSKNOIR: DUSCLOPS}
    LINE_TARGET = 2            # Dreepy+Drakloak bodies wanted before anything else is searched
    SETUP_TURNS = 6

    def __init__(self, deck, profile=None):
        super().__init__(deck, profile)
        d = dict(((profile or {}).get("line") or {}).get("dusk") or {})
        # DUSK_RULES=front,charge,search,bench[,cap=2] overrides the config for ABLATION.
        # The A/B runs two processes with the same --seed rather than two arms in one, so the
        # switch has to be reachable from the environment; tuning.json stays the shipped value.
        env = os.environ.get("DUSK_RULES")
        if env is not None:
            d = {}
            for tok in env.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                k, _, v = tok.partition("=")
                d[k] = int(v) if v else True
        self._d_front = bool(d.get("front"))        # promote the body closest to a dive
        self._d_charge = bool(d.get("charge"))      # attach toward {R}{P}, and stop at 2
        self._d_search = bool(d.get("search"))      # prohibition + setup priority on searches
        self._d_bench = bool(d.get("bench"))        # develop Dreepy first
        self._d_cap = int(d.get("energy_cap") or 0)  # 0 = off; vs energy-scaling attackers
        self._d_spread = bool(d.get("spread"))      # counters that make bodies dive-lethal
        self._d_boss = bool(d.get("boss"))          # drag the wounded one into the dive
        self._d_ogre = bool(d.get("ogre"))          # the ogerpon_mono meshed plan
        # The GAMBIT (user 2026-08-16): against ogerpon only the fast-assembly branch of
        # the draw distribution wins, so stop hedging for the slow branch -- dig through
        # brick hands with shuffle-draw regardless of hand size, and focus every energy
        # on one body (the anti-hammer split is a hedge that only pays in long games the
        # gambit has already written off).
        self._d_rush = bool(d.get("rush"))
        if self._d_boss:
            self.ladder = ("rule_boss_the_wounded",) + self.ladder
        if self._d_front:
            self.ladder = ("rule_front_the_dive",) + self.ladder
        if self._d_ogre:
            # Order: take the free KOs first (close), then remove their +30 before any
            # damage math is settled (stadium), then the supporter that fuels us
            # (crispin), then the items that slow their ramp (hammer, fan). front/boss
            # keep their existing slots below these.
            self.ladder = ("rule_ogre_close", "rule_ogre_attach", "rule_ogre_stadium",
                           "rule_ogre_crispin", "rule_rush_dig", "rule_ogre_stretcher",
                           "rule_ogre_stamp", "rule_ogre_hammer", "rule_ogre_trap",
                           "rule_ogre_fan") + self.ladder
            # Route the TO_BENCH sub-select (Poffin's direct-to-bench path) through
            # _is_spare_ex_sub, where the ogre diet below can DECLINE -- score ranking
            # alone cannot: round 1 measured Duskull@bench deaths going UP (209 -> 351)
            # because the diet reordered the menu but the engine still filled maxCount.
            self._bench_sub_guard = True
            # A wide bench is TIME. Their six prizes come one chaff kill per turn, so
            # every extra 1-prize body on our bench is a full extra turn of fuel and
            # draws; the diet above keeps the bench pure, this makes it deep.
            self.bench_target = 5

    # ----- shared perception ---------------------------------------------- #
    def _pd_cost(self):
        a = _ATTACKS.get(self.PHANTOM_DIVE)
        return list(a.energies or []) if a is not None else []

    def _can_dive(self, v):
        """Does THIS body have Phantom Dive available right now? Asked through the
        engine's own `ready` list so it is the same payment test the engine resolves
        with -- a hand-rolled {R}{P} check would drift from _can_pay."""
        return v is not None and any(aid == self.PHANTOM_DIVE for aid, _d, _c in v.ready)

    def _dive_progress(self, v, extra=None):
        """How close is this body to a Phantom Dive: 2 = can dive, 1 = one energy short,
        0 = not on the line at all. `extra` asks the question about a hypothetical attach."""
        if v is None or v.id not in self.LINE:
            return 0
        have = list(v.energy) + ([extra] if extra is not None else [])
        cost = self._pd_cost()
        if not cost:
            return 0
        if _can_pay(cost, have):
            return 2 if v.id == self.PULT else 1     # only Dragapult ex actually HAS the attack
        for e in cost:                               # one short, and typed-correct so far
            if _can_pay([x for x in cost if x != e][:len(have)], have):
                return 1
        return 1 if have else 0

    def _line_bodies(self, ctx):
        return sum(1 for v in ctx.me.inplay() if v.id in (self.DREEPY, self.DRAKLOAK, self.PULT))

    def _pd_damage(self):
        a = _ATTACKS.get(self.PHANTOM_DIVE)
        return (a.damage or 0) if a is not None else 0

    # ----- 5. the six counters are a KO SETUP, not chip damage -------------- #
    def decide_target(self, ctx, kind):
        """Phantom Dive's six bench counters, spent so that each one CREATES a target.

        The base scorer biases spread onto the LOWEST-hp body (`base += 10000 - pk.hp`),
        so every counter of a dive lands on the same Pokemon -- measured against
        ogerpon_mono, 485 of 613 counters went to the bench and 0 games were won by it.
        The arithmetic says why that is the wrong pile: Phantom Dive hits the Active for
        200 and Teal Mask Ogerpon ex has 210 HP, so a benched body needs exactly ONE
        counter to become a Boss's Orders away from a prize. Six counters on one body is
        one 60-damage Pokemon; six counters on six bodies is six lethal ones.
        """
        # trap-gust target (#4): rule_ogre_trap played the Boss for TEMPO -- the base
        # gust scorer would drag the biggest threat in; the trap wants the EMPTY one.
        if kind == "gust" and getattr(self, "_trap_gust", False) and self._ogre(ctx):
            self._trap_gust = False
            opt2 = ctx.sel.option

            def tk(i):
                pk = ctx.opp_pokemon_at(opt2[i])
                return len(pk.energies or []) if pk is not None else 99
            return sorted(range(len(opt2)), key=tk)
        if not self._d_spread or kind != "spread":
            return super().decide_target(ctx, kind)
        opt = ctx.sel.option
        dive = self._pd_damage()
        remain = (getattr(ctx.sel, "remainDamageCounter", None) or 0) * 10
        # Is the dive still available THIS turn? These menus come from Cursed Blast /
        # Adrena-Brain, which never end the turn -- so an Active that these counters
        # bring within 200 dies to the dive that follows. Phantom Dive's own placement
        # menu is bench-only, so this term never double-counts its own damage.
        a = ctx.me.active
        atk = dive if (a is not None and self._can_dive(a)) else 0

        def sc(i):
            o = opt[i]
            pk = ctx.opp_pokemon_at(o)
            if pk is None:
                return (-1, 0)
            hp = pk.hp or 0
            # A body THESE counters finish outranks every bank: this is the Adrena-Brain /
            # Cursed Blast close (the dive's own menu is bench-only, where the same test
            # catches a 60-or-less body). Without this tier the ACTIVE ranked last and an
            # Adrena fired to kill a 10 HP survivor sent its counters to the bench.
            if 0 < hp <= remain:
                return (3, -hp)
            if o.area == AreaType.ACTIVE:
                # Blast + Dive combo: counters that bring the Active into the 200's
                # range ARE a kill, same tier -- without this the 13 counters of a
                # combo-fired Dusknoir drained to the bench and the 330 never landed.
                if atk and 0 < hp <= remain + atk:
                    return (3, -hp)
                return (0, 0)                     # the Active is already taking the 200
            over = hp - dive
            if over > 0:
                # #3: among bank targets, the most-CHARGED body first -- energy is the
                # best available predictor of which one they promote next.
                return (2, -over, -len(pk.energies or []))
            return (1, -hp, 0)                    # already lethal: pile on only as a last resort
        return sorted(range(len(opt)), key=lambda i: sc(i) + (0,) * (3 - len(sc(i))),
                      reverse=True)

    # ----- 6. cash the setup ------------------------------------------------ #
    def rule_boss_the_wounded(self, ctx):
        """When the Active cannot be knocked out by a dive but a benched body can, the
        gust IS the attack. Boss's Orders was offered on 629 turns and played on 208."""
        a = ctx.me.active
        if not self._can_dive(a):
            return None
        dive = self._pd_damage()
        oa = ctx.opp.active
        if oa is not None and oa.hp <= dive:
            return None                            # what is in front already dies
        if not any((v.hp or 0) <= dive for v in ctx.opp.bench):
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and self._in_bucket(c, _GUST):
                return [i]
        return None

    # ----- 7. the ogerpon_mono meshed plan (`ogre`) -------------------------- #
    def _ogre(self, ctx):
        return self._d_ogre and any(v.id == self.OGERPON for v in ctx.opp.inplay())

    def _ogre_intel(self, ctx):
        """Counting, made a data structure. The opponent's DISCARD is public and their
        LIST is known (the user's premise): the difference is what they still hold.
        Every human counter-play below keys off one of these numbers -- most importantly
        boss_left, because with both Boss's Orders spent NOTHING they run touches our
        bench, and every bench-diet tax stops being worth paying."""
        disc = [getattr(c, "id", None) for c in (ctx.opp_ps.discard or [])]
        inplay = [v.id for v in ctx.opp.inplay()]
        st = [getattr(c, "id", None) for c in (ctx.state.stadium or [])]
        return {
            "boss_left": max(0, 2 - disc.count(1182)),
            "lively_left": max(0, 2 - disc.count(self.LIVELY) - st.count(self.LIVELY)),
            "cape_left": max(0, 1 - disc.count(1159)),
            "hammer_left": max(0, 4 - disc.count(self.HAMMER)),
            "ice_left": max(0, 3 - disc.count(1147)),
            "briar_left": max(0, 1 - disc.count(1201)),
            "bodies_left": max(0, 4 - disc.count(self.OGERPON)) ,
            "bodies_off_board": max(0, 4 - disc.count(self.OGERPON) - inplay.count(self.OGERPON)),
        }

    def _ogre_movable(self, ctx):
        """Counters Adrena-Brain can send: up to 3, from ONE living body of ours."""
        best = 0
        for v in ctx.me.inplay():
            if (v.hp or 0) <= 0:
                continue
            best = max(best, min(30, (v.max_hp or 0) - (v.hp or 0)))
        return (best // 10) * 10

    def rule_ogre_close(self, ctx):
        """Fire the abilities that TAKE a knock-out, or keep the diver alive.

        Adrena-Brain is both halves of the plan at once: their 210s survive a dive at
        exactly 10, and the damage they land on our 320 is the counter supply -- moving
        it heals the diver AND finishes their body, without ending the turn. Fired when
        it kills, or as pure sustain once the diver carries 30+.

        The Cursed Blasts are AMMUNITION, not a last resort. The trade is 1 prize for 2
        every time, and they field exactly FOUR attackers -- each blast KO thins the
        only win condition they have. So: Dusknoir fires on any body in its 130, and
        Dusclops on any body in its 50, with two economies kept: a kill the dive takes
        for free this turn is not paid for with a prize, and a BENCH target (which
        cannot heal) waits for the Dusknoir in hand that does 2.6x at the same cost.
        One hard stop: never hand them their LAST prize unless the blast wins first."""
        if not self._ogre(ctx) or not ctx.abilities:
            return None
        opt = ctx.sel.option
        mv = self._ogre_movable(ctx)
        opp_hps = [(v.hp or 0) for v in ctx.opp.inplay()]
        kill_mv = mv > 0 and any(0 < h <= mv for h in opp_hps)
        diver = None
        for v in ctx.me.inplay():
            if v.id in self.LINE and (diver is None or v.energy_count > diver.energy_count):
                diver = v
        heal = (diver is not None and diver.energy_count > 0
                and ((diver.max_hp or 0) - (diver.hp or 0)) >= 30)
        my_left = ctx.prize.mine if ctx.prize else 99
        opp_left = ctx.prize.opp if ctx.prize else 99
        oa = ctx.opp.active
        act_hp = (oa.hp or 0) if oa is not None else 0
        bench_hps = [(v.hp or 0) for v in ctx.opp.bench]
        a = ctx.me.active
        dive_now = a is not None and self._can_dive(a)
        hand_ids = [getattr(h, "id", None) for h in (ctx.me_ps.hand or [])]

        dive = self._pd_damage()
        # What the ACTIVE needs on top of this turn's dive: the COMBO case. With the
        # dive available, a blast is worth firing exactly when act_hp sits in
        # (dive, dive + reach] -- Blast first (does not end the turn), then the 200
        # finishes. 330 total is the Hero's-Cape answer; 250 covers a fresh Lively 240.
        needed = (act_hp - dive) if (dive_now and act_hp > dive) else None
        clops_here = any(ctx.field_pk(opt[j]) is not None
                         and ctx.field_pk(opt[j]).id == self.DUSCLOPS
                         for j in ctx.abilities)

        def blast_ok():
            """A blast that takes 2 wins us the game at my_left<=2 BEFORE their prize
            from our self-KO matters; otherwise never blast into their last prize."""
            return my_left <= 2 or opp_left > 1
        for i in ctx.abilities:
            pk = ctx.field_pk(opt[i])
            if pk is None:
                continue
            if pk.id == self.MUNKIDORI:
                combo_mv = needed is not None and mv > 0 and needed <= mv
                # bank: with Risky Ruins seeding counters on our own Basics, the
                # once-per-turn move is free value even without a kill -- 30 a turn
                # toward a body still above the dive's 200.
                bank = mv >= 10 and any((v.hp or 0) > dive for v in ctx.opp.inplay())
                if kill_mv or combo_mv or heal or bank:
                    return [i]
            # DOOMED-BLAST (Duskull-line audit, 2026-08-16): a Dusclops/Dusknoir that is
            # our ACTIVE and inside their ready damage dies THIS rotation whatever we
            # do -- the only choice is whether its counters die with it. Firing into
            # the bank (the spread scorer routes them at the most-charged out-of-range
            # body) converts a mute death into 50/130 banked.
            _mine_v = pk and PokemonView(pk, self.roles.get(pk.id))
            _doomed = (ctx.me.active is not None and pk is ctx.me.active.pk
                       and ctx.opp.active is not None
                       and ctx.opp.active.best_ready_dmg >= (pk.hp or 0) > 0)
            if pk.id == self.DUSKNOIR and blast_ok():
                bench_kill = any(0 < h <= 130 for h in bench_hps)
                act_kill = 0 < act_hp <= 130 and not dive_now
                # combo: only when the cheaper finishers cannot cover it -- Adrena is
                # free and Dusclops costs the same prize for the sub-50 gap.
                act_combo = needed is not None and needed <= 130 \
                    and needed > mv and not (clops_here and needed <= 50)
                if bench_kill or act_kill or act_combo or my_left <= 2 or _doomed:
                    return [i]
            if pk.id == self.DUSCLOPS and blast_ok():
                noir_waiting = self.DUSKNOIR in hand_ids
                bench_kill = any(0 < h <= 50 for h in bench_hps)
                act_kill = 0 < act_hp <= 50 and not dive_now
                act_combo = needed is not None and needed <= 50 and needed > mv
                # the ACTIVE can heal (Jumbo Ice Cream) or grow a Cape -- take it now
                # even with Dusknoir waiting; a bench target keeps for the upgrade.
                if act_kill or act_combo or (bench_kill and not noir_waiting) \
                        or my_left <= 2 or _doomed:
                    return [i]
        return None

    def rule_ogre_attach(self, ctx):
        """Attach BEFORE the supporter step. main_ladder runs trainer -> attach, so our
        own Lillie's Determination (x4, shuffle-draw) was flushing the held {R}/{P} back
        into the deck one step before the attach could bank it. The manual attach is
        free and once per turn: the moment a positive line target exists, take it."""
        if not self._ogre(ctx) or ctx.state.energyAttached or not ctx.attaches:
            return None
        return self.decide_energy_target(ctx)

    def rule_rush_dig(self, ctx):
        """GAMBIT: a hand with no {R}/{P}, no Dragapult ex and no Rare Candy cannot
        advance the only plan that wins -- shuffle it away NOW, at any hand size. The
        default draw threshold (hand <= 5) is calibrated for hedged play; the gambit
        pays cards for tempo because the slow branch is already conceded."""
        if not (self._d_rush and self._ogre(ctx)) or ctx.state.supporterPlayed:
            return None
        cost = self._pd_cost()
        if any(v.id in self.LINE and _can_pay(cost, v.energy) for v in ctx.me.inplay()):
            return None
        hand = [getattr(h, "id", None) for h in (ctx.me_ps.hand or [])]
        if any(x in hand for x in (self.FIRE_E, self.PSY_E, self.PULT, 1079,
                                   self.DRAKLOAK, self.CRISPIN)):
            return None                          # the hand still advances the plan
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is not None and c.cardId in (1227, 1213):   # Lillie / Judge
                return [i]
        return None

    def rule_ogre_stretcher(self, ctx):
        """Night Stretcher as the third fuel source. 6 basic {R}/{P} in 60 cards is the
        whole budget, and every one that hits the discard (Ultra Ball costs, retreat,
        hammered) is 17% of it -- the recover item puts one back in hand for free."""
        if not self._ogre(ctx):
            return None
        cost = self._pd_cost()
        if any(v.id in self.LINE and _can_pay(cost, v.energy) for v in ctx.me.inplay()):
            return None
        hand_e = sum(1 for h in (ctx.me_ps.hand or [])
                     if getattr(h, "id", None) in (self.FIRE_E, self.PSY_E))
        if hand_e >= 2:
            return None
        disc = [getattr(x, "id", None) for x in (ctx.me_ps.discard or [])]
        if self.FIRE_E not in disc and self.PSY_E not in disc:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is not None and c.cardId == self.STRETCHER:
                return [i]
        return None

    def rule_ogre_stadium(self, ctx):
        """The stadium war, in priority order.

        Risky Ruins is SLAMMED the moment the table is empty: every Basic non-{D} they
        bench after it lands at 190 -- inside Phantom Dive's 200 forever. The self-chip
        on our own Basics costs nothing here (their 70+ attacks one-shot the chaff with
        or without it, and Adrena-Brain turns the counters into ammunition).
        Their Lively Stadium makes every Ogerpon 240 (dive arithmetic 200+40), so it is
        bumped on sight -- with Ruins first (bump AND mark) and Watchtower as the
        second bullet ({C}-only text touches neither side here; it exists to bounce)."""
        if not self._ogre(ctx) or ctx.state.stadiumPlayed:
            return None
        st = ctx.state.stadium
        empty = not st
        lively = bool(st) and any(getattr(c, "id", None) == self.LIVELY for c in st)
        if not (empty or lively):
            return None
        # Priority among OUR stadiums vs ogerpon: Jamming Tower first (it kills Hero's
        # Cape and Jumbo Ice Cream is unaffected but the Cape is the 310-HP problem),
        # then Ruins (bump + mark), Watchtower last (bump only).
        best = None
        order = {self.JAMMING: 0, self.RUINS: 1, self.WATCHTOWER: 2}
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is None or c.cardId not in order:
                continue
            if best is None or order[c.cardId] < order[best[1]]:
                best = (i, c.cardId)
        if best is None:
            return None
        if empty or lively:
            # Watchtower on an EMPTY table is a spent bump with nothing to bump -- hold
            # it for their Lively unless it is all we will ever draw.
            if best[1] == self.WATCHTOWER and empty:
                return None
            return [best[0]]
        return None

    def rule_ogre_crispin(self, ctx):
        """While no line body can pay {R}{P}, the supporter IS Crispin -- the deck's only
        acceleration in a matchup where hammers strip 1.58 energies a game and the fuel
        is the whole game ({R}{P} banked in 24% of games without it)."""
        if not self._ogre(ctx) or ctx.state.supporterPlayed:
            return None
        cost = self._pd_cost()
        if any(v.id in self.LINE and _can_pay(cost, v.energy) for v in ctx.me.inplay()):
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is not None and c.cardId == self.CRISPIN:
                return [i]
        return None

    def rule_ogre_stamp(self, ctx):
        """HUMAN COUNTER-PLAY #5: Unfair Stamp lands hardest when their HAND is rich --
        the option's very presence proves the legality window (one of ours was Knocked
        Out last turn), so the only judgement left is theirs to lose: 4+ cards shuffled
        away for a 2-card hand while their board still wants Tera Orbs and energy."""
        if not self._ogre(ctx):
            return None
        if (ctx.opp.hand_count or 0) < 4:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is not None and c.cardId == 1080:
                return [i]
        return None

    def rule_ogre_trap(self, ctx):
        """HUMAN COUNTER-PLAY #4: the Boss that KILLS nothing but costs them a turn.
        When no knockout exists for us this turn, their Active is charged, and a fresh
        0-energy Ogerpon sits on their bench, gusting the empty one forward makes them
        choose between a wasted attack turn (Teal Dance one energy at a time) and
        burning N's Plan. Budgeted: only while a second Boss remains in OUR deck-or-hand
        economy is not tracked, so the rule simply never fires once lethal_boss or the
        gust-for-kill paths could use the copy -- they run earlier in the ladder."""
        if not self._ogre(ctx) or ctx.state.supporterPlayed:
            return None
        oa = ctx.opp.active
        if oa is None or oa.energy_count < 2:
            return None
        # no kill for us this turn (dive nor blast reaches anything)
        dive = self._pd_damage()
        a = ctx.me.active
        can_kill = a is not None and self._can_dive(a) \
            and any(0 < (v.hp or 0) <= dive for v in ctx.opp.inplay())
        if can_kill:
            return None
        empty_bench = [v for v in ctx.opp.bench if v.energy_count == 0]
        if not empty_bench:
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is not None and self._in_bucket(c, _GUST):
                self._trap_gust = True     # the gust SELECT must take the EMPTY body
                return [i]
        return None

    def rule_ogre_hammer(self, ctx):
        """Crushing Hammer at every energy the coin allows: each {G} on their Active is
        30 more Myriad Leaf Shower, the third enables Jumbo Ice Cream's 80 heal, and
        N's Plan pulls bench energy forward -- bench energy is not safe either."""
        if not self._ogre(ctx):
            return None
        if not any(v.energy_count >= 1 for v in ctx.opp.inplay()):
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is not None and c.cardId == self.HAMMER:
                return [i]
        return None

    def rule_ogre_fan(self, ctx):
        """Handheld Fan on the line: every attack they land on the wearer moves one of
        THEIR energies back to their bench -- -30 off the next Leaf Shower and pressure
        off the Ice Cream threshold. The target select routes through decide_target's
        my-side scoring, which already prefers the best attacker."""
        if not self._ogre(ctx):
            return None
        if not any(v.id in self.LINE for v in ctx.me.inplay()):
            return None
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c is not None and c.cardId == self.FAN:
                return [i]
        return None

    def _ogre_bench_ban(self, ctx, cid, taken=()):
        """The bench diet's HARD half: bodies that must not reach the board at all.

        Their farm needs six 1-prize kills or three 2-prize ones, and round 1 measured
        the soft (score-only) diet failing: Duskull@bench deaths ROSE because ranking
        cannot decline. Fez/Meowth are 2 prizes parked for nothing; Duskull without its
        evolution in hand is a multi-turn gust target. A genuinely thin board overrides
        everything -- a lost promotion is the only thing worse than a fed prize."""
        board = len(ctx.me.inplay()) + len(taken)
        if ctx.sel.context == SelectContext.SETUP_BENCH_POKEMON:
            board += 1                       # the committed Active is not in obs yet
        # HUMAN COUNTER-PLAY #2/#8: once both Boss's Orders are in their discard, their
        # deck has NO way to touch our bench -- every diet rule below is a tax with no
        # collector. Fez comes down for its draw, Duskull comes down freely.
        try:
            if self._ogre_intel(ctx)["boss_left"] == 0:
                return False
        except Exception:                          # noqa: BLE001
            pass
        # A 2-prize support ex is NEVER worth benching into the farm -- the pokehubguide
        # list carries only six 1-prize bodies, so their six prizes are 4 kills once a
        # Fez joins the buffet (measured: game over on our turn 7.0). Desperation floor
        # applies only to 1-prize bodies.
        if cid in (self.FEZ, self.MEOWTH):
            return True
        if board <= 1:
            return False
        if cid == self.DUSKULL:
            # Duskull deaths ran 0.9/game: benched early it is farmed for turns before
            # any spike exists. It comes down only when the spike can DEPLOY at once --
            # Dusclops in hand, or Rare Candy + Dusknoir in hand (skip the Stage 1) --
            # AND in a Boss-safe window: both their Boss's Orders counted out, or their
            # hand freshly reset to 3 or fewer (a held Boss is then unlikely). The
            # mandatory one-turn pre-evolution exposure is exactly what their two
            # gusts farm; choosing WHEN to expose is the whole counter-play.
            hand_ids = [getattr(c, "id", None) for c in (ctx.me_ps.hand or [])]
            ready = self.DUSCLOPS in hand_ids \
                or (1079 in hand_ids and self.DUSKNOIR in hand_ids)
            try:
                _it2 = self._ogre_intel(ctx)
                safe = _it2["boss_left"] == 0 or (ctx.opp.hand_count or 9) <= 3
            except Exception:                      # noqa: BLE001
                safe = True
            second = self.DUSKULL in taken \
                or any(v.id in (self.DUSKULL, self.DUSCLOPS, self.DUSKNOIR)
                       for v in ctx.me.inplay())
            return not (ready and safe) or second
        return False

    def _is_spare_ex_sub(self, ctx, cid, taken):
        # Flag-only fallback when the opponent is not visible yet (setup): a DEDICATED
        # ogerpon engine diets from the first bench too -- round 2 measured Duskull@bench
        # deaths staying at 318/400 because the setup bench was exempt.
        live = self._ogre(ctx) or not any(True for _ in ctx.opp.inplay())
        if self._d_ogre and live and self._ogre_bench_ban(ctx, cid, taken):
            return True
        return super()._is_spare_ex_sub(ctx, cid, taken)

    def decide_bench(self, ctx):
        """MAIN-menu benching from hand, with the same hard diet as the sub-select."""
        if not (self._d_ogre and self._ogre(ctx)):
            return super().decide_bench(ctx)
        if len(ctx.me.bench) >= self.bench_target:
            return None
        cands = []
        for i in ctx.plays:
            card = ctx.hand_card(ctx.sel.option[i])
            if card and card.cardType == CardType.POKEMON and card.basic:
                if self._ogre_bench_ban(ctx, card.cardId):
                    continue
                if _SPARE_EX_GUARD and self._is_spare_ex(ctx, card):
                    continue
                cands.append((self._bench_score(card.cardId), i))
        if not cands:
            return None
        return [max(cands)[1]]

    def choose_sub(self, ctx):
        """Two sub-selects the base routes to decide_acquire's card-need ranking, which
        is the wrong question for both (round-4 trace: Crispin's energy landed on a
        benched Meowth ex, because a big attacker out-needs a Dreepy):

        ATTACH_TO   -- WHICH basic energies to take (Crispin's deck pick). Take the two
                       dive types; {R} and {P} are 'different types' so both are legal.
        ATTACH_FROM -- WHICH of our bodies receives. The pair math, not the need score:
                       a line body one type short of {R}{P} first, Dragapult ex above
                       Drakloak above Dreepy, non-line bodies never."""
        if self._d_ogre and self._ogre(ctx):
            c = ctx.sel.context
            opt = ctx.sel.option
            if c == SelectContext.ATTACH_TO:
                fire, psy, rest = [], [], []
                for i in range(len(opt)):
                    cid = self._opt_card_id(ctx, opt[i])
                    card = _CARDS.get(cid)
                    et = card.energyType if card is not None else None
                    (fire if et == 2 else psy if et == 5 else rest).append(i)
                if fire or psy:
                    return (fire[:1] + psy[:1]) or fire[:2] or psy[:2]
            if c == SelectContext.ATTACH_FROM:
                cost = self._pd_cost()

                def ak(i):
                    pk = ctx.field_pk(opt[i])
                    if pk is None:
                        return -1
                    v = PokemonView(pk, self.roles.get(pk.id))
                    if v.id not in self.LINE:
                        return 0
                    have_r = 2 in v.energy
                    have_p = 5 in v.energy
                    one_short = have_r != have_p and not _can_pay(cost, v.energy)
                    stage = {self.PULT: 3, self.DRAKLOAK: 2, self.DREEPY: 1}[v.id]
                    return (100 if one_short else 10) + stage
                return sorted(range(len(opt)), key=ak, reverse=True)
        return super().choose_sub(ctx)

    def decide_discard(self, ctx):
        """Discard costs (Ultra Ball's 2) keep their base ranking except for the cards
        this matchup cannot replace: the round-4 trace paid an Ultra Ball with the
        {R}{P} PAIR -- the exact two cards the game is about -- to fetch a Dreepy.
        Basic {R}/{P} (6 in 60), Crispin, and the Dragapult line go to the back."""
        base = super().decide_discard(ctx)
        if not (self._d_ogre and self._ogre(ctx)) or not base:
            return base
        precious = {2, 5, self.CRISPIN, self.PULT, self.DRAKLOAK}

        def is_precious(i):
            return self._opt_card_id(ctx, ctx.sel.option[i]) in precious
        return [i for i in base if not is_precious(i)] + [i for i in base if is_precious(i)]

    def decide_ability(self, ctx):
        """In ogre mode the prize-conceding abilities belong to rule_ogre_close alone:
        the base 'clearly beneficial' test fires Cursed Blast eagerly, which is how the
        engine feeds cheap prizes. Recon Directive still flows through the base."""
        r = super().decide_ability(ctx)
        if r and self._d_ogre and self._ogre(ctx):
            pk = ctx.field_pk(ctx.sel.option[r[0]])
            if pk is not None and pk.id in (self.DUSCLOPS, self.DUSKNOIR, self.MUNKIDORI):
                return None
        return r

    # ----- 1. promotion: the -20..-25pt arrow ------------------------------ #
    def rule_front_the_dive(self, ctx):
        """Put the Dragapult ex that can already pay {R}{P} in FRONT, even though the
        current Active could throw a chip attack instead.

        `FocusL2.rule_promote_focus` is the same idea gated on `not ctx.attacks`, which is
        exactly the case the forensics say never happens: the Active can nearly always do
        SOMETHING, so the armed Dragapult never comes up. The trade this rule makes is one
        chip attack (Jet Headbutt 70 / Itchy Pollen 20) for a 200 + six-counter turn, so it
        only fires when the chip attack is not itself decisive.
        """
        if ctx.state.retreated:
            return None
        armed = next((v for v in ctx.me.bench if v.id == self.PULT and self._can_dive(v)), None)
        if armed is None:
            return None
        a = ctx.me.active
        if a is not None and self._can_dive(a):
            return None                              # already the right body in front
        # never trade away a KO that ends the game, nor a KO the Active can take right now
        if ctx.attacks:
            prize = getattr(ctx, "prize", None)
            if prize is not None and prize.can_close and ctx.ko_targets:
                return None
            oa = ctx.opp.active
            if oa is not None and a is not None and a.best_ready_dmg >= oa.hp:
                return None
        # a free switch first: retreating pays its cost by DISCARDING attached energy, and
        # this deck's energy is the scarce resource (ogerpon strips 1.58 per game)
        for i in ctx.plays:
            c = ctx.hand_card(ctx.sel.option[i])
            if c and self._in_bucket(c, _SWITCH_CARDS):
                return [i]
        if ctx.retreat_idx is None:
            return None
        # do not strand the line's own energy: an Active that is itself a charging line body
        # is one evolve away from being the attacker, so it is worth more where it stands
        if a is not None and a.id in self.LINE and a.energy_count:
            return None
        return [ctx.retreat_idx]

    def decide_active(self, ctx, mode="setup"):
        """KO-replacement and setup promotion: a body that can dive outranks everything.

        In ogre mode the promotion IS the prize race. Their six prizes are farmed off
        whatever we put in front, so the order is: armed Dragapult (dive now), then the
        cheapest EMPTY body (Budew first: its free Itchy Pollen chips 10 and locks their
        ~23 Items for a turn while it dies), and the charging bodies / the Drakloak draw
        engine / the unarmed 2-prize ex stay OFF the front. Promoting a body that
        carries {R}/{P} buries the fuel with it -- empty sacrifices only."""
        # No mode gate: choose_sub calls this with mode="setup" for EVERY TO_ACTIVE
        # (round-4 trace: the sacrifice ordering never ran and an unarmed Dragapult ex
        # was promoted into Myriad Leaf Shower). At true setup the opponent's board is
        # empty, so _ogre() itself is the setup gate.
        if self._d_ogre and self._ogre(ctx):
            opt = ctx.sel.option
            intel = self._ogre_intel(ctx)
            oa = ctx.opp.active
            opp_dmg = oa.best_ready_dmg if oa is not None else 0
            my_left = len(ctx.me_ps.prize or [])

            def okey(i):
                pk = ctx.field_pk(opt[i])
                if pk is None:
                    return (99, 0)
                v = PokemonView(pk, self.roles.get(pk.id))
                if self._can_dive(v):
                    return (0, -v.energy_count)      # armed Dragapult: dive next turn
                # The Jet Headbutt bridge: one colorless pays for 70, so a HALF-charged
                # Dragapult in front chips a fresh 210 into blast/dive range while the
                # second energy assembles -- but only while their ramp cannot answer:
                # Myriad Leaf Shower needs ~9 energies across both Actives to one-shot
                # 320, so the bridge closes once their board carries 5+.
                # Gate: a one-shot on 320 needs 30+30x(their E + ours) >= 320, i.e.
                # ~8 of theirs against a 2-energy bridge -- at <=6 in play the worst
                # hit is 270 and the bridge survives to chip at least twice.
                # #7 (Briar): at exactly 2 prizes remaining their Tera KO takes THREE --
                # while their Briar is unspent, a 2-prize body does not volunteer.
                if v.id == self.PULT and v.energy_count >= 1 \
                        and ctx.opp.energy_in_play <= 6 \
                        and not (my_left == 2 and intel["briar_left"] > 0):
                    return (1, -v.energy_count)
                # #9: among equal-prize sacrifices, prefer the one their CURRENT ready
                # damage cannot kill -- a surviving sacrifice is a whole free turn, or
                # forces them to spend an attach/N's Plan they wanted elsewhere.
                _survives = 0 if (v.hp or 0) > opp_dmg else 1
                if v.id == self.BUDEW:
                    return (2, _survives)            # lock + 10 chip, dies for 1 prize
                if v.id == self.DREEPY and not v.energy_count:
                    return (3, _survives)
                if v.id == self.MUNKIDORI and self.DARK_E not in v.energy:
                    return (4, _survives)
                # Dusclops BEFORE Duskull: 90 HP survives their early 60-70s, and if it
                # is doomed anyway the doomed-blast fires its 50 on the way out. A raw
                # Duskull in front is unfired ammunition handed over for nothing.
                if v.id == self.DUSCLOPS:
                    return (5, _survives)
                if v.id == self.DUSKULL:
                    return (6, _survives)
                if v.id == self.MUNKIDORI:
                    return (7, 0)                    # {D} attached: the closer, spend late
                if v.id == self.DREEPY:
                    return (8, -v.energy_count)      # charging: its energy dies with it
                if v.id == self.DRAKLOAK:
                    return (9, -v.energy_count)      # the draw engine
                if v.id == self.PULT:
                    return (10, -v.energy_count)     # unarmed ex: 2 free prizes
                return (11, 0)
            return sorted(range(len(opt)), key=okey)
        if not self._d_front:
            return super().decide_active(ctx, mode)
        opt = ctx.sel.option
        base = super().decide_active(ctx, mode)
        rank = {i: n for n, i in enumerate(base)}

        def key(i):
            pk = ctx.field_pk(opt[i])
            v = PokemonView(pk, self.roles.get(pk.id)) if pk is not None else None
            return (-self._dive_progress(v), rank.get(i, len(opt)))
        return sorted(range(len(opt)), key=key)

    # ----- 2. energy: the ogerpon-specific arrow --------------------------- #
    def decide_energy_target(self, ctx):
        """Attach toward a Phantom Dive, then STOP.

        Two measured facts drive this. (a) The line is the only thing in the deck that
        converts energy into prizes, and against ogerpon our Active carried a mean of 0.83
        energies -- we were not paying for our own attack. (b) Myriad Leaf Shower does
        30 more for each Energy on BOTH Actives, so the third energy on our Active is a
        gift of 30 damage to the opponent and buys us nothing: Phantom Dive costs two.
        """
        if not self._d_charge or ctx.state.energyAttached:
            return super().decide_energy_target(ctx)
        atts = self._energy_attach_opts(ctx)
        if not atts:
            return super().decide_energy_target(ctx)
        opt = ctx.sel.option
        ogre = self._d_ogre and self._ogre(ctx)

        def sc(i):
            pk = ctx.field_pk(opt[i])
            if pk is None:
                return (-1, 0)
            v = PokemonView(pk, self.roles.get(pk.id))
            c = ctx.hand_card(opt[i])
            et = c.energyType if c is not None else None
            # {D} pays nothing toward Phantom Dive; its whole job is switching on
            # Adrena-Brain. On a line body it is a wasted slot the charge scorer
            # below would otherwise credit.
            if ogre and et == self.DARK_E:
                # ... and only while Munkidori sits on the BENCH: the round-4 trace
                # attached {D} to an ACTIVE Munkidori and paid it straight back out as
                # the retreat cost two decisions later. Adrena-Brain works from the
                # bench, which Boss aside nothing of theirs can reach.
                o = opt[i]
                tgt_area = o.inPlayArea if o.inPlayArea is not None else o.area
                if v.id == self.MUNKIDORI and self.DARK_E not in v.energy \
                        and tgt_area != AreaType.ACTIVE:
                    return (50, 0)
                return (0, -v.energy_count)
            if v.id not in self.LINE:
                return (0, -v.energy_count)
            # Anti-Boss split: a 70 HP Dreepy carrying BOTH dive energies is their best
            # play in the game -- Boss's Orders erases body and fuel together. #10 makes
            # the split CONDITIONAL, the way a human plays it: it only buys anything
            # while they still hold the removal (2+ hammers or a Boss); with the threat
            # counted out of their deck, focusing one body is strictly faster.
            if ogre and not self._d_rush and v.id == self.DREEPY and v.energy_count >= 1:
                _it = self._ogre_intel(ctx)
                if _it["hammer_left"] >= 2 and _it["boss_left"] >= 1:
                    return (0, -v.energy_count)
            if self._d_cap and v.energy_count >= self._d_cap:
                return (0, -v.energy_count)          # full: another one only feeds Myriad
            before = self._dive_progress(v)
            after = self._dive_progress(v, et)
            # a body that BECOMES payable is worth more than one that merely inches closer,
            # and Dragapult ex (which owns the attack) more than the Drakloak under it.
            # The flat bonus is CONDITIONAL on progress: unconditioned, it scored a second
            # {P} onto a {P}-holding Dragapult positive, and the round-4 trace died with
            # the diver stuck on {P}{P} while the {R} went to a Dreepy.
            d = after - before
            if d <= 0:
                return (0, -v.energy_count)
            return (10 * d + (3 if v.id == self.PULT else 1), -v.energy_count)
        best = max(atts, key=sc)
        if sc(best)[0] > 0:
            return [best]
        if ogre:
            # No PROGRESS target -- but with 8 attack energies and games this short
            # (their farm closes by our turn ~7), holding is worse than banking: put it
            # on any line body below the cap. Only when even that does not exist is the
            # attach declined (the old hold, kept for the Munkidori-class waste).
            for i in atts:
                pk = ctx.field_pk(ctx.sel.option[i])
                if pk is None:
                    continue
                v = PokemonView(pk, self.roles.get(pk.id))
                if v.id in self.LINE and (not self._d_cap
                                          or v.energy_count < self._d_cap):
                    return [i]
            return None
        return super().decide_energy_target(ctx)

    # ----- 3. search: the +6.55/+10.62 rule, engine-side -------------------- #
    def decide_acquire(self, ctx):
        if not self._d_search:
            return super().decide_acquire(ctx)
        order = super().decide_acquire(ctx)
        opt = ctx.sel.option
        my = [v.id for v in ctx.me.inplay()]
        hand = [c.id for c in (ctx.me_ps.hand or [])]
        turn = ctx.state.turn or 0
        want = None
        want_e = False
        if self._d_ogre and self._ogre(ctx):
            # No turn gate and no Duskull step: against ogerpon the game IS finishing the
            # line before their farm takes six, and a benched Duskull is the farm's
            # favourite meal (1.14 deaths/game). Dragapult ex itself joins the chain --
            # Ultra Ball can fetch it, and a line with no Pult in sight is the loss.
            if my.count(self.DREEPY) + my.count(self.DRAKLOAK) < self.LINE_TARGET:
                want = self.DREEPY
            elif self.DRAKLOAK not in my and self.PULT not in my \
                    and self.DRAKLOAK not in hand:
                want = self.DRAKLOAK
            elif self.PULT not in my and self.PULT not in hand:
                want = self.PULT
            elif not any(x in my for x in (self.DUSKULL, self.DUSCLOPS, self.DUSKNOIR)) \
                    and self.DUSKULL not in hand:
                want = self.DUSKULL         # the blast line: 2 prizes per 1, on demand
            elif my.count(self.DREEPY) < 3 and self.DREEPY not in hand:
                want = self.DREEPY          # sacrifice depth: each Dreepy is a turn
            # line assembled but unpaid: a search that can see basic {R}/{P} (Night
            # Stretcher's discard menu) should take the fuel over another body.
            if want is None and not any(
                    v.id in self.LINE and _can_pay(self._pd_cost(), v.energy)
                    for v in ctx.me.inplay()):
                want_e = True
        elif turn <= self.SETUP_TURNS:
            if my.count(self.DREEPY) + my.count(self.DRAKLOAK) < self.LINE_TARGET:
                want = self.DREEPY
            elif self.DRAKLOAK not in my and self.PULT not in my:
                want = self.DRAKLOAK
            elif self.DUSKULL not in my:
                want = self.DUSKULL
        rank = {i: n for n, i in enumerate(order)}

        _candy = 1079 in hand                    # Rare Candy: Basic -> Stage 2 directly

        def key(i):
            cid = self._opt_card_id(ctx, opt[i])
            pre = self.PREREQ.get(cid)
            # dead in hand: nothing on the board (or in hand) to evolve it from
            bad = 1 if (pre is not None and pre not in my and pre not in hand) else 0
            if bad and _candy:
                base = self.PREREQ.get(pre)
                if base is not None and (base in my or base in hand):
                    bad = 0                      # Candy skips the missing Stage 1
            good = -1 if ((want is not None and cid == want)
                          or (want_e and cid in (self.FIRE_E, self.PSY_E))) else 0
            return (bad, good, rank.get(i, len(opt)))
        return sorted(range(len(opt)), key=key)

    # ----- 4. bench: two line bodies before anything else ------------------ #
    def _bench_score(self, cid):
        if not self._d_bench:
            return super()._bench_score(cid)
        # `_bench_score` has no ctx, so the board count is read off the live perception the
        # policy keeps for the current decision (set in act()); absent it, fall through.
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            return super()._bench_score(cid)
        n = self._line_bodies(ctx)
        if self._d_ogre and self._ogre(ctx):
            # The bench diet: every non-line body we park is a 1-prize meal for their
            # Boss's Orders farm (Duskull 1.14 deaths/game, 0.74 off the bench). Dreepy
            # is the engine, Budew is a planned sacrifice that locks Items while it
            # waits, Munkidori is the closer. Duskull only with its evolution IN HAND
            # (one-turn exposure), and the spare 2-prize exes never.
            hand_ids = [c.id for c in (ctx.me_ps.hand or [])]
            if cid == self.DREEPY:
                return 10000 if n < self.LINE_TARGET else 500
            if cid == self.BUDEW:
                return 600
            if cid == self.DUSKULL:
                have = any(v.id in (self.DUSKULL, self.DUSCLOPS, self.DUSKNOIR)
                           for v in ctx.me.inplay())
                if not have:
                    return 450               # first copy = ammunition, bench it
                return 300 if (self.DUSCLOPS in hand_ids
                               or self.DUSKNOIR in hand_ids) else 20
            if cid == self.MUNKIDORI:
                return 350
            if cid in (self.MEOWTH, self.FEZ):
                return 10
        if cid == self.DREEPY:
            return 10000 if n < self.LINE_TARGET else 400
        if cid == self.DUSKULL:
            return 900 if n >= self.LINE_TARGET else 300
        return super()._bench_score(cid)


_PERDECK = {
    "dusknoir": DuskNoirL2,
    "mega_lucario_tr": MegaLucarioTRL2,
    "alakazam": AlakazamL2, "doublade": DoubladeL2, "hydrapple": HydrappleL2,
    "rockets_mewtwo": RocketsMewtwoL2, "mamoswine": MamoswineL2,
    "marnie_grimmsnarl": MarnieGrimmsnarlL2, "mega_lucario": MegaLucarioL2, "cynthia_garchomp": CynthiaGarchompL2, "mega_feraligatr": MegaFeraligatrL2, "omatsuri": OmatsuriL2, "ns_zoroark": ZoroarkL2, "ethan_hooh": EthanHoohL2, "manectric": ManectricL2, "mega_venusaur": MegaVenusaurL2, "mega_gardevoir": MegaGardevoirL2, "mega_diancie": MegaDiancieL2, "black_kyurem": BlackKyuremL2, "mega_latias": MegaLatiasL2, "mega_zygarde": MegaZygardeL2, "cubchoo_control": CubchooL2, "slowking_combo": SlowkingComboL2, "slowking_hybrid": HybridSlowkingL2, "slowking_live": SlowkingLiveL2, "config": ConfigL2,
    "metagross": MetagrossL2, "dudunsparce_box": DudunsparceBoxL2,
    "briar": BriarL2, "waitress": WaitressL2, "klinklang": KlinklangL2,
}
_ARCHETYPES = {
    "aggro": AggroPolicy,
    "beatdown": BeatdownPolicy,
    "ramp": RampPolicy,
    "spread": SpreadPolicy,
    "toolbox": ToolboxPolicy,
    "control": ControlPolicy,
    "combo": ComboPolicy,
    "midrange": BeatdownPolicy,   # legacy fallback if a deck is still tagged midrange
}
_POLICY_CACHE = {}


_MIXIN_CACHE = {}


def _with_archetype(cls, archetype):
    """Compose a generic L2 with its deck's ARCHETYPE policy.

    `l2` wins over `archetype` in make_policy, and the generic `config` L2 descends from
    FocusL2 -- so a deck tagged `control` with `l2: config` silently loses ControlPolicy
    entirely: its ladder, its `step_disrupt`, its `_DENIAL` list and its wall/lock/denial
    role inference. Measured 2026-07-20: **41 of 62 decks never run their archetype's
    ladder** (control 5/5, ramp 10/10, toolbox 6/6). comfey_yveltal is the clean case --
    a mill/control deck whose Xerosic (929 offers) / Crushing Hammer (596) / Acerola (118)
    were played **0** times, because the step that plays denial was never in its ladder.

    Opt-in per deck (`arch_mixin: true`) because the existing L2s were tuned against the
    BROKEN resolution; flipping all 41 at once is not validated."""
    arch = _ARCHETYPES.get(archetype or "base", BasePolicy)
    if arch is BasePolicy or issubclass(cls, arch):
        return cls
    key = (cls.__name__, arch.__name__)
    if key not in _MIXIN_CACHE:
        _MIXIN_CACHE[key] = type(f"{cls.__name__}_{arch.__name__}", (cls, arch), {})
    return _MIXIN_CACHE[key]


def make_policy(deck, profile=None):
    profile = profile or {}
    l2 = profile.get("l2")
    if l2 and l2 in _PERDECK:                       # per-deck L2 wins over archetype
        cls = _PERDECK[l2]
        if profile.get("arch_mixin"):
            cls = _with_archetype(cls, profile.get("archetype"))
        p = cls(deck, profile)
    else:
        cls = _ARCHETYPES.get(profile.get("archetype", "base"), BasePolicy)
        p = cls(deck, profile)
    # `ladder` as DATA. choose_main already resolves rules by NAME
    # (`getattr(self, name)(ctx)`), and ConfigL2 already rewrites its own ladder from
    # config -- so a deck's playbook is one ordered list of rule names, which belongs in
    # tuning.json, not in a bespoke subclass. Applied HERE, after every __init__, so it
    # wins over the class attribute and over ConfigL2's in-place ladder edits.
    # Opt-in: a profile without "ladder" changes nothing.
    lad = profile.get("ladder")
    if lad is not None:
        missing = [r for r in lad if not callable(getattr(p, r, None))]
        if missing:                                 # fail loud: a typo'd rule is silent death
            raise ValueError(f"profile ladder names unknown rule(s) {missing} "
                             f"for {type(p).__name__}; known: "
                             f"{sorted(a for a in dir(p) if a.startswith('rule_'))}")
        p.ladder = tuple(lad)
    return p


def act(obs_dict, deck, profile=None, style=None, hints=None, policy=None):
    """Compatibility entry mirroring agents._engine.act's leading args.

    L0 ignores style/hints/policy. A policy is built once per distinct deck object
    and cached (roles are inferred at construction)."""
    key = id(deck)
    p = _POLICY_CACHE.get(key)
    if p is None or p.deck != list(deck):
        p = make_policy(deck, profile)
        _POLICY_CACHE[key] = p
    return p.act(obs_dict)


def load_deck(name):
    """Same loader as the legacy engine (local decks/ or bundled deck.csv)."""
    candidates = [
        os.path.join("decks", name + ".csv"),
        name + ".csv", "deck.csv", "/kaggle_simulations/agent/deck.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return [int(line) for line in f if line.strip()]
    raise FileNotFoundError("deck not found: " + ", ".join(candidates))
