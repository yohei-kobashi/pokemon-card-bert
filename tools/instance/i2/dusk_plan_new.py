#!/usr/bin/env python3
"""The dragapult_dusknoir game plan, written as checkable rules over (state, chosen option).

WHY THIS REPLACES THE WIN-RATE REWARD. The gate reads +-2.5pt at 400 games, which is coarser
than one round's true effect, and the prize term that dominated the old potential is an EFFECT
of winning rather than a cause -- conditioning on it is the trap `lm_mirror_log`'s docstring
and `setup-execution-audit-and-budew-overattack` both name. So the objective becomes: does the
policy MAKE THE PLAYS, measured per decision, with no outcome anywhere in it.

WHAT THAT COSTS, stated plainly: with no win/loss term there is no ground truth left, so a
wrong rule here cannot be caught by measurement -- it will simply be learned. The precedent
that makes it worth it is slowking, where the same reframing took Seek Inspiration from 0 of
40 games to 40 of 40 once execution rather than win rate was the objective.

EACH RULE IS (trigger, conformant set, weight). A rule only scores when its trigger fires, so
the denominator is OPPORTUNITIES, not decisions -- "Budew was never benched" and "Budew was
benched and we declined to lock" are different failures and must not average together.

    PYTHONPATH=cg-lib python3 tools/dusk_plan.py --traces /root/traces_r4.s0.jsonl.gz
"""

import argparse
import collections
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DREEPY, DRAKLOAK, PULT = 119, 120, 121
DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
BUDEW, MUNKIDORI, FEZ, MEOWTH = 235, 112, 140, 1071
# VERIFIED against the card DB, not guessed. The first draft had Phantom Dive at 122 (which
# is not even in this line) and the damage-counter contexts at (5,6,7); all three rules that
# depended on them reported zero opportunities in 600 games -- the same silent-zero failure as
# the Phi_lock term that read a field the observation does not have. A rule that never triggers
# reads as "not measured", never as "not done", which is why the table prints NO TRIGGER.
ITCHY_POLLEN, PHANTOM_DIVE, JET_HEADBUTT = 323, 154, 153
# SelectContext::ToHand.  ToJson.h emits `(int)state.selectContext - 1`, so the enum's 8 arrives
# as 7 -- the same shift that makes the damage-counter menus (13, 14) line up with DamageCounter
# and DamageCounterAny, which is how this was checked rather than assumed.
_TO_HAND_CTX = 7
_PREREQ = {}          # filled below, once the line ids exist
DMG_COUNTER_CTX = (13, 14)          # SelectContext.DAMAGE_COUNTER / DAMAGE_COUNTER_ANY
FIRE, PSY, DARK = 2, 5, 7
PD_DMG = 200        # Phantom Dive's damage -- to the ACTIVE only
PD_COUNTERS = 60    # ... and 6 counters to the BENCH, which is what finishes bench bodies
BOSS_ORDERS = 1182  # the only way a banked BENCH body ever meets the 200
DUSCLOPS, DUSKNOIR = 132, 133   # same self-KO Cursed Blast, 5 counters vs 13
JUDGE, WATCHTOWER = 1213, 1256

_ATK = None


def _attack_damage(attack_id):
    """Printed damage for an attack id, or None. Loaded once from the engine's own table so the
    lethal check reads the same numbers the engine resolves with."""
    global _ATK
    if _ATK is None:
        try:
            import cg.api as _api
            _ATK = {a.attackId: (a.damage or 0) for a in _api.all_attack()}
        except Exception:                                      # noqa: BLE001
            _ATK = {}
    return _ATK.get(attack_id)


def _prizes_for(card):
    """How many prizes the opponent takes for knocking this body out."""
    if card is None:
        return 1
    if getattr(card, "megaEx", False):
        return 3
    return 2 if getattr(card, "ex", False) else 1
_CLOPS_HOLD = os.environ.get("DUSK_CLOPS_HOLD", "") not in ("", "0")
# Same gating as clops_hold: a new rule changes the labels of every RL round that consumes
# them, so it is enabled only by the run that is measuring it.
_NEW_EXCL = os.environ.get("DUSK_NEW_RULES", "") not in ("", "0")
# clops_hold branch (3): let the blast be justified by the attack we can actually PAY for this
# turn, not by Phantom Dive alone. Gated so it can be A/B'd against the shipped behaviour --
# it changes when a prize is conceded, which is the most expensive thing this plan decides.
_BLAST_REACH = os.environ.get("DUSK_BLAST_REACH", "") not in ("", "0")
_FUEL_LOW = 14                      # deck-count gate for draw-adding evolves (anti-deckout)
_BASICS = (DREEPY, DUSKULL, BUDEW, MUNKIDORI, FEZ, MEOWTH)
# what each evolution needs UNDER it before it can ever be played
_PREREQ = {DRAKLOAK: DREEPY, PULT: DRAKLOAK, DUSCLOPS: DUSKULL, DUSKNOIR: DUSCLOPS}
# The engine's turn counter is shared by both seats, so six of its turns is roughly our first
# three -- which is the window the template describes and the window where the measured defect
# lives (with a Drakloak up, the same menus already read 30-40%).
_SETUP_TURNS = 6
_LINE_TARGET = 2          # Dreepy-or-Drakloak bodies before the search moves on

# id -> (rule name, weight). Weights are relative importance INSIDE the plan; there is no
# prize unit any more, because there are no prizes in this objective.
RULES = {
    "lock_early":   ("use Itchy Pollen while still setting up", 1.0),
    "bench_line":   ("put Dreepy on the bench when offered", 1.5),
    "recon":        ("use Recon Directive every turn, unless the deck is nearly out", 1.0),
    "energy_line":  ("attach energy to a Dreepy/Drakloak/Dragapult, not elsewhere", 1.5),
    "energy_focus": ("finish {R}{P} on ONE body before starting a second", 1.0),
    "evolve_line":  ("evolve the line when the evolution is in hand", 1.5),
    "phantom_dive": ("attack with Phantom Dive when it is legal", 2.0),
    "spread_aim":   ("finish a body when the counters reach it; prefer the two-prize one", 2.0),
    "boss_damaged": ("gust a body that is ALREADY in range, not a fresh one", 1.5),
    # NAMED FOR WHAT IT CHECKS. The old description promised counter PLACEMENT, but the
    # trigger only asks whether Munkidori's ability was USED -- it never looks at where the
    # counters went. The placement that follows is a DAMAGE_COUNTER menu and is judged by
    # spread_aim, so the work was covered; the name was not. A rule whose name overstates
    # its test is how a 67.2% reading got read as "counter placement is fine".
    "clops_hold":   ("do not fire Dusclops' Cursed Blast while Dusknoir is in hand", 2.0),
    "judge_timing": ("do not Judge while the opponent holds no more cards than you", 1.5),
    "stadium_replace": ("do not play a Stadium onto an identical Stadium", 1.0),
    "search_bottom": ("do not search out a card whose pre-evolution is not in play", 1.5),
    "setup_search": ("search out the step the line is missing, while it is still short", 2.0),
    "lethal_now":   ("attack when Active + bench knock-outs take the last prizes", 3.0),
    "spare_ex_bench": ("do not bench a body worth the prizes they need to win", 2.0),
    "retreat_energy": ("do not retreat a body carrying {R}/{P}", 1.5),
    "munki_move":   ("use Munkidori's ability when it is available", 1.0),
}


def _slots(ps):
    return list(ps.get("active") or []) + list(ps.get("bench") or [])


def _energies(pk):
    return list((pk or {}).get("energies") or [])


def _can_pd(pk):
    e = _energies(pk)
    for w in (FIRE, PSY):
        if w in e:
            e.remove(w)
        elif 0 in e:
            e.remove(0)
        else:
            return False
    return True


def _energy_type_of(obs, yi, o, text=""):
    """Which energy TYPE this attach option would move. The option names the hand card by
    reference, and the rendered text carries it as `attach:cNNN@SLOT`, so either path resolves."""
    cid = None
    if isinstance(o, dict) and o.get("cardId"):
        cid = o["cardId"]
    if cid is None and text.startswith("attach:c"):
        try:
            cid = int(text.split("attach:c", 1)[1].split("@", 1)[0])
        except (ValueError, IndexError):
            cid = None
    if cid is None:
        cid = _card_of(obs, yi, o)
    return {2: FIRE, 5: PSY, 7: DARK}.get(cid)


def opportunities(obs, seat=None):
    """Which rules are LIVE at this decision, and which option indices satisfy each.

    Returns {rule: (conformant, SCOPE)}. SCOPE is the set of options the rule is entitled to
    judge, and it matters as much as the answer: the plan can say "if you are attaching, attach
    HERE", but it cannot say "attach rather than play a trainer". Scoring every unlabelled
    option as wrong made Fezandipiti's draw, Poke Pad's search and Night Stretcher's recovery
    into negatives -- all correct plays the plan simply has no rule about. Training is
    restricted to each rule's own scope so silence stays silence.
    """
    from lm.actions import encode_option, _card_at
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if not opts:
        return {}
    pl = cur.get("players") or []
    yi = cur.get("yourIndex", 0) if seat is None else seat
    if yi >= len(pl):
        return {}
    me, opp = pl[yi] or {}, pl[1 - yi] or {}
    mine = _slots(me)
    my_ids = [p.get("id") for p in mine if isinstance(p, dict)]
    turn = cur.get("turn") or 0
    out = {}

    def _scope(pred):
        return {i for i, o, t in texts() if pred(i, o, t)}

    def texts():
        for i, o in enumerate(opts):
            try:
                yield i, o, encode_option(o, obs)
            except Exception:                                  # noqa: BLE001
                yield i, o, ""

    # --- hold Dusclops for the Dusknoir upgrade ---------------------------------------------
    # Dusclops and Dusknoir carry the SAME ability: put counters on one of the opponent's
    # Pokemon, then this Pokemon is Knocked Out. Dusclops pays a prize for 6 counters; Dusknoir
    # pays the same prize for 13. Firing the small one while the big one is in hand spends the
    # body for less than half its value at identical cost.
    #
    # This is not a hypothetical. Over 800 replayed games Dusknoir reached the HAND in 90.2% of
    # them, and in 31.8% the pilot fired Dusclops' Cursed Blast with Dusknoir sitting in hand --
    # after which the evolve can never be offered, which is why Dusknoir reached the BOARD in
    # 1.2%. The human list this deck was copied from (Millar, NAIC 2026 13th) runs the same
    # 2/2/1 line, so the 1-of is the plan, not the problem.
    #
    # The rule says only "not this option": every other play on the menu is fine, including
    # ending the turn, because the point is to still HAVE the Dusclops next turn.
    # OFF BY DEFAULT, and deliberately so. A new rule changes the rule-weight labels of every
    # RL round that consumes them, and this one is unproven: the playout probe came back
    # +0.117 on the best alternative (selection-biased, max of 3 noisy branches) but -0.028 on
    # the mean and 62/120 on the sign test. It is enabled only by the experiment that is
    # measuring it, so the mirror chain's reward does not silently move underneath it.
    hand_ids = [h.get("id") if isinstance(h, dict) else h for h in (me.get("hand") or [])]
    if _CLOPS_HOLD:
        # WHEN IS CURSED BLAST WORTH A PRIZE? Every published guide for this deck says the same
        # three things, and the shipped rule encoded only the first:
        #   * hold Dusclops while Dusknoir is in hand -- same prize, 50 damage instead of 130
        #   * Dusknoir is the LATE FINISHER: 200 + 130 = 330 takes a 320-330 HP ex in one turn,
        #     conceding 1 prize to take 2
        #   * the other legitimate use is removing a BENCH body that is still growing into a
        #     threat (Ralts before Gardevoir, Abra before Alakazam) -- and the guides condition
        #     that on the prize race being level or in our favour, because it trades 1 for 1
        # Anything else is a free prize. Hareruya describes the whole game as roughly three
        # Phantom Dives and ONE Cursed Blast; this rule is what makes it one.
        #
        # Our list is more constrained than the reference lists, which is why the bar is set
        # here rather than left to taste: those run Counter Catcher / Iono / Roxanne, so the
        # conceded prize turns on a comeback. Ours runs none of them, and both Unfair Stamp and
        # Fezandipiti ex require a knock-out on the OPPONENT's turn, so a self-KO on our own turn
        # switches nothing on. For us the prize is pure cost.
        #
        # lethal_now returns early and alone, so a blast that WINS is never reached by this rule.
        _blast = {}                     # option index -> counters it would place
        for _bi, _bo, _bt in texts():
            if _bt.startswith("ability:c%d" % DUSCLOPS):
                _blast[_bi] = 5
            elif _bt.startswith("ability:c%d" % DUSKNOIR):
                _blast[_bi] = 13
        if _blast:
            from agents._engine import _CARDS as _CD6
            _opp_pz = len(opp.get("prize") or [])
            _my_pz = len(me.get("prize") or [])
            _oact = (opp.get("active") or [None])[0]
            _obench = [b for b in (opp.get("bench") or []) if isinstance(b, dict)]
            # Phantom Dive has to actually be reachable this turn for the combo test to mean
            # anything: either it is on this menu, or Dragapult is Active and paid for.
            _pd_ready = any(isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE
                            for o in opts)
            if not _pd_ready:
                _ma = (me.get("active") or [None])[0]
                _pd_ready = (isinstance(_ma, dict) and _ma.get("id") == PULT and _can_pd(_ma))

            # The hardest-hitting attack ON THIS MENU, which is the same thing as "the best
            # attack we can pay for": the engine only offers an attack whose cost is met. Taken
            # from the menu rather than from our energy so that no cost model can drift out of
            # step with the engine's. Weakness and resistance are read off the defender, exactly
            # as lethal_now does, so the two rules cannot disagree about what an attack does.
            _best_atk = 0
            if _BLAST_REACH:
                _mact = (me.get("active") or [None])[0]
                _acard = _CD6.get(_mact.get("id")) if isinstance(_mact, dict) else None
                _dcard = _CD6.get(_oact.get("id")) if isinstance(_oact, dict) else None
                for _ao in opts:
                    if not isinstance(_ao, dict) or not _ao.get("attackId"):
                        continue
                    _dd = _attack_damage(_ao["attackId"]) or 0
                    if not _dd:
                        continue
                    if _acard is not None and _dcard is not None:
                        if getattr(_dcard, "weakness", None) == getattr(_acard, "energyType",
                                                                        None):
                            _dd *= 2
                        elif getattr(_dcard, "resistance", None) == getattr(_acard,
                                                                            "energyType", None):
                            _dd -= 30
                    _best_atk = max(_best_atk, _dd)

            def _worth(counters):
                dmg = counters * 10
                # A prize handed over when they need one is the game, whatever it buys.
                if _opp_pz <= 1:
                    return False
                for _t in ([_oact] if isinstance(_oact, dict) else []) + _obench:
                    _hp = _t.get("hp") or 0
                    _c = _CD6.get(_t.get("id"))
                    if _hp <= 0:
                        continue
                    # (1) it kills something worth two prizes: 1 conceded for 2 taken
                    if dmg >= _hp and (_prizes_for(_c) or 1) >= 2:
                        return True
                    # (2) it kills a still-developing BENCH body, at level or better prizes
                    if (dmg >= _hp and _t is not _oact and _my_pz <= _opp_pz
                            and getattr(_c, "name", None) in _evolvable_names()):
                        return True
                    # (3) it is the half that lets an ATTACK finish the job THIS turn.
                    #
                    # This used to assume the attack is Phantom Dive: `_reach` was 200 to the
                    # Active or the six counters to a bench body, gated on Phantom Dive being
                    # payable. Against ogerpon_mono that makes the whole rule dead, and it is
                    # dead for the same reason we lose. Their four Crushing Hammers hold our
                    # Active at 1.05 energy (1.77 vs marnie); Phantom Dive costs two, so it is
                    # offered 1.5 times a game against 3.9, and the deck falls back on 70-damage
                    # Jet Headbutts (44 uses to Phantom Dive's 14 -- the reverse of every other
                    # matchup). So the one damage source they CANNOT deny, an ability that costs
                    # no energy, is withheld by a test that depends on the resource they are
                    # denying: measured 0.30 Cursed Blasts a game here against 0.47 vs marnie.
                    #
                    # `_reach` therefore becomes what we can actually attack with on THIS menu.
                    # Only Phantom Dive reaches the bench, so bench targets still require it.
                    if dmg < _hp:
                        if _BLAST_REACH:
                            _reach = _best_atk if _t is _oact else 0
                            if _pd_ready:
                                _reach = max(_reach,
                                             PD_DMG if _t is _oact else PD_COUNTERS)
                            # The prize condition is NOT optional once `_reach` can be a small
                            # attack. At 200 damage the bodies Phantom Dive finishes are ex-sized
                            # anyway, so branch (3) never needed to say so; at 70 the same branch
                            # would spend a prize to take a 120 HP basic -- one for one, which is
                            # the trade this rule exists to refuse. The bar is the deck's own
                            # doctrine, already written in branch (1): concede 1 to take 2.
                            if _reach and _hp - dmg <= _reach \
                                    and (_prizes_for(_c) or 1) >= 2:
                                return True
                        elif _pd_ready:
                            _reach = PD_DMG if _t is _oact else PD_COUNTERS
                            if _hp - dmg <= _reach:
                                return True
                return False

            _bad = set()
            for _bi, _cnt in _blast.items():
                # the upgrade guard, unchanged and first: firing the 5 while the 13 is in hand
                # spends the body for under half its value at identical cost
                if _cnt == 5 and DUSKNOIR in hand_ids and DUSCLOPS in my_ids:
                    _bad.add(_bi)
                elif not _worth(_cnt):
                    _bad.add(_bi)
            if _bad and len(_bad) < len(opts):
                out["clops_hold"] = (set(range(len(opts))) - _bad, set(range(len(opts))))

    # --- Judge: only when it does not hand the opponent cards -------------------------------
    # Judge shuffles BOTH hands away and draws BOTH players 4, so its card economy is exactly
    #     (4 - my hand) - (4 - their hand) = their hand - my hand
    # and with their handCount visible that is a subtraction, not a read. The rule EXCLUDES only
    # the arithmetically-losing case -- playing it while they hold no more than we do gives them
    # the cards. Everything else is left to the model, because the reason to Judge at a card
    # LOSS is board-dependent: alakazam's Powerful Hand places 2 damage counters PER CARD IN ITS
    # HAND, so cutting their hand can be worth more than the cards it costs, and no arithmetic
    # here can see that.
    if _NEW_EXCL:
        jud = _scope(lambda i, o, t: t.startswith("play:c%d" % JUDGE))
        opp_hand = opp.get("handCount")
        my_hand = len(me.get("hand") or [])
        if jud and opp_hand is not None and opp_hand <= my_hand and len(opts) > len(jud):
            out["judge_timing"] = (set(range(len(opts))) - jud, set(range(len(opts))))

        # --- Stadium: never overwrite the same Stadium ---------------------------------------
        # Only one Stadium is in play at a time, so ours is worth a card when it REPLACES
        # something -- marnie's Spikemuth Gym (4 copies, a free "Marnie's Pokemon" tutor every
        # turn) or ogerpon's Lively Stadium (+30 HP to every Basic, which pushes Teal Mask
        # Ogerpon ex out of Phantom Dive's range). Playing it onto an identical Stadium changes
        # nothing and costs the card; that, and only that, is excluded. When to spend the second
        # copy rather than hold it stays with the model.
        std = _scope(lambda i, o, t: t.startswith("play:c%d" % WATCHTOWER))
        in_play = {(x or {}).get("id") for x in (cur.get("stadium") or []) if isinstance(x, dict)}
        if std and WATCHTOWER in in_play and len(opts) > len(std):
            out["stadium_replace"] = (set(range(len(opts))) - std, set(range(len(opts))))

    # --- lethal: the one time attacking beats developing ------------------------------------
    # Nothing in this plan mentioned PRIZES until now, and the attack rules are deliberately
    # added last -- "attacking ends the turn, so do not demand it on a menu that also offers
    # development". That ordering is right except in the one case where the turn does not need
    # to be developed because the game ends: if this attack knocks the Active out and that
    # knock-out takes our last prizes, every other rule on the menu is advice about a game that
    # is already over. So this one is computed FIRST and, when it fires, it is the only rule
    # returned.
    if _NEW_EXCL:
        from agents._engine import _CARDS as _CD3
        # EVERY way this deck can take its last prizes this turn, not just Phantom Dive.
        #
        # The first version credited bench knock-outs only to Phantom Dive's six counters. That
        # is one of four sources: Dusclops' Cursed Blast puts 5 counters on ANY one of the
        # opponent's Pokemon, Dusknoir's puts 13, and Munkidori's Adrena-Brain moves up to 3 from
        # one of ours to one of theirs. All three are ABILITIES -- they do not end the turn, so
        # they chain with each other and with the attack, and a line like "Dusknoir's 130 kills a
        # benched Kadabra, then attack the Active" was invisible to a rule that only read the
        # attack menu.
        #
        # Two costs the search has to respect:
        #   * Cursed Blast KNOCKS OUT THE USER, handing the opponent a prize. A line that brings
        #     them to zero loses the game before ours resolves, so it is refused outright.
        #   * An attack ends the turn. When a winning line needs an ability first, only the
        #     ABILITY is nominated; the attack half is nominated on the next menu, once the
        #     counters have actually landed and the board says so.
        _need = len(me.get("prize") or [])
        _opp_prizes = len(opp.get("prize") or [])
        _oa = (opp.get("active") or [None])[0]
        if _need and isinstance(_oa, dict):
            _def = _CD3.get(_oa.get("id"))
            # targets: index 0 is the Active, 1.. are the bench, as (hp, prizes it yields)
            _tgt = [((_oa.get("hp") or 0), _prizes_for(_def))]
            for _b in (opp.get("bench") or []):
                if isinstance(_b, dict):
                    _tgt.append(((_b.get("hp") or 0), _prizes_for(_CD3.get(_b.get("id")))))

            # counter packets offered on THIS menu: (option index, counters, prizes conceded)
            _packs = []
            for _pi, _po, _pt in texts():
                if not (_pt.startswith("abl") or _pt.startswith("ability")):
                    continue
                _who = _field_id(obs, yi, _po)
                if _who == DUSKNOIR:
                    _packs.append((_pi, 13, _prizes_for(_CD3.get(DUSKNOIR)) or 1))
                elif _who == DUSCLOPS:
                    _packs.append((_pi, 5, _prizes_for(_CD3.get(DUSCLOPS)) or 1))
                elif _who == MUNKIDORI:
                    # Adrena-Brain MOVES counters, so it can only supply what is already on one
                    # of ours -- an undamaged board makes this ability worth zero here.
                    _mv = 0
                    for _q in mine:
                        if isinstance(_q, dict):
                            _mv = max(_mv, min(3, ((_q.get("maxHp") or 0)
                                                   - (_q.get("hp") or 0)) // 10))
                    if _mv:
                        _packs.append((_pi, _mv, 0))

            _myact = (me.get("active") or [None])[0]
            _att = _CD3.get((_myact or {}).get("id")) if isinstance(_myact, dict) else None
            _atks = []          # (option index, damage on the Active, spreads 6 to the bench)
            for _ai, _ao in enumerate(opts):
                if not isinstance(_ao, dict) or not _ao.get("attackId"):
                    continue
                _d = _attack_damage(_ao["attackId"])
                if not _d:
                    continue
                if _att is not None and _def is not None:
                    if getattr(_def, "weakness", None) == getattr(_att, "energyType", None):
                        _d *= 2
                    elif getattr(_def, "resistance", None) == getattr(_att, "energyType", None):
                        _d -= 30
                _atks.append((_ai, _d, _ao["attackId"] == PHANTOM_DIVE))

            def _spread(res, cap):
                """Best prizes from `cap` counters spread over residual bench bodies. Exact:
                at most five bodies, so brute force over subsets costs nothing."""
                import itertools as _it
                items = [(((h + 9) // 10), pz) for h, pz in res if h > 0]
                best = 0
                for _r in range(1, len(items) + 1):
                    for _c in _it.combinations(items, _r):
                        if sum(x for x, _v in _c) <= cap:
                            best = max(best, sum(v for _x, v in _c))
                return best

            def _clears(res, cap):
                """Can `cap` counters finish EVERY surviving bench body? A different question
                from _spread, which maximises prizes and will happily leave a body standing."""
                return sum(((h + 9) // 10) for h in res if h > 0) <= cap

            # Our own board must survive the plan too: Cursed Blast knocks the user out, and
            # emptying our OWN bench loses the game just as surely as emptying theirs wins it.
            _my_bodies = sum(1 for _q in mine if isinstance(_q, dict))

            # Search: which abilities to fire, where each lands, and which attack to follow with.
            # <= 3 packets x 6 targets x <= 5 attacks is a few thousand states.
            import itertools as _it2
            _win_abl, _win_atk = set(), set()
            for _use in range(1 << len(_packs)):
                _chosen = [_packs[k] for k in range(len(_packs)) if _use >> k & 1]
                _conceded = sum(c for _i, _n, c in _chosen)
                # a self-KO that empties their prize count wins the game for them, not us
                if _conceded and _conceded >= _opp_prizes:
                    continue
                _selfko = sum(1 for _i, _n, c in _chosen if c)
                if _my_bodies - _selfko < 1:
                    continue           # the last Cursed Blast would empty our own board
                for _where in _it2.product(range(len(_tgt)), repeat=len(_chosen)):
                    _hp = [h for h, _pz in _tgt]
                    for (_oi, _cnt, _c), _w in zip(_chosen, _where):
                        _hp[_w] -= _cnt * 10
                    _got = sum(_tgt[t][1] for t in range(len(_tgt))
                               if _tgt[t][0] > 0 >= _hp[t])
                    _alive = [t for t in range(len(_tgt)) if _hp[t] > 0]
                    if _got >= _need or not _alive:
                        # abilities alone close it, on prizes or by clearing their board --
                        # nominate them and never spend the turn
                        _win_abl.update(_oi for _oi, _n, _c in _chosen)
                        continue
                    for _ai, _d, _is_pd in _atks:
                        _tot = _got
                        _act_dead = _hp[0] <= 0 or _d >= _hp[0] > 0
                        if _hp[0] > 0 and _d >= _hp[0]:
                            _tot += _tgt[0][1]
                        _bench_res = [_hp[t] for t in range(1, len(_tgt))]
                        if _is_pd:
                            _tot += _spread([(_hp[t], _tgt[t][1])
                                             for t in range(1, len(_tgt))
                                             if _tgt[t][0] > 0 < _hp[t]], 6)
                        # WIN CONDITION 2: they have nothing left to promote. The attack must
                        # kill the Active, and every surviving bench body must fall to the six
                        # counters -- which is _clears, not _spread: maximising prizes can leave
                        # exactly the one body that keeps them alive.
                        _wipe = _act_dead and (_clears(_bench_res, 6) if _is_pd
                                               else not any(h > 0 for h in _bench_res))
                        if _tot >= _need or _wipe:
                            if _chosen:
                                _win_abl.update(_oi for _oi, _n, _c in _chosen)
                            else:
                                _win_atk.add(_ai)
            # An ability-first line is preferred whenever one exists: it does not end the turn,
            # so taking it keeps every attack still on the table for the next menu.
            kill = _win_abl or _win_atk
            if kill and len(kill) < len(opts):
                return {"lethal_now": (kill, set(range(len(opts))))}

    # --- do not hand over the prizes that lose the game -------------------------------------
    # A benched {ex} is a two-prize target that has to be defended for the rest of the game.
    # That is a fair trade early; it is not one when the opponent is two prizes from winning,
    # because the body we just put down is exactly the last two prizes they need. Counting
    # only -- their remaining prizes, and what each candidate would be worth.
    if _NEW_EXCL:
        from agents._engine import _CARDS as _CD4
        _their_prize = len(opp.get("prize") or [])
        if 0 < _their_prize <= 2:
            rich = set()
            for i, o, t in texts():
                if not t.startswith("play"):
                    continue
                cid = _card_of(obs, yi, o)
                c = _CD4.get(cid)
                if c is not None and c.hp and _prizes_for(c) >= _their_prize:
                    rich.add(i)
            if rich and len(rich) < len(opts):
                out["spare_ex_bench"] = (set(range(len(opts))) - rich,
                                         set(range(len(opts))))

    # --- do not retreat the body the energy was spent on -------------------------------------
    # Retreating discards Energy equal to the retreat cost, and Phantom Dive's {R}{P}+1 takes
    # several turns to assemble on one body. Retreating a loaded attacker throws those turns
    # away. Only the loaded case is excluded -- retreating an empty body is free and often
    # right, and the rule says nothing about it.
    if _NEW_EXCL:
        from agents._engine import _CARDS as _CD5
        act = (me.get("active") or [None])[0]
        if isinstance(act, dict):
            c = _CD5.get(act.get("id"))
            loaded = any(e in (FIRE, PSY) for e in _energies(act))
            if c is not None and (getattr(c, "retreatCost", 0) or 0) > 0 and loaded:
                ret = _scope(lambda i, o, t: t == "retreat")
                if ret and len(ret) < len(opts):
                    out["retreat_energy"] = (set(range(len(opts))) - ret,
                                             set(range(len(opts))))

    # --- attacks: collected here, but ADDED LAST -------------------------------------------
    # ATTACKING ENDS THE TURN. On a MAIN menu the plays, evolutions and abilities all come
    # first and the attack is the last thing done, so demanding the attack at a menu that also
    # offers development is demanding the turn be thrown away. Measured: 2,693 of 5,908
    # phantom_dive opportunities co-fired with bench_line / energy_line / recon / evolve_line,
    # and the rule was scoring the policy WRONG for correctly developing first. Same shape as
    # boss_damaged preferring a fresh 70 HP body -- a broken rule mis-diagnoses good play.
    atk = {i: o for i, o in enumerate(opts) if isinstance(o, dict) and o.get("attackId")}
    # The scope of an attack rule is the TURN-ENDING choices, not just the attacks. Limiting
    # it to attacks alone wiped lock_early out entirely: Itchy Pollen is only legal when Budew
    # is Active, and Budew has no other attack, so every option in scope conformed and the
    # decision was discarded as teaching nothing. The real question there is "lock, or pass and
    # keep building" -- which cannot be asked without `end` in the comparison.
    _end = {i for i, _o, t in texts() if t in ("end", "retreat")}
    _atk_scope = set(atk) | _end
    _attack_rules = {}
    if atk:
        pd = {i for i, o in atk.items() if o["attackId"] == PHANTOM_DIVE}
        if pd:
            _attack_rules["phantom_dive"] = (pd, _atk_scope)
        pol = {i for i, o in atk.items() if o["attackId"] == ITCHY_POLLEN}
        # The lock is worth a turn only while we are still building; late it is a wasted turn.
        if pol and turn <= 6 and PULT not in my_ids:
            _attack_rules["lock_early"] = (pol, _atk_scope)

    # --- energy attachment ----------------------------------------------------------------
    att = [(i, o, t) for i, o, t in texts() if t.startswith("attach")]
    if att:
        # WHERE an energy belongs depends on WHICH energy it is. {R} and {P} pay Phantom
        # Dive and belong on the line; {D} pays nothing there and exists solely to switch on
        # Munkidori's Adrena-Brain, so sending it to a Dreepy is a wasted attachment that the
        # first version of this rule scored as correct.
        line = set()
        for i, o, t in att:
            giving = _energy_type_of(obs, yi, o, t)
            area, idx = o.get("inPlayArea", o.get("area")), o.get("inPlayIndex", o.get("index"))
            try:
                pk = (me.get("active") or [None])[0] if area == 1 else (me.get("bench") or [])[idx]
            except (IndexError, TypeError):
                continue
            if not isinstance(pk, dict):
                continue
            tgt = pk.get("id")
            if giving == DARK:
                if tgt == MUNKIDORI and DARK not in _energies(pk):
                    line.add(i)
            elif giving in (FIRE, PSY) and tgt in (DREEPY, DRAKLOAK, PULT):
                line.add(i)
        if line:
            out["energy_line"] = (line, {i for i, _o, _t in att})
            # Finish one body -- and this rule has to read BOTH types: the one already on the
            # body and the one being attached. The first version asked only "does this body
            # hold any energy", which marked a Dreepy holding a lone {D} as one attachment
            # from ready. {D} pays nothing toward Phantom Dive's {R}{P}, so that target can
            # never complete, and the rule was teaching a play that cannot work.
            # c2 = basic {R}, c5 = basic {P}, c7 = basic {D}; only the first two matter here.
            finishers, partials = set(), set()
            for i, o, t in att:
                giving = _energy_type_of(obs, yi, o, t)
                if giving is None:
                    continue
                area, idx = o.get("inPlayArea", o.get("area")), o.get("inPlayIndex", o.get("index"))
                try:
                    pk = (me.get("active") or [None])[0] if area == 1 else (me.get("bench") or [])[idx]
                except (IndexError, TypeError):
                    continue
                if not isinstance(pk, dict) or pk.get("id") not in (DREEPY, DRAKLOAK, PULT):
                    continue
                after = dict(pk)
                after["energies"] = _energies(pk) + [giving]
                if _can_pd(after):
                    finishers.add(i)            # this attachment COMPLETES {R}{P}
                elif giving in (FIRE, PSY) and not any(x in (FIRE, PSY) for x in _energies(pk)):
                    partials.add(i)             # first useful energy on an empty body
            if not any(_can_pd(p) for p in mine if isinstance(p, dict)):
                if finishers:
                    out["energy_focus"] = (finishers, {i for i, _o, _t in att})
                elif partials:
                    out["energy_focus"] = (partials, {i for i, _o, _t in att})

    # --- deck searches that put the card in HAND ------------------------------------------
    # See the RULES entry. Poffin (context ToBench) already takes a Dreepy 94% of the time; this
    # is the other search path, and on our first turn with no Drakloak in play it takes the
    # Dreepy 5% of the time and Dragapult ex or Dusknoir instead.
    if sel.get("context") == _TO_HAND_CTX:
        bad, ready = set(), set()
        for i, o in enumerate(opts):
            if not isinstance(o, dict):
                continue
            cid = _card_at(o, obs)
            pre = _PREREQ.get(cid)
            if pre is not None and pre not in my_ids:
                bad.add(i)                       # unplayable until the step below it exists
            elif cid in _PREREQ or cid in (DREEPY, DUSKULL):
                ready.add(i)                     # a line card we could actually use
        # `ready` is the guard: forbidding the top of the line when nothing better is on the
        # menu would only push the pick onto a Trainer, which this rule has no opinion about.
        if bad and ready and len(bad) < len(opts):
            out["search_bottom"] = (set(range(len(opts))) - bad, set(range(len(opts))))

        # ... and the positive half: name the step the board is missing. Ordered, because the
        # template is ordered -- a second Drakloak is worth less than a first Dreepy, and
        # Duskull only matters once the Dragapult line can actually run.
        if turn <= _SETUP_TURNS:
            want = None
            if my_ids.count(DREEPY) + my_ids.count(DRAKLOAK) < _LINE_TARGET:
                want = DREEPY
            elif DRAKLOAK not in my_ids and PULT not in my_ids:
                want = DRAKLOAK
            elif DUSKULL not in my_ids:
                want = DUSKULL
            if want is not None:
                got = {i for i, o in enumerate(opts)
                       if isinstance(o, dict) and _card_at(o, obs) == want}
                if got and len(got) < len(opts):
                    out["setup_search"] = (got, set(range(len(opts))))

    # --- benching / evolving ---------------------------------------------------------------
    play = [(i, o, t) for i, o, t in texts() if t.startswith("play")]
    if play:
        # The rule is named "put Dreepy on the bench" and used to accept Duskull or Budew for
        # it. That is right once the line is up and wrong before: the template wants three
        # Dreepy first, and this deck runs a single Budew whose lock is worth a turn only while
        # something else is developing. Below the target, only Dreepy conforms.
        _want_early = ((DREEPY,) if my_ids.count(DREEPY) + my_ids.count(DRAKLOAK) < _LINE_TARGET
                       else (DREEPY, DUSKULL, BUDEW))
        b = {i for i, o, _ in play
             if isinstance(o, dict) and _card_of(obs, yi, o) in _want_early}
        if b and len([p for p in mine if isinstance(p, dict) and p.get("id")]) < 5:
            out["bench_line"] = (b, {i for i, _o, _t in play
                              if _card_of(obs, yi, _o) in _BASICS})
    ev = [(i, o, t) for i, o, t in texts() if t.startswith("evolve")]
    if ev:
        # Evolving into Drakloak is always right (it ADDS Recon Directive). Evolving Drakloak
        # into Dragapult is not: the guides are explicit that a Drakloak left alone keeps
        # drawing, so the step is worth taking when the body can then ATTACK, or when we have
        # no Dragapult at all. A blanket "always evolve" trades the draw engine for nothing.
        e = set()
        have_pult = PULT in my_ids
        # Deck-fuel gate. Drakloak ADDS Recon Directive, so a second one is another draw engine,
        # and draw is what runs a deck out. engine_v2 carries the measured version of exactly
        # this (_FUEL_LOW = 14, suppress draw-adding evolves once the engine is already up):
        # adding it moved crustle_stall 54% -> 61% with no regression on the field. The shape is
        # copied deliberately -- only the draw-ADDING evolve is gated, and only once a Drakloak
        # is already in play, so the line still assembles normally.
        # `or 99` would be wrong here: deckCount == 0 is falsy, and 0 is the one value where
        # the gate matters MOST -- the next Recon draw is the loss.
        _dc = me.get("deckCount")
        low_fuel = (_dc is not None and _dc <= _FUEL_LOW) and DRAKLOAK in my_ids
        for i, o, t in ev:
            into = _card_of(obs, yi, o)
            if into == DRAKLOAK and low_fuel:
                continue
            if into in (DRAKLOAK, DUSCLOPS, DUSKNOIR):
                e.add(i)
            elif into == PULT:
                area, idx = o.get("inPlayArea", o.get("area")), o.get("inPlayIndex", o.get("index"))
                try:
                    pk = (me.get("active") or [None])[0] if area == 1 else (me.get("bench") or [])[idx]
                except (IndexError, TypeError):
                    pk = None
                if not have_pult or (isinstance(pk, dict) and _can_pd(pk)):
                    e.add(i)
        if e:
            out["evolve_line"] = (e, {i for i, _o, _t in ev})

    # --- abilities: Recon Directive, Adrena-Brain -------------------------------------------
    abl = [(i, o, t) for i, o, t in texts() if t.startswith("abl") or t.startswith("ability")]
    if abl:
        rec = {i for i, o, _ in abl if _field_id(obs, yi, o) == DRAKLOAK}
        # DECK-COUNT GUARD, and it is the difference between a rule and a deck-out. Recon
        # Directive digs, and "use it every turn" run to the end of a thin deck is how the
        # alakazam pilot lost games it was winning ([[alakazam-deckout-fix]], _FUEL_LOW = 14).
        # Declining late is the one time the pilot is RIGHT to decline, and without this the
        # rule would score that correct decline as a miss -- and, deferred, would force the
        # loss instead of merely mismeasuring it.
        _dcr = me.get("deckCount")
        if rec and not (_dcr is not None and _dcr <= _FUEL_LOW):
            out["recon"] = (rec, {i for i, _o, _t in abl})
        mk = {i for i, o, _ in abl if _field_id(obs, yi, o) == MUNKIDORI}
        if mk:
            out["munki_move"] = (mk, {i for i, _o, _t in abl})

    # --- Boss's Orders: gust something ALREADY in range --------------------------------------
    # The plan is to cash banked damage, not to drag a fresh 320 HP body into the Active Spot.
    #
    # CONTEXT-GUARDED, and the guard is measured, not stylistic. "Any option pointing at an
    # opponent body" also describes every damage-counter placement menu, and on 600 replayed
    # games this rule fired 2,169 times of which 1,918 (88%) were counter placements -- where
    # "prefer the body already at <=200" is the OPPOSITE of spread_aim's label on the same
    # menu (put counters where they bring a NEW body into range; piling onto a dead body
    # wastes them). The two rules directly contradicted each other on 210 menus, which is
    # also the likely reason boss_damaged was the one rule that resisted memorisation (79.5%
    # at lr 1e-5): most of its 8,241 training rows were counter placements labelled with gust
    # semantics. Excluding the two VERIFIED damage-counter contexts rather than allowlisting
    # unverified gust contexts keeps the no-guessing rule this file is written under.
    gust = [(i, o) for i, o in enumerate(opts)
            if isinstance(o, dict) and o.get("playerIndex") not in (None, yi)
            and _field_id(obs, yi, o) is not None]
    if sel.get("context") in DMG_COUNTER_CTX:
        gust = []
    if gust and len(gust) > 1:
        # "In range" is not enough: a FRESH 70 HP Dreepy is in range and is worth one prize
        # for a whole turn, while the 320 HP ex we have been spreading onto is worth two. The
        # rule exists to CASH BANKED DAMAGE, so the body must actually be damaged, and an ex
        # outranks a non-ex among those.
        from agents._engine import _CARDS as _CD
        ready, ready_ex = set(), set()
        for i, o in gust:
            try:
                pk = (opp.get("bench") or [])[o.get("inPlayIndex", o.get("index"))]
            except (IndexError, TypeError):
                continue
            if not isinstance(pk, dict):
                continue
            hp, mx = pk.get("hp") or 0, pk.get("maxHp") or 0
            if hp <= PD_DMG and hp < mx:
                ready.add(i)
                c = _CD.get(pk.get("id"))
                if c is not None and (c.ex or c.megaEx):
                    ready_ex.add(i)
        best = ready_ex or ready
        if best and len(best) < len(gust):
            out["boss_damaged"] = (best, {i for i, _o in gust})

    # --- counter placement: only the part that is a subtraction ------------------------------
    # THREE VERSIONS OF THIS RULE HAVE BEEN WRONG, and each failed the same way: it tried to
    # decide the whole placement.
    #   v1 graded every placement by "does it bring the body within 200" -- but Phantom Dive's
    #      200 hits the ACTIVE and its 6 counters hit the BENCH, so it measured a finisher that
    #      never reaches. 0.51x chance.
    #   v2 tiered by reachability and took only the top tier, so a killable 30 HP Budew made
    #      setting up the Morgrem that gates three Grimmsnarl ex score as WRONG. 0.87x.
    #   v3 filtered out "unreachable" targets. Measured against what actually left play over 800
    #      games: 39-48% of its exclusions were WRONG at every horizon tried. The cash-in paths
    #      are Phantom Dive's counters, Boss's Orders + 200, Dusclops' 5, Dusknoir's 13,
    #      Munkidori moving 3, Cruel Arrow's 100, and the opponent simply promoting the body --
    #      the last of which is not computable at all.
    #
    # So the rule now claims only what a subtraction can support: when THIS placement finishes a
    # body, and finishing it matters most. `hp <= 10 * remain` needs no damage model and no
    # forecast. Everything else -- which body gates their engine, when to bank instead of cash --
    # stays with the model, which was already putting 59% of its counters on bodies that later
    # left play.
    if sel.get("context") in DMG_COUNTER_CTX and sel.get("remainDamageCounter"):
        from agents._engine import _CARDS as _CD2
        remain = int(sel.get("remainDamageCounter") or 0)
        # EVERY damage source still on the board, not just the counters in hand. A body at 100
        # with 5 counters left is not out of reach when a live Dusknoir is standing next to it.
        #
        # The "hp > 0" test is what keeps this honest: Cursed Blast knocks its own user out, and
        # the engine shows that body at 0 HP DURING its own placement menu -- so the Dusclops
        # whose counters we are placing right now is excluded automatically, and only OTHER
        # copies count. Without that check the rule would credit the same 50 twice.
        extra = 0
        for pk in mine:
            if not isinstance(pk, dict) or (pk.get("hp") or 0) <= 0:
                continue
            pid = pk.get("id")
            if pid == DUSCLOPS:
                extra += 50                     # Cursed Blast, 5 counters
            elif pid == DUSKNOIR:
                extra += 130                    # Cursed Blast, 13 counters
            elif pid == MUNKIDORI and DARK in _energies(pk):
                extra += 30                     # Adrena-Brain moves up to 3 counters across
        # Phantom Dive's counters go to the BENCH only, so if the opponent's Active is on this
        # menu at all the sequence is a Cursed Blast or Adrena-Brain -- the attack has NOT been
        # made yet and its damage is still available this turn. That is the only case where an
        # attack may be added, and only to the Active, which is the only thing it can hit.
        _oa = (opp.get("active") or [None])[0]
        _act_serial = (_oa or {}).get("serial") if isinstance(_oa, dict) else None
        _my_act = (me.get("active") or [None])[0]
        atk_bonus = 0
        if isinstance(_my_act, dict) and _can_pd(_my_act):
            atk_bonus = PD_DMG
        now, rich, pz = set(), set(), {}
        for i, o in enumerate(opts):
            if not isinstance(o, dict) or o.get("playerIndex") in (None, yi):
                continue
            area = o.get("inPlayArea", o.get("area"))
            idx = o.get("inPlayIndex", o.get("index"))
            try:
                pk = ((opp.get("active") or [None])[0] if area == 1
                      else (opp.get("bench") or [])[idx])
            except (IndexError, TypeError):
                continue
            if not isinstance(pk, dict):
                continue
            hp = pk.get("hp") or 0
            reach = 10 * remain + extra
            if _act_serial is not None and pk.get("serial") == _act_serial:
                reach += atk_bonus
            if 0 < hp <= reach:
                now.add(i)
                c = _CD2.get(pk.get("id"))
                pz[i] = _prizes_for(c)
                if c is not None and (c.ex or c.megaEx):
                    rich.add(i)
        # Their last prize is the one that ends the game, so a knock-out available now is not a
        # tempo choice any more. Below that, taking A kill is still usually right but WHICH kill
        # is a judgement, and the only part of it the plan can defend is that two prizes beat
        # one -- so it speaks only when an {ex} is among the killable.
        aim = set()
        # The placement side of the multi-KO lethal: if finishing THIS body takes our last
        # prizes, that placement wins the game and outranks everything, including the ex
        # preference -- a 1-prize kill that ends the game beats a 2-prize kill that doesn't.
        _my_left = len(me.get("prize") or [])
        win_now = {i for i in now if _my_left and pz.get(i, 1) >= _my_left}
        if win_now:
            aim = win_now
        elif now and len(opp.get("prize") or []) <= 1:
            aim = rich or now
        elif rich and len(rich) < len(now):
            aim = rich
        if aim and len(aim) < len(opts):
            out["spread_aim"] = (aim, set(range(len(opts))))

    # The turn-ending gate: an attack is only the right answer once nothing else on this menu
    # is. Phantom Dive is still worth 2.0 when it fires -- it just stops firing on a menu where
    # a Dreepy is waiting to be benched.
    if _attack_rules and not out:
        out.update(_attack_rules)
    return out


def _card_of(obs, yi, o):
    cur = obs.get("current") or {}
    pl = cur.get("players") or []
    if o.get("cardId"):
        return o["cardId"]
    try:
        h = (pl[yi] or {}).get("hand") or []
        return (h[o.get("index")] or {}).get("id")
    except (IndexError, TypeError):
        return None



_EVOLVABLE = None


def _evolvable_names():
    """Names that SOMETHING evolves from -- i.e. bodies still developing into a real attacker.

    Cursed Blast is worth spending on Ralts before it is Gardevoir ex and on Abra before it is
    Alakazam; it is not worth spending on a body that is already everything it will ever be.
    `evolvesFrom` carries the pre-evolution NAME, so the set of "can still grow" cards is just
    the image of that field over the whole card DB.
    """
    global _EVOLVABLE
    if _EVOLVABLE is None:
        from agents._engine import _CARDS as _CD
        _EVOLVABLE = {getattr(c, "evolvesFrom", None) for c in _CD.values()
                      if getattr(c, "evolvesFrom", None)}
    return _EVOLVABLE


def _field_id(obs, yi, o):
    cur = obs.get("current") or {}
    pl = cur.get("players") or []
    area = o.get("inPlayArea", o.get("area"))
    idx = o.get("inPlayIndex", o.get("index"))
    try:
        ps = pl[o.get("playerIndex", yi)] or {}
        if area == 1:
            return ((ps.get("active") or [None])[0] or {}).get("id")
        return ((ps.get("bench") or [])[idx] or {}).get("id")
    except (IndexError, TypeError):
        return None


def score(obs, picks, seat=None):
    """(hits, opportunities) per rule for one decision, given the indices actually chosen."""
    live = opportunities(obs, seat)
    chosen = set(picks if isinstance(picks, (list, tuple)) else [picks])
    return ({r: (1 if (c & chosen) else 0) for r, (c, _s) in live.items()},
            {r: 1 for r in live})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="comma-separated trace files")
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--mirror-so", default="")
    a = ap.parse_args()

    import library
    from mirror_env import DEFAULT_SO, MirrorEngine
    eng = MirrorEngine(a.mirror_so or DEFAULT_SO)
    ids = {}

    def deck_ids(n):
        if n not in ids:
            ids[n] = [int(x) for x in open(library.deck_path(n)) if x.strip()]
        return ids[n]

    hit, opp_n = collections.Counter(), collections.Counter()
    games = 0
    for path in [p for p in a.traces.split(",") if p]:
        for line in gzip.open(path, "rt"):
            d = json.loads(line)
            if d.get("header"):
                continue
            d0 = d.get("deck0") or d.get("deck")
            d1 = d.get("deck1") or d.get("deck")
            if a.deck not in (d0, d1):
                continue
            seat = 0 if d0 == a.deck else 1
            obs = eng.start(deck_ids(d0), deck_ids(d1), d["seed"], mirror=1)
            games += 1
            try:
                for pick in d["picks"]:
                    if obs is None:
                        break
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1 or obs.get("select") is None:
                        break
                    if cur.get("yourIndex") == seat and pick is not None:
                        h, n = score(obs, pick, seat)
                        hit.update(h)
                        opp_n.update(n)
                    if pick is None:
                        break
                    obs = eng.select(pick)
            finally:
                eng.finish()
    print("EXECUTION over %d games -- no win rate anywhere in this table\n" % games)
    print("%-14s %8s %10s %9s  %s" % ("rule", "taken", "chances", "rate", "what it is"))
    for r, (name, w) in sorted(RULES.items(), key=lambda kv: -kv[1][1]):
        n = opp_n.get(r, 0)
        if not n:
            print("%-14s %8s %10d %9s  %s" % (r, "-", 0, "NO TRIGGER", name))
            continue
        print("%-14s %8d %10d %8.1f%%  %s" % (r, hit.get(r, 0), n, 100.0 * hit[r] / n, name))


if __name__ == "__main__":
    main()
