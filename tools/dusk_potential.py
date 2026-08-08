#!/usr/bin/env python3
"""Phi(s) for dragapult_dusknoir: the shaping potential of docs/rl_dusknoir_design.md.

NO engine_v2. Nothing here runs a policy; it reads a board and returns a number. The number is
used as a POTENTIAL, i.e. the per-step reward is gamma*Phi(s') - Phi(s), which leaves the
optimal policy unchanged whatever Phi says (Ng, Harada & Russell 1999). That property is the
whole reason the terms below may be opinionated: a wrong Phi costs sample efficiency and cannot
make the agent prefer a losing line. `shaping-potential-refuted` is the reason it is written
this way -- a generic evaluator used AS the signal picked the better move at chance.

One prize is the unit. Every other term answers "how much of a prize is this worth?".

    PYTHONPATH=cg-lib python3 tools/dusk_potential.py --selftest
    PYTHONPATH=cg-lib python3 tools/dusk_potential.py --check-traces /root/traces_r4.s0.jsonl.gz
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

DRAGAPULT_EX, DRAKLOAK, DREEPY = 121, 120, 119
DUSKNOIR, DUSCLOPS, DUSKULL = 133, 132, 131
PHANTOM_DIVE_DMG = 200                 # {R}{P}: 200 to the Active + 6 counters on their bench

W_PRIZE = 1.00                         # the unit
W_READY = 0.45                         # a body a single Phantom Dive now removes
W_PROG = 0.15                          # partial progress toward that threshold
READY_CAP = 3                          # stop rewarding a spread we can never cash
W_PAYER1, W_PAYER2 = 0.20, 0.10        # the Crispin split: a SECOND payer, not more energy
# Partial credit for the FIRST energy of the pair. The guides are explicit that "games where
# you miss the first few early attachments are the toughest", and counting only completed
# payers gave those attachments a gradient of exactly zero -- the same blind spot Dreepy had.
# A half-charged body is worth well under a charged one, so 0.06 against 0.20.
W_HALF, HALF_CAP = 0.06, 2
W_MUNKI_D = 0.05                       # one {D} on Munkidori turns on Adrena-Brain
MUNKIDORI = 112
LINE = (119, 120, 121)                 # Dreepy / Drakloak / Dragapult ex
# The evolution line, valued by STAGE. Dreepy was missing entirely in the first draft, which
# left the deck's most important early action -- Buddy-Buddy Poffin for two Dreepy on turn 1 --
# with no gradient at all: the potential rewarded arriving at two Drakloak without rewarding
# any step on the way there.
W_STAGE = {119: 0.05, 120: 0.10, 121: 0.15}    # Dreepy / Drakloak (Recon Directive) / Dragapult
LINE_CAP = 0.45
# DISRUPTION, measured off the opponent's own board. What Crushing Hammer, Boss's Orders,
# Unfair Stamp and the Budew lock all buy is the same thing: the opponent does not get to
# attack this turn, or has nothing to do it with. The first draft counted only their total
# attached energy, which is too crude in both directions -- it pays us for an opponent who
# simply has not drawn energy, and it misses the case where they hold six energy on bodies
# that cannot use them.
W_NO_ATTACK = 0.12      # their Active cannot pay ANY of its own attacks
W_ENERGY = 0.08         # sustained energy denial, saturating at 3
W_HAND = 0.06           # resource denial: Unfair Stamp leaves them at 2
W_COND = 0.06           # asleep / paralysed: they lose the attack outright
DENY_FROM_TURN = 3      # before this, "they cannot attack" is just the start of the game
DENY_ENERGY_AT, DENY_HAND_AT = 3, 4
W_LOCK, LOCK_UNTIL_TURN = 0.10, 6


def _slots(ps):
    return list(ps.get("active") or []) + list(ps.get("bench") or [])


def _energy_types(pk):
    return list(pk.get("energies") or [])


def _can_pay_rp(pk):
    """Can this body pay Phantom Dive's {R}{P}? Types are the engine's EnergyType ints;
    2 = {R} and 5 = {P} in this pool (checked against the deck's own attack costs)."""
    e = list(_energy_types(pk))
    for want in (2, 5):
        if want in e:
            e.remove(want)
        elif 0 in e:                    # a colourless-flexible source, if one is ever attached
            e.remove(0)
        else:
            return False
    return True


def _pay(cost, have):
    """Can `have` (attached energy types) pay `cost`? Specific requirements first, then the
    colourless remainder from anything left. Written here rather than imported from engine_v2:
    engine_v2 is barred from the learning path, and this is a rule of the game, not a policy.
    The card DATABASE is a different thing and is still read for the costs themselves."""
    pool = list(have)
    for c in cost:
        if c == 0:
            continue
        if c in pool:
            pool.remove(c)
        else:
            return False
    return len(pool) >= sum(1 for c in cost if c == 0)


def _can_attack(pk):
    """Does this body have the energy for at least one of its own attacks?"""
    from agents._engine import _CARDS, _ATTACKS
    c = _CARDS.get(pk.get("id"))
    if c is None:
        return False
    have = list(pk.get("energies") or [])
    for aid in (c.attacks or []):
        a = _ATTACKS.get(aid)
        if a is not None and _pay(list(a.energies or []), have):
            return True
    return False


def phi(obs, mi=None, detail=False, lock=False):
    """Potential from player `mi`'s perspective (default: the observing player).

    `lock` is the item-lock flag; it cannot be read from the observation and must be supplied
    by whoever knows it (the rollout, which chose the Itchy Pollen).
    """
    cur = obs.get("current") or obs
    pl = cur.get("players") or []
    if len(pl) < 2:
        return ({}, 0.0) if detail else 0.0
    mi = cur.get("yourIndex", 0) if mi is None else mi
    me, opp = pl[mi] or {}, pl[1 - mi] or {}
    t = {}

    # --- prizes: both sides, and the spine of everything else -------------------------------
    my_left = len(me.get("prize") or [])
    op_left = len(opp.get("prize") or [])
    t["prize"] = W_PRIZE * (op_left - my_left)

    # --- banked bench damage: distance to a Phantom Dive KO, not raw damage ------------------
    # Raw damage rewards spraying counters where they never convert. Phantom Dive's six
    # counters exist to bring a body INTO RANGE, so the potential saturates at the threshold.
    ready = prog = 0.0
    scored = 0
    for pk in _slots(opp):
        if not isinstance(pk, dict) or not pk.get("id"):
            continue
        hp, mx = pk.get("hp") or 0, pk.get("maxHp") or 0
        if mx <= 0 or scored >= READY_CAP:
            continue
        if hp <= PHANTOM_DIVE_DMG:
            ready += 1
        else:
            need = mx - PHANTOM_DIVE_DMG          # damage still required, > 0 here
            dealt = mx - hp
            prog += min(1.0, max(0.0, dealt / float(need)))
        scored += 1
    t["spread"] = W_READY * ready + W_PROG * prog

    # --- attack readiness ---------------------------------------------------------------
    # Only bodies IN THE LINE count: Drakloak's Dragon Headbutt costs the same {R}{P} as
    # Phantom Dive, so energy on a Drakloak works now AND carries through the evolution --
    # which is why the plan attaches before evolving. {R}{P} sitting on a Duskull can never
    # be spent on either attack, and the first draft scored it as readiness.
    payers = halves = 0
    munki_d = False
    for pk in _slots(me):
        if not isinstance(pk, dict) or not pk.get("id"):
            continue
        e = _energy_types(pk)
        if pk["id"] == MUNKIDORI and 7 in e:            # 7 = {D}
            munki_d = True
        if pk["id"] not in LINE:
            continue
        if _can_pay_rp(pk):
            payers += 1
        elif 2 in e or 5 in e or 0 in e:
            halves += 1
    t["energy"] = (W_PAYER1 * min(payers, 1)
                   + W_PAYER2 * min(max(payers - 1, 0), 1)
                   + W_HALF * min(halves, HALF_CAP)
                   + (W_MUNKI_D if munki_d else 0.0))

    # --- the line, valued by stage so every step toward Dragapult has a gradient -------------
    ids = [pk.get("id") for pk in _slots(me) if isinstance(pk, dict)]
    t["line"] = min(LINE_CAP, sum(W_STAGE.get(i, 0.0) for i in ids))

    # --- disruption: what Crushing Hammer and the Budew lock actually buy --------------------
    # The opponent's attached energy IS visible (their slots carry `energies`), so denial is
    # measurable. The item lock is NOT: the observation has no such field -- verified against a
    # real board, whose player keys are active/asleep/bench/benchMax/burned/confused/deckCount/
    # discard/hand/handCount/paralyzed/poisoned/prize. The first draft read `cantPlayItem` and
    # therefore scored 0 forever. The rollout knows when WE used Itchy Pollen, so it is passed
    # in rather than guessed at.
    turn0 = cur.get("turn") or 0
    if turn0 >= DENY_FROM_TURN:
        opp_energy = sum(len(_energy_types(pk)) for pk in _slots(opp) if isinstance(pk, dict))
        oa = (opp.get("active") or [None])[0]
        no_attack = 1.0 if (isinstance(oa, dict) and oa.get("id")
                            and not _can_attack(oa)) else 0.0
        stopped = 1.0 if (opp.get("asleep") or opp.get("paralyzed")) else 0.0
        hand = opp.get("handCount")
        hand = DENY_HAND_AT if hand is None else hand
        t["deny"] = (W_NO_ATTACK * no_attack
                     + W_ENERGY * min(1.0, max(0, DENY_ENERGY_AT - opp_energy) / DENY_ENERGY_AT)
                     + W_HAND * min(1.0, max(0, DENY_HAND_AT - hand) / DENY_HAND_AT)
                     + W_COND * stopped)
    else:
        t["deny"] = 0.0

    turn = cur.get("turn") or 0
    t["lock"] = W_LOCK if (lock and turn <= LOCK_UNTIL_TURN) else 0.0

    total = sum(t.values())
    return (t, total) if detail else total


def _check_traces(paths, so, deck_a="dragapult_dusknoir"):
    """Does Phi separate winners? Replay recorded games and bucket decisions by Phi.

    Replay uses the ENGINE (the game itself), never engine_v2 the pilot -- the picks are the
    ones the policy already made, fed back in. If the win rate does not rise with Phi, the
    shaping is decoration and the design says stop here rather than train on it.
    """
    import library
    from mirror_env import MirrorEngine, engine_fingerprint

    eng = MirrorEngine(so)
    fp = engine_fingerprint(eng, [int(x) for x in open(library.deck_path(
        sorted(library.list_decks())[0])) if x.strip()])
    ids = {}

    def deck_ids(n):
        if n not in ids:
            ids[n] = [int(x) for x in open(library.deck_path(n)) if x.strip()]
        return ids[n]

    buckets = collections.defaultdict(lambda: [0, 0])      # bucket -> [wins, n]
    strata = collections.defaultdict(lambda: [0, 0])       # (prize_lead, rest) -> [wins, n]
    games = kept = 0
    for path in paths:
        for line in gzip.open(path, "rt"):
            d = json.loads(line)
            if d.get("header"):
                if d.get("fp") and d["fp"] != fp:
                    sys.exit("trace fingerprint %s != local %s" % (d["fp"], fp))
                continue
            games += 1
            d0 = d.get("deck0") or d.get("deck")
            d1 = d.get("deck1") or d.get("deck")
            if deck_a not in (d0, d1) or d.get("result") not in (0, 1):
                continue
            seat = 0 if d0 == deck_a else 1
            won = 1 if d["result"] == seat else 0
            obs = eng.start(deck_ids(d0), deck_ids(d1), d["seed"], mirror=1)
            kept += 1
            try:
                for pick in d["picks"]:
                    if obs is None:
                        break
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1 or obs.get("select") is None:
                        break
                    if cur.get("yourIndex") == seat:
                        terms, v = phi(obs, seat, detail=True)
                        buckets[round(v * 2) / 2.0][0] += won
                        buckets[round(v * 2) / 2.0][1] += 1
                        # PRIZE-MATCHED. Phi contains the prize lead, and late in a game the
                        # prize lead nearly IS the outcome, so the headline separation is
                        # mostly tautology. The question that decides whether the deck-specific
                        # terms carry information is whether they still separate INSIDE a fixed
                        # prize lead -- the same stratification that turned "winners attack
                        # more" back into "winners are winning".
                        lead = int(round(terms.get("prize", 0.0)))
                        rest = round((v - terms.get("prize", 0.0)) * 2) / 2.0
                        strata[(lead, rest)][0] += won
                        strata[(lead, rest)][1] += 1
                    if pick is None:
                        break
                    obs = eng.select(pick)
            finally:
                eng.finish()
    print("replayed %d of %d traced games | %d decisions"
          % (kept, games, sum(n for _, n in buckets.values())))
    print("%8s %8s %9s" % ("Phi", "n", "win rate"))
    run = [(k, w, n) for k, (w, n) in sorted(buckets.items()) if n >= 30]
    for k, w, n in run:
        print("%8.1f %8d %8.1f%%" % (k, n, 100.0 * w / n))
    if len(run) >= 3:
        lo = sum(w for _, w, _ in run[:len(run) // 3]) / max(1, sum(n for _, _, n in run[:len(run) // 3]))
        hi = sum(w for _, w, _ in run[-len(run) // 3:]) / max(1, sum(n for _, _, n in run[-len(run) // 3:]))
        print("\nbottom third %.1f%% -> top third %.1f%%  (separation %+.1fpt)"
              % (100 * lo, 100 * hi, 100 * (hi - lo)))
        print("VERDICT (unstratified): %s" % ("Phi separates"
                               if hi - lo > 0.15 else
                               "Phi does NOT separate enough -- do not train on it"))
    print("\n=== PRIZE-MATCHED: does the rest of Phi separate at a FIXED prize lead? ===")
    print("%6s %8s %8s %9s" % ("lead", "rest", "n", "win rate"))
    gains = []
    for lead in sorted({k[0] for k in strata}):
        rows = [(r, w, n) for (l, r), (w, n) in sorted(strata.items())
                if l == lead and n >= 200]
        if len(rows) < 3:
            continue
        for r, w, n in rows:
            print("%6d %8.1f %8d %8.1f%%" % (lead, r, n, 100.0 * w / n))
        k = max(1, len(rows) // 3)
        lo = sum(w for _, w, _ in rows[:k]) / max(1, sum(n for _, _, n in rows[:k]))
        hi = sum(w for _, w, _ in rows[-k:]) / max(1, sum(n for _, _, n in rows[-k:]))
        gains.append((lead, hi - lo, sum(n for _, _, n in rows)))
        print("%6d  -> low %.1f%% high %.1f%%  (%+.1fpt within this lead)"
              % (lead, 100 * lo, 100 * hi, 100 * (hi - lo)))
    if gains:
        wsum = sum(g * n for _, g, n in gains) / max(1, sum(n for _, _, n in gains))
        print("\nweighted within-lead separation %+.1fpt" % (100 * wsum))
        print("VERDICT: %s" % ("the deck-specific terms carry information beyond the prize "
                               "count -- the shaping is real"
                               if wsum > 0.05 else
                               "the deck-specific terms add nothing over the prize count; keep "
                               "Phi = prize lead only and drop the rest"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-traces", default="", help="comma-separated trace files")
    ap.add_argument("--mirror-so", default="")
    a = ap.parse_args()

    if a.check_traces:
        from mirror_env import DEFAULT_SO
        _check_traces([p for p in a.check_traces.split(",") if p], a.mirror_so or DEFAULT_SO)
        return
    if a.selftest:
        # Hand-built boards: the ordering is the claim, not the absolute values.
        def board(my_prize, op_prize, opp_hps, my_energy, my_ids, turn=4, lock=False):
            return {"current": {
                "yourIndex": 0, "turn": turn,
                "players": [
                    {"prize": [1] * my_prize, "active": [{"id": my_ids[0], "hp": 300,
                                                          "maxHp": 320, "energies": my_energy[0]}],
                     "bench": [{"id": i, "hp": 90, "maxHp": 90, "energies": e}
                               for i, e in zip(my_ids[1:], my_energy[1:])]},
                    {"prize": [1] * op_prize, "cantPlayItem": lock,
                     "active": [{"id": 999, "hp": opp_hps[0], "maxHp": 320, "energies": []}],
                     "bench": [{"id": 998, "hp": h, "maxHp": 320, "energies": []}
                               for h in opp_hps[1:]]},
                ]}}
        cases = [
            ("even, nothing set up", board(6, 6, [320, 320], [[], []], [DREEPY, DREEPY])),
            # The Crispin split means TWO bodies that can each pay {R}{P}, so that losing the
            # attacker does not stop the attack -- not one energy of each type on two bodies,
            # which leaves NEITHER able to attack. The first draft of this selftest made that
            # mistake and reported the stacked board as better, which is how it was caught.
            ("one charged body", board(6, 6, [320, 320], [[2, 5], []],
                                       [DRAKLOAK, DRAKLOAK])),
            ("FOUR energy stacked on one", board(6, 6, [320, 320], [[2, 5, 2, 5], []],
                                                 [DRAKLOAK, DRAKLOAK])),
            ("two charged bodies (the split)", board(6, 6, [320, 320], [[2, 5], [2, 5]],
                                                     [DRAKLOAK, DRAKLOAK])),
            ("Dragapult up, 1 body in range", board(6, 6, [320, 190], [[2, 5], []],
                                                    [DRAGAPULT_EX, DRAKLOAK])),
            ("two in range", board(6, 6, [190, 190], [[2, 5], []], [DRAGAPULT_EX, DRAKLOAK])),
            ("two prizes ahead", board(4, 6, [320, 320], [[2, 5], []],
                                       [DRAGAPULT_EX, DRAKLOAK])),
        ]
        print("%-32s %8s   %s" % ("board", "Phi", "terms"))
        for name, ob in cases:
            terms, v = phi(ob, 0, detail=True)
            print("%-32s %8.2f   %s" % (name, v,
                  " ".join("%s=%.2f" % (k, x) for k, x in terms.items() if x)))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
