#!/usr/bin/env python3
"""Paired A/B + mechanism counters for the ogerpon_mono specialised engine (`ogre`).

One process = one arm; pairing comes from running both arms with the same --seed range
(the established two-process pattern -- DUSK_RULES is read at policy construction).

    PYTHONPATH=.:cg-lib:tools python3 tools/ogre_ab.py --games 100 --seed 1000       # base
    DUSK_RULES=front,charge,search,bench,spread,boss,energy_cap=2,ogre \
    PYTHONPATH=.:cg-lib:tools python3 tools/ogre_ab.py --games 100 --seed 1000 --l2 dusknoir

Counters are the ones the audits said matter: does Dragapult exist, does it dive, do the
dives convert, and WHAT are we feeding the farm.
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ES = {"current": None, "logs": [], "remainingOverageTime": 600.0,
       "search_begin_input": None, "select": None, "step": 1}
DREEPY, DRAKLOAK, PULT = 119, 120, 121
PHANTOM_DIVE = 154
NAME = {119: "Dreepy", 120: "Drakloak", 121: "Dragapult", 131: "Duskull", 132: "Dusclops",
        133: "Dusknoir", 235: "Budew", 112: "Munkidori", 140: "Fez", 1071: "Meowth"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--opp", default="ogerpon_mono")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--l2", default=None, help="override profile l2 (e.g. dusknoir)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--search-playouts", type=int, default=0,
                    help="LOOKAHEAD PILOT: at every exactly/up-to-one menu of ours with "
                         "2..10 options, roll each candidate to the end N times with the "
                         "engine on both sides (the opponent's deck AND play are known) "
                         "and take the argmax. 0 = the plain 1-ply rule pilot.")
    a = ap.parse_args()

    import copy

    import mirror_match as mm
    from agents import engine_v2
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

    eng = MirrorEngine(DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    my_ids, opp_ids = mm.load_deck(a.deck), mm.load_deck(a.opp)
    prof = copy.deepcopy(tuning.get(a.deck, {}))
    if a.l2:
        prof["l2"] = a.l2
    pol = engine_v2.make_policy(my_ids, prof)
    opp_pol = engine_v2.make_policy(opp_ids, tuning.get(a.opp, {}))

    C = collections.Counter()
    deaths = collections.Counter()
    firsts = collections.defaultdict(list)
    g = {}

    base_act = pol.act
    if a.search_playouts > 0:
        import random as _rnd

        import rl_branch
        _srng = _rnd.Random(a.seed * 7919 + 17)

        _search_ctx = {int(x) for x in
                       (os.environ.get("SEARCH_CTX", "") or "").split(",") if x}

        def search_act(obs):
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            lo = sel.get("minCount", 1) or 0
            hi = sel.get("maxCount", 1) or 1
            if _search_ctx and sel.get("context") not in _search_ctx:
                return base_act(obs)
            if not (2 <= len(opts) <= 10) or not (lo <= 1 <= hi):
                return base_act(obs)
            cur = obs.get("current") or {}
            yi = cur.get("yourIndex", 0)
            sels = [[i] for i in range(len(opts))]
            per = [[] for _ in sels]
            try:
                for _ in range(a.search_playouts):
                    q = rl_branch.branch_values(obs, my_ids, opp_ids, yi, sels,
                                                base_act, opp_pol.act,
                                                n_playouts=1, rng=_srng)
                    for k, v in enumerate(q):
                        if v is not None:
                            per[k].append(v)
            except Exception:                              # noqa: BLE001
                return base_act(obs)                       # mid-resolution etc.: 1-ply
            best, bv = None, None
            for k, vals in enumerate(per):
                if not vals:
                    continue
                m = sum(vals) / len(vals)
                if bv is None or m > bv:
                    best, bv = k, m
            if best is None:
                return base_act(obs)
            C["searched_menus"] += 1
            return [best]
        pol_act = search_act
    else:
        pol_act = base_act

    def reset():
        g.update(turn=None, ourturn=0, prev=None, prev_prizes=None,
                 saw_pult=None, saw_pay=None, saw_dive=None, dives=0,
                 rp_prev=0, rp_gained=0, rp_lost=0)

    def _slots(ps):
        act = [(p, True) for p in (ps.get("active") or []) if isinstance(p, dict)]
        return act + [(p, False) for p in (ps.get("bench") or []) if isinstance(p, dict)]

    def _rp(p):
        return sum(1 for e in (p.get("energies") or []) if e in (2, 5))

    def _can_pd(p):
        e = list(p.get("energies") or [])
        for w in (2, 5):
            if w in e:
                e.remove(w)
            elif 0 in e:
                e.remove(0)
            else:
                return False
        return True

    def watched(obs):
        cur = obs.get("current") or {}
        sel = obs.get("select") or {}
        yi = cur.get("yourIndex", 0)
        pl = cur.get("players") or []
        me = (pl[yi] or {}) if yi < len(pl) else {}
        turn = cur.get("turn")
        if turn != g["turn"]:
            if g["turn"] is not None:
                g["ourturn"] += 1
            now = _slots(me)
            if g["prev"] is not None:
                remaining = [(p.get("id"), _rp(p)) for p, _ in now]
                for p, act in g["prev"]:
                    key = (p.get("id"), _rp(p))
                    if key in remaining:
                        remaining.remove(key)
                    # evolution is not death: Dreepy->Drakloak->Dragapult (119/120/121)
                    # and Duskull->Dusclops->Dusknoir (131/132/133) are +1 chains. The
                    # first version listed the EVOLVED ids here, so every successful
                    # Duskull evolve was booked as a Duskull death -- which is also the
                    # suspicion hanging over the original 1.14/game number.
                    elif p.get("id") in (DREEPY, DRAKLOAK, 131, 132) \
                            and (p.get("id") + 1, _rp(p)) in remaining:
                        remaining.remove((p.get("id") + 1, _rp(p)))
                    else:
                        mine_now = len(me.get("prize") or [])
                        if (g["prev_prizes"] is not None and mine_now < g["prev_prizes"]) \
                                or p.get("id") not in (DREEPY, DRAKLOAK, PULT):
                            deaths[(NAME.get(p.get("id"), p.get("id")), act)] += 1
                            C["fed_prizes"] += 1
            g["prev"] = now
            g["prev_prizes"] = len(me.get("prize") or [])
            g["turn"] = turn
            if g["saw_pult"] is None and any(p.get("id") == PULT for p, _ in now):
                g["saw_pult"] = g["ourturn"]
            if g["saw_pay"] is None and any(
                    _can_pd(p) for p, _ in now if p.get("id") in (DREEPY, DRAKLOAK, PULT)):
                g["saw_pay"] = g["ourturn"]
            # fuel ledger: {R}/{P} ON LINE BODIES, turn-start to turn-start. Gains are
            # attaches (manual + Crispin); losses are hammers, deaths and retreat costs.
            rp_now = sum(_rp(p) for p, _ in now if p.get("id") in (DREEPY, DRAKLOAK, PULT))
            d = rp_now - g["rp_prev"]
            if d > 0:
                g["rp_gained"] += d
            elif d < 0:
                g["rp_lost"] += -d
            g["rp_prev"] = rp_now
        pick = pol_act(obs)
        opts = sel.get("option") or []
        for i in (pick or []):
            if not (0 <= i < len(opts) and isinstance(opts[i], dict)):
                continue
            o = opts[i]
            if o.get("attackId") == PHANTOM_DIVE:
                g["dives"] += 1
                if g["saw_dive"] is None:
                    g["saw_dive"] = g["ourturn"]
            # ability uses by owner (OptionType.ABILITY == 10 in the raw dict)
            if o.get("type") == 10:
                area = o.get("inPlayArea", o.get("area"))
                idx = o.get("inPlayIndex", o.get("index"))
                try:
                    pk = ((me.get("active") or [None])[0] if area in (1, 4)
                          else (me.get("bench") or [])[idx])
                except (IndexError, TypeError):
                    pk = None
                if isinstance(pk, dict):
                    C["abl_%s" % pk.get("id")] += 1
        return pick

    def opp_agent(obs):
        return opp_pol.act(obs)

    wins = 0
    for i in range(a.games):
        reset()
        watched(dict(_ES))
        opp_agent(dict(_ES))
        seed, mine = a.seed + i // 2, i % 2
        r = (play(eng, watched, opp_agent, my_ids, opp_ids, seed, mirror=1) if mine == 0
             else play(eng, opp_agent, watched, opp_ids, my_ids, seed, mirror=1))
        won = int(r == mine)
        wins += won
        # the gambit premise, measured: does the win mass live in the fast-assembly cohort?
        if g["saw_pay"] is not None and g["saw_pay"] <= 4:
            C["n_pay4"] += 1
            C["win_pay4"] += won
            C["pay4_dives"] += g["dives"]
            C["pay4_turns"] += g["ourturn"]
            if not won:
                C["pay4_loss_dives"] += g["dives"]
                C["pay4_loss_n"] += 1
        else:
            C["n_slow"] += 1
            C["win_slow"] += won
        C["dives"] += g["dives"]
        C["rp_gained"] += g["rp_gained"]
        C["rp_lost"] += g["rp_lost"]
        C["turns"] += g["ourturn"]
        for k in ("saw_pult", "saw_pay", "saw_dive"):
            if g[k] is not None:
                firsts[k].append(g[k])

    n = float(a.games)
    out = {
        "arm": "ogre" if os.environ.get("DUSK_RULES") else "base",
        "l2": a.l2, "games": a.games, "seed": a.seed, "wins": wins,
        "win_pct": round(100.0 * wins / n, 1),
        "dives_per_game": round(C["dives"] / n, 2),
        "pult_games": len(firsts["saw_pult"]),
        "pult_turn": round(sum(firsts["saw_pult"]) / max(1, len(firsts["saw_pult"])), 2),
        "pay_games": len(firsts["saw_pay"]),
        "pay_turn": round(sum(firsts["saw_pay"]) / max(1, len(firsts["saw_pay"])), 2),
        "dive_games": len(firsts["saw_dive"]),
        "dive_turn": round(sum(firsts["saw_dive"]) / max(1, len(firsts["saw_dive"])), 2),
        "rp_gained_pg": round(C["rp_gained"] / n, 2),
        "rp_lost_pg": round(C["rp_lost"] / n, 2),
        "turns_pg": round(C["turns"] / n, 2),
        "abl_recon_pg": round(C["abl_120"] / n, 2),
        "abl_munki_pg": round(C["abl_112"] / n, 2),
        "abl_clops_pg": round(C["abl_132"] / n, 2),
        "abl_noir_pg": round(C["abl_133"] / n, 2),
        "n_pay4": C["n_pay4"], "win_pay4": C["win_pay4"],
        "n_slow": C["n_slow"], "win_slow": C["win_slow"],
        "pay4_dives": C["pay4_dives"], "pay4_turns": C["pay4_turns"],
        "pay4_loss_dives": C["pay4_loss_dives"], "pay4_loss_n": C["pay4_loss_n"],
        "deaths": {("%s@%s" % (nm, "act" if act else "bench")): c
                   for (nm, act), c in sorted(deaths.items(), key=lambda kv: -kv[1])},
    }
    print(json.dumps(out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
