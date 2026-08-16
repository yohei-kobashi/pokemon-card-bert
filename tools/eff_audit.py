#!/usr/bin/env python3
"""Play-efficiency audit for the SHIPPING pilot, across arbitrary opponents.

The published guide describes this deck as able to fight anyone -- attack with almost any
Pokemon, win faster with a Cursed Blast or two. The gate says 44%. This measures the gap
mechanically, per opponent, from fresh games of a given --spec (the LM wrapper or engine):

    turns where we did NOT attack        <- the guide's core implicit standard
    turns where we did NOT attach        <- one free energy per turn, use it or lose it
    prize tempo (taken vs given, per our-turn)
    dives / blasts / Adrena / Boss actually used
    line assembly clock, fuel ledger, deaths by body (evolution-corrected)

    PYTHONPATH=.:cg-lib:tools python3 tools/eff_audit.py --opp alakazam_nz --games 100 \
        --spec 'planfilter:...:hf:/root/out/fld_r39a@dusk' --json out.json
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
DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
PHANTOM_DIVE, JET = 154, 153
BOSS, CRISPIN, LILLIE = 1182, 1198, 1227
NAME = {119: "Dreepy", 120: "Drakloak", 121: "Dragapult", 131: "Duskull", 132: "Dusclops",
        133: "Dusknoir", 235: "Budew", 112: "Munkidori", 140: "Fez", 1071: "Meowth"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--opp", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fmt", default="dusk", choices=("prompt", "dusk"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import mirror_match as mm
    from agents._engine import _CARDS
    from cg.api import CardType
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

    eng = MirrorEngine(DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    my_ids, opp_ids = mm.load_deck(a.deck), mm.load_deck(a.opp)
    mm._FMT = a.fmt
    agent, _ = mm.make_agent(a.spec, a.deck, my_ids, tuning.get(a.deck, {}))
    opp_agent, _ = mm.make_agent("engine", a.opp, opp_ids, tuning.get(a.opp, {}))

    C = collections.Counter()
    deaths = collections.Counter()
    firsts = collections.defaultdict(list)
    g = {}

    def reset():
        g.update(turn=None, ourturn=0, prev=None, my_pz_prev=None, op_pz_prev=None,
                 saw_pult=None, saw_pay=None, saw_dive=None,
                 t_attacked=False, t_attached=False, rp_prev=0,
                 t_atk_offered=False, t_att_offered=False)

    def _slots(ps):
        return [p for p in (ps.get("active") or []) + (ps.get("bench") or [])
                if isinstance(p, dict)]

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

    def close_turn():
        C["turns"] += 1
        if not g["t_attacked"]:
            C["no_attack_turns"] += 1
            if g["t_atk_offered"]:
                C["declined_attack_turns"] += 1    # an attack was ON the menu, we passed
        if not g["t_attached"]:
            C["no_attach_turns"] += 1
            if g["t_att_offered"]:
                C["declined_attach_turns"] += 1
        g["t_attacked"] = g["t_attached"] = False
        g["t_atk_offered"] = g["t_att_offered"] = False

    def watched(obs):
        cur = obs.get("current") or {}
        sel = obs.get("select") or {}
        yi = cur.get("yourIndex", 0)
        pl = cur.get("players") or []
        me = (pl[yi] or {}) if yi < len(pl) else {}
        opp = (pl[1 - yi] or {}) if yi < len(pl) else {}
        turn = cur.get("turn")
        if turn != g["turn"]:
            if g["turn"] is not None:
                g["ourturn"] += 1
                close_turn()
            now = _slots(me)
            # prize tempo: my prize count falling = prizes WE took
            mp, op = len(me.get("prize") or []), len(opp.get("prize") or [])
            if g["my_pz_prev"] is not None and mp < g["my_pz_prev"]:
                C["pz_taken"] += g["my_pz_prev"] - mp
            if g["op_pz_prev"] is not None and op < g["op_pz_prev"]:
                C["pz_given"] += g["op_pz_prev"] - op
            g["my_pz_prev"], g["op_pz_prev"] = mp, op
            # deaths (evolution-corrected)
            if g["prev"] is not None:
                remaining = [(p.get("id"), _rp(p)) for p in now]
                for p in g["prev"]:
                    key = (p.get("id"), _rp(p))
                    if key in remaining:
                        remaining.remove(key)
                    elif p.get("id") in (DREEPY, DRAKLOAK, DUSKULL, DUSCLOPS) \
                            and (p.get("id") + 1, _rp(p)) in remaining:
                        remaining.remove((p.get("id") + 1, _rp(p)))
                    else:
                        deaths[NAME.get(p.get("id"), p.get("id"))] += 1
            g["prev"] = now
            g["turn"] = turn
            if g["saw_pult"] is None and any(p.get("id") == PULT for p in now):
                g["saw_pult"] = g["ourturn"]
            if g["saw_pay"] is None and any(
                    _can_pd(p) for p in now if p.get("id") in (DREEPY, DRAKLOAK, PULT)):
                g["saw_pay"] = g["ourturn"]
            rp_now = sum(_rp(p) for p in now if p.get("id") in (DREEPY, DRAKLOAK, PULT))
            d = rp_now - g["rp_prev"]
            C["rp_gained" if d > 0 else "rp_lost"] += abs(d) if d else 0
            g["rp_prev"] = rp_now

        pick = agent(obs)
        opts = sel.get("option") or []
        for o in opts:
            if isinstance(o, dict):
                if o.get("attackId"):
                    g["t_atk_offered"] = True
                if o.get("type") == 8:
                    g["t_att_offered"] = True
        for i in (pick or []):
            if not (0 <= i < len(opts) and isinstance(opts[i], dict)):
                continue
            o = opts[i]
            if o.get("attackId"):
                g["t_attacked"] = True
                C["atk_%s" % ("dive" if o["attackId"] == PHANTOM_DIVE
                              else "jet" if o["attackId"] == JET else "other")] += 1
                if o["attackId"] == PHANTOM_DIVE and g["saw_dive"] is None:
                    g["saw_dive"] = g["ourturn"]
            if o.get("type") == 8:
                g["t_attached"] = True
            if o.get("type") == 10:
                area = o.get("inPlayArea", o.get("area"))
                idx = o.get("inPlayIndex", o.get("index"))
                try:
                    pk = ((me.get("active") or [None])[0] if area in (1, 4)
                          else (me.get("bench") or [])[idx])
                except (IndexError, TypeError):
                    pk = None
                if isinstance(pk, dict) and pk.get("id") in (DUSCLOPS, DUSKNOIR, 112):
                    C["blast_or_adrena"] += 1
            # MAIN-menu plays are type 7 (t7+i<hand index> in the traces); the first
            # version only counted type-3 sub-select picks and undercounted supporters.
            if o.get("type") in (3, 7) and o.get("index") is not None \
                    and o.get("area") in (None, 2):
                h = me.get("hand") or []
                if o["index"] < len(h) and isinstance(h[o["index"]], dict):
                    cid = h[o["index"]].get("id")
                    if cid == BOSS:
                        C["boss_played"] += 1
                    elif cid == CRISPIN:
                        C["crispin_played"] += 1
                    elif cid == LILLIE:
                        C["lillie_played"] += 1
                    c = _CARDS.get(cid)
                    if c is not None and c.cardType == CardType.SUPPORTER:
                        C["supporters"] += 1
        return pick

    wins = 0
    for i in range(a.games):
        reset()
        watched(dict(_ES))
        opp_agent(dict(_ES))
        seed, mine = a.seed + i // 2, i % 2
        r = (play(eng, watched, opp_agent, my_ids, opp_ids, seed, mirror=1) if mine == 0
             else play(eng, opp_agent, watched, opp_ids, my_ids, seed, mirror=1))
        close_turn()
        wins += int(r == mine)
        for k in ("saw_pult", "saw_pay", "saw_dive"):
            if g[k] is not None:
                firsts[k].append(g[k])

    n = float(a.games)
    t = float(max(1, C["turns"]))
    out = {
        "opp": a.opp, "games": a.games, "wins": wins,
        "win_pct": round(100.0 * wins / n, 1),
        "turns_pg": round(C["turns"] / n, 2),
        "no_attack_turn_pct": round(100.0 * C["no_attack_turns"] / t, 1),
        "declined_attack_pct": round(100.0 * C["declined_attack_turns"] / t, 1),
        "no_attach_turn_pct": round(100.0 * C["no_attach_turns"] / t, 1),
        "declined_attach_pct": round(100.0 * C["declined_attach_turns"] / t, 1),
        "lillie_pg": round(C["lillie_played"] / n, 2),
        "dives_pg": round(C["atk_dive"] / n, 2),
        "jets_pg": round(C["atk_jet"] / n, 2),
        "other_atks_pg": round(C["atk_other"] / n, 2),
        "pz_taken_pg": round(C["pz_taken"] / n, 2),
        "pz_given_pg": round(C["pz_given"] / n, 2),
        "supporters_pg": round(C["supporters"] / n, 2),
        "boss_pg": round(C["boss_played"] / n, 2),
        "crispin_pg": round(C["crispin_played"] / n, 2),
        "blast_adrena_pg": round(C["blast_or_adrena"] / n, 2),
        "rp_gained_pg": round(C["rp_gained"] / n, 2),
        "rp_lost_pg": round(C["rp_lost"] / n, 2),
        "pult_games_pct": round(100.0 * len(firsts["saw_pult"]) / n, 1),
        "pult_turn": round(sum(firsts["saw_pult"]) / max(1, len(firsts["saw_pult"])), 1),
        "dive_games_pct": round(100.0 * len(firsts["saw_dive"]) / n, 1),
        "deaths": dict(sorted(deaths.items(), key=lambda kv: -kv[1])[:8]),
    }
    print(json.dumps(out))
    if a.json:
        json.dump(out, open(a.json, "w"))


if __name__ == "__main__":
    main()
