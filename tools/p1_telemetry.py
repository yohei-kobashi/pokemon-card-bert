"""P1 telemetry — GENERIC, deck-agnostic behavior instrumentation for the L2
Discovery Pipeline (docs/l2_pipeline.md). Runs the current engine_v2 policy for a
deck vs a diverse panel (process-isolated, low noise) and emits the symptom
metrics the pipeline keys on, plus a P2-style cross-check against
docs/p0_<deck>.json hypotheses when present.

Usage:  PYTHONPATH=cg-lib python tools/p1_telemetry.py <deck> [--games 16]

The metrics are computed for ANY deck (no per-deck code):
  - dead_card_audit: plays/game per distinct card  (find cards the engine never uses)
  - loss_split: deckout / board(no-Pokemon) / prize
  - energy_attach: top target Pokémon + overload share (energy past the main cost)
  - post_ko_attack_rate: did we attack the turn after our attacker was KO'd (chain)
  - self_removal_fires/game: abilities that shuffle/put their OWNER away (suicide class)
  - first_attack_turn, nonattacking_ends/game, gust koable share
Then any p0 hypothesis whose `metric` maps to one of these is scored PASS/FLAG.
"""
import os, sys, json, argparse, statistics, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import library
from multiprocessing import Pool
from collections import Counter, defaultdict
from cg.api import to_observation_class, SelectContext, OptionType, AreaType, CardType
from cg.game import battle_start, battle_select, battle_finish
from agents._engine import _CARDS, _ATTACKS

TUN = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
PANEL = ["zangoose", "mega_venusaur", "dragapult", "crustle_stall", "archaludon",
         "hop_zacian", "mega_gardevoir", "slowking"]


def _cheapest_cost(cid):
    c = _CARDS.get(cid)
    if not c:
        return 0
    costs = [len(_ATTACKS[a].energies or []) for a in c.attacks
             if _ATTACKS.get(a) and _ATTACKS[a].damage]
    return min(costs) if costs else 0


def _self_removes(cid):
    c = _CARDS.get(cid)
    for s in (c.skills if c else []) or []:
        t = (s.text or "").lower()
        if (("shuffle this pok" in t or "put this pok" in t)
                and ("into your deck" in t or "into your hand" in t)):
            return True
    return False


def _inv_ok(inv, st, me):
    """Evaluate one P0-line invariant (pipeline v2, P1' conformance)."""
    t = inv.get("type")
    if t == "stadium_in_play":
        return bool(st.stadium) and any(x.id == inv["cid"] for x in st.stadium)
    if t == "armed_bodies":
        bodies = ([me.active[0]] if me.active else []) + [p for p in (me.bench or []) if p]
        n = sum(1 for p in bodies
                if p.id == inv["cid"]
                and sum(1 for e in (p.energies or []) if e == inv["energy"]) >= inv["min_energy"])
        return n >= inv["count"]
    return None                      # attack_rate_after is computed post-game


def _play_game(pol, deck, opp_act, opp_deck, me0, invariants=()):
    M = Counter(); L = defaultdict(list)
    my = 0 if me0 else 1
    obs, sd = battle_start(deck if me0 else opp_deck, opp_deck if me0 else deck)
    prev_opp_prizes = 6
    pending_ko = None
    first_atk = None
    own_turns = set(); atk_turns = set(); inv_first = {}
    # decision-point histogram (pipeline v2.1): per own turn — did an attack
    # OPTION exist, and did we take one? Split by the active body. "wall" turns
    # (no option: can't pay/locked) vs "declined" turns (option present, not
    # taken) point at DIFFERENT bugs: arming/pivot vs attack-choice gates.
    turn_info = {}
    try:
        for _ in range(4000):
            cur = obs.get("current")
            if cur is None:
                return M, L
            if cur.get("result", -1) != -1:
                res = cur["result"]
                reason = next((lg.get("reason") for lg in (obs.get("logs") or [])
                               if lg.get("type") == 23), None)
                me = cur["players"][my]
                M["games"] += 1
                if res == my:
                    M["wins"] += 1
                elif res == (1 - my):
                    M["losses"] += 1
                    if reason == 2 or me.get("deckCount", 9) == 0:
                        M["loss_deckout"] += 1
                    elif reason == 3:
                        M["loss_board"] += 1
                    else:
                        M["loss_prize"] += 1
                if first_atk:
                    L["first_attack_turn"].append(first_atk)
                # P1' line conformance + loss post-mortem records
                for inv in invariants:
                    if inv.get("type") == "attack_rate_after":
                        t0 = inv["turn"]
                        own = [t for t in own_turns if t >= t0]
                        if own:
                            L[f"invrate_{inv['id']}"].append(
                                len([t for t in atk_turns if t >= t0]) / len(own))
                    else:
                        L[f"inv_{inv['id']}"].append(inv_first.get(inv["id"], 999))
                if res == (1 - my):
                    L["loss_pm"].append({"reason": reason, "turns": cur.get("turn") or max(own_turns or [0]),
                                         "atk_turns": len(atk_turns), "own_turns": len(own_turns)})
                for t, ti in turn_info.items():
                    M["dh_turns"] += 1
                    if not ti["opt"]:
                        M["dh_wall"] += 1
                        M[f"dhw_{ti['active']}"] += 1
                    elif not ti["atk"]:
                        M["dh_declined"] += 1
                        M[f"dhd_{ti['active']}"] += 1
                return M, L
            yi = cur["yourIndex"]
            if yi != my:
                obs = battle_select(opp_act(obs))
                continue
            choice = pol.act(obs)
            ob = to_observation_class(obs)
            sel = ob.select; st = ob.current
            me = st.players[my]; op = st.players[1 - my]
            if sel is not None and sel.context == SelectContext.MAIN:
                own_turns.add(st.turn)
                for inv in invariants:
                    if inv["id"] not in inv_first and _inv_ok(inv, st, me):
                        inv_first[inv["id"]] = st.turn
                ti = turn_info.setdefault(st.turn, {
                    "active": me.active[0].id if me.active else None,
                    "opt": False, "atk": False})
                if any(o.type == OptionType.ATTACK for o in sel.option):
                    ti["opt"] = True
                if choice and sel.option[choice[0]].type == OptionType.ATTACK:
                    ti["atk"] = True
            opp_prizes = len(op.prize or [])
            if opp_prizes < prev_opp_prizes:
                pending_ko = st.turn
            prev_opp_prizes = opp_prizes
            if sel is not None and sel.context == SelectContext.MAIN and choice:
                o = sel.option[choice[0]]
                have_attack = any(x.type == OptionType.ATTACK for x in sel.option)
                if o.type == OptionType.PLAY:
                    h = me.hand or []
                    if o.index is not None and o.index < len(h):
                        M[f"play_{h[o.index].id}"] += 1
                elif o.type == OptionType.ABILITY:
                    pk = None
                    try:
                        pk = me.active[0] if o.area == AreaType.ACTIVE else me.bench[o.index]
                    except Exception:
                        pass
                    if pk is not None and _self_removes(pk.id):
                        M["self_removal_fires"] += 1
                        bodies = len([x for x in ([me.active[0]] if me.active else []) + list(me.bench) if x])
                        if bodies <= 1:
                            M["self_removal_suicide"] += 1
                elif o.type == OptionType.ATTACH:
                    h = me.hand or []
                    cid = h[o.index].id if (o.index is not None and o.index < len(h)) else None
                    c = _CARDS.get(cid)
                    if c and c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                        pk = None
                        try:
                            pk = me.active[0] if o.inPlayArea == AreaType.ACTIVE else me.bench[o.inPlayIndex or 0]
                        except Exception:
                            pass
                        if pk is not None:
                            M["e_attach"] += 1
                            M[f"eatt_{pk.id}"] += 1
                            need = _cheapest_cost(pk.id)
                            if need and len(pk.energies) >= need:
                                M["e_overload"] += 1
                elif o.type == OptionType.ATTACK:
                    M["attacks"] += 1
                    atk_turns.add(st.turn)
                    if first_atk is None:
                        first_atk = st.turn
                    if pending_ko is not None:
                        M["post_ko_attacked"] += 1; pending_ko = None
                elif o.type == OptionType.END and have_attack:
                    M["nonattacking_ends"] += 1
            if (sel is not None and sel.context == SelectContext.SWITCH and choice
                    and any(x.playerIndex is not None and x.playerIndex != my for x in sel.option)):
                o = sel.option[choice[0]]
                try:
                    tgt = op.active[0] if o.area == AreaType.ACTIVE else op.bench[o.index]
                    # active damage proxy = best offered attack this game state is unknown here;
                    # use the active's best listed attack damage as a rough KO check.
                    ad = 0
                    if me.active:
                        for a in (_CARDS.get(me.active[0].id).attacks if _CARDS.get(me.active[0].id) else []):
                            aa = _ATTACKS.get(a)
                            if aa and aa.damage:
                                ad = max(ad, aa.damage)
                    M["gusts"] += 1
                    if ad and tgt.hp <= ad:
                        M["gust_koable"] += 1
                except Exception:
                    pass
            if pending_ko is not None and st.turn > pending_ko + 2:
                M["post_ko_failed"] += 1; pending_ko = None
            obs = battle_select(choice)
        return M, L
    finally:
        battle_finish()


def _load_invariants(deck_name):
    p0p = os.path.join(ROOT, "docs", f"p0_{deck_name}.json")
    if not os.path.exists(p0p):
        return ()
    p0 = json.load(open(p0p))
    return tuple(i for ln in p0.get("lines", []) for i in ln.get("invariants", []))


class _Wrap:
    """Wrap a bare act(obs) function (legacy agent) as a policy object."""
    def __init__(self, act):
        self.act = act


def _task(args):
    deck_name, opp, games, pilot = args
    from agents import engine_v2
    deck = library.read_deck(deck_name)
    if pilot == "legacy":
        # ORACLE CONFORMANCE BENCHMARKING (pipeline v2.1, P2'): measure the
        # reference pilot on the SAME invariants/metrics. The differentiator is
        # whichever invariant the reference satisfies and the v2 pilot doesn't.
        from battle_log import load_agent
        pol = _Wrap(load_agent(deck_name))
    else:
        pol = engine_v2.make_policy(deck, TUN.get(deck_name, {}))
    od = library.read_deck(opp)
    opol = engine_v2.make_policy(od, TUN.get(opp, {}))
    inv = _load_invariants(deck_name)
    M = Counter(); L = defaultdict(list)
    for g in range(games):
        m, l = _play_game(pol, deck, opol.act, od, g % 2 == 0, inv)
        M.update(m)
        for k, v in l.items():
            L[k] += v
    # tier-1 tripwire: unhandled sub-select contexts that resolved as option[0]
    # (counted per policy instance inside engine_v2.choose_sub)
    for c, n in getattr(pol, "_fallback_hits", {}).items():
        M[f"fallback_ctx_{c}"] += n
    return M, dict(L)


def run(deck_name, games_per_opp=16, panel=None, pilot="v2"):
    panel = [d for d in (panel or PANEL) if d != deck_name]
    with Pool(min(len(panel), 20), maxtasksperchild=1) as pool:
        rows = pool.map(_task, [(deck_name, o, games_per_opp, pilot) for o in panel])
    M = Counter(); L = defaultdict(list)
    for m, l in rows:
        M.update(m)
        for k, v in l.items():
            L[k] += v
    g = max(1, M["games"])
    cnt = Counter(library.read_deck(deck_name))
    dead = {}
    for cid, copies in cnt.items():
        dead[cid] = round(M[f"play_{cid}"] / g, 3)
    eatt_total = max(1, M["e_attach"])
    eatt_top = sorted(((cid, M[f"eatt_{cid}"]) for cid in {int(k[5:]) for k in M if k.startswith("eatt_")}),
                      key=lambda x: -x[1])[:4]
    rep = {
        "deck": deck_name, "games": M["games"], "win_rate": round(M["wins"]/g, 3),
        "loss_split": {"deckout": M["loss_deckout"], "board": M["loss_board"], "prize": M["loss_prize"]},
        "self_removal": {"fires": M["self_removal_fires"], "suicides": M["self_removal_suicide"]},
        "energy": {"overload_share": round(M["e_overload"]/eatt_total, 3),
                   "top_targets": [{"cid": c, "name": _CARDS[c].name, "share": round(n/eatt_total, 3)} for c, n in eatt_top]},
        "post_ko_attack_rate": round(M["post_ko_attacked"]/max(1, M["post_ko_attacked"]+M["post_ko_failed"]), 3),
        "first_attack_turn_median": statistics.median(L["first_attack_turn"]) if L.get("first_attack_turn") else None,
        "nonattacking_ends_per_game": round(M["nonattacking_ends"]/g, 2),
        "gust_koable_share": round(M["gust_koable"]/max(1, M["gusts"]), 3) if M["gusts"] else None,
        "dead_card_audit": {str(c): dead[c] for c in sorted(dead, key=lambda x: dead[x])},
        "subselect_fallback_per_game": {
            SelectContext(int(k[13:])).name: round(M[k] / g, 2)
            for k in sorted(M) if k.startswith("fallback_ctx_")},
    }
    # ---- P1' decision-point histogram (pipeline v2.1) ----------------------
    dht = max(1, M["dh_turns"])

    def _top(prefix):
        rows = sorted(((k[len(prefix):], v) for k, v in M.items()
                       if k.startswith(prefix)), key=lambda x: -x[1])[:4]
        out = {}
        for cid, n in rows:
            c = _CARDS.get(int(cid)) if cid not in ("None",) else None
            out[c.name if c else cid] = round(n / dht, 3)
        return out
    rep["decision_histogram"] = {
        "wall_share": round(M["dh_wall"] / dht, 3),        # no attack option at all
        "declined_share": round(M["dh_declined"] / dht, 3),  # option present, not taken
        "wall_by_active": _top("dhw_"),
        "declined_by_active": _top("dhd_"),
    }
    # ---- P1' line conformance (pipeline v2) --------------------------------
    conf = {}
    for inv in _load_invariants(deck_name):
        iid = inv["id"]
        if inv.get("type") == "attack_rate_after":
            rates = L.get(f"invrate_{iid}", [])
            if rates:
                conf[iid] = {"mean_rate": round(statistics.mean(rates), 3),
                             "meets_min": round(sum(1 for r in rates if r >= inv["min"]) / len(rates), 3)}
        else:
            firsts = L.get(f"inv_{iid}", [])
            if firsts:
                bt = inv.get("by_turn")
                conf[iid] = {"by_turn_rate": round(sum(1 for t in firsts if t <= bt) / len(firsts), 3) if bt else None,
                             "ever_rate": round(sum(1 for t in firsts if t < 999) / len(firsts), 3),
                             "median_turn": statistics.median([t for t in firsts if t < 999]) if any(t < 999 for t in firsts) else None}
    rep["line_conformance"] = conf
    pm = L.get("loss_pm", [])
    if pm:
        rep["loss_postmortem"] = {
            "losses": len(pm),
            "mean_turns": round(statistics.mean(x["turns"] for x in pm), 1),
            "mean_attack_share": round(statistics.mean(
                (x["atk_turns"] / x["own_turns"]) for x in pm if x["own_turns"]), 3),
            "by_reason": dict(Counter(x["reason"] for x in pm)),
        }
    # ---- P2 cross-check vs p0 JSON --------------------------------------------
    p0p = os.path.join(ROOT, "docs", f"p0_{deck_name}.json")
    flags = []
    if os.path.exists(p0p):
        p0 = json.load(open(p0p))
        for h in p0.get("hypotheses", []):
            m = (h.get("metric") or "")
            val = None
            mm = re.match(r"play_rate\[(\d+)\]", m)
            if mm:
                val = dead.get(int(mm.group(1)))
            elif "loss_share[deckout]" in m or "loss_deckout" in m:
                val = round(M["loss_deckout"]/max(1, M["losses"]), 3)
            elif "post_ko" in m:
                val = rep["post_ko_attack_rate"]
            elif "nonattacking" in m:
                val = rep["nonattacking_ends_per_game"]
            elif "overload" in m:
                val = rep["energy"]["overload_share"]
            flags.append({"id": h.get("id"), "severity": h.get("severity"),
                          "metric": m, "value": val, "direction": h.get("direction"),
                          "auto": val is not None})
    rep["p0_crosscheck"] = flags
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--pilot", choices=["v2", "legacy"], default="v2",
                    help="legacy = oracle conformance benchmarking (P2' v2.1)")
    args = ap.parse_args()
    print(json.dumps(run(args.deck, args.games, pilot=args.pilot), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
