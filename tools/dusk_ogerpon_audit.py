#!/usr/bin/env python3
"""Why dragapult_dusknoir wins 3-7% against ogerpon_mono, decision by decision.

    PYTHONPATH=cg-lib:tools python3 tools/dusk_ogerpon_audit.py --games 40 \
        --spec 'planfilter:lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace:hf:/root/out/fld_r2a'

This matchup is not "slightly unfavourable" -- it is 3-7% across nine gate rounds while every
other opponent sits at 19-53%. A gap that size is usually one mechanic, not one hundred small
mistakes, and the decklist points at three candidates before a single game is played:

  * ogerpon_mono's ONLY Pokemon is 4x Teal Mask Ogerpon ex, and it is Tera: "as long as this
    Pokemon is on your Bench, prevent all damage done to this Pokemon by attacks". Phantom
    Dive's six bench counters are half of what our deck does. If the engine applies that
    prevention to them, every one of those counters is thrown away -- and neither dusk_plan.py
    nor dusk_spread.py contains the word "tera".
  * Ogerpon ex has 210 HP. Phantom Dive does 200. Our main attack cannot knock one out, so a
    KO needs an ability packet (Dusknoir 13 counters, Dusclops 5) on top.
  * Myriad Leaf Shower does "30 more damage for each Energy attached to BOTH Active Pokemon".
    Our own attachments arm it. Phantom Dive costs {R}{P}, so simply paying for our attack
    hands them +60, and every extra energy on our Active is +30 more.

So this measures those three directly rather than looking for mistakes in general. Everything
here is counted from OUR decision points, which is all a pilot can see anyway.
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

_EPISODE_START = {"current": None, "logs": [], "remainingOverageTime": 600.0,
                  "search_begin_input": None, "select": None, "step": 1}
OGERPON = 96
PHANTOM_DIVE = 154
HAMMER = 1120                      # Crushing Hammer, ours as well as theirs
JUDGE, STAMP, BOSS, FAN = 1213, 1080, 1182, 1161


def _side(obs, seat):
    st = (obs.get("current") or {})
    ps = st.get("player") or st.get("players") or []
    if isinstance(ps, list) and len(ps) > seat:
        return ps[seat] or {}
    return {}


def _bodies(side):
    a = (side.get("active") or [None])[0]
    b = [x for x in (side.get("bench") or []) if isinstance(x, dict)]
    return ([a] if isinstance(a, dict) else []), b


def _energy(body):
    if not isinstance(body, dict):
        return 0
    for k in ("energy", "attachedEnergy", "energies"):
        v = body.get(k)
        if isinstance(v, list):
            return len(v)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--opp", default="ogerpon_mono",
                    help="comma list; each is audited separately so a rate can be "
                         "read against a control rather than in isolation")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--fmt", default="dusk", choices=("prompt", "dusk"))
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import mirror_match as mm
    from lm.actions import encode_option
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

    eng = MirrorEngine(a.mirror_so or DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    my_ids, opp_ids = mm.load_deck(a.deck), mm.load_deck(a.opp)
    mm._FMT = a.fmt
    agent, _sc = mm.make_agent(a.spec, a.deck, my_ids, tuning.get(a.deck, {}))
    opp_agent, _ = mm.make_agent("engine", a.opp, opp_ids, tuning.get(a.opp, {}))

    S = collections.Counter()
    my_e_at_their_hp = []          # energy on OUR active, sampled at our decisions
    bench_seen = {}                # (game, slot) -> last seen hp, to detect counter landings
    cur = [0]
    spread_targets = collections.Counter()
    declined = collections.Counter()
    # PER-TURN opportunity sets. A card sits in hand across every menu of a turn, so counting
    # menus answers a question nobody asked: phantom_dive read 39% per menu and 94.7% per turn,
    # and the 39% was reported as this deck's largest defect before the error was caught.
    T = collections.defaultdict(set)
    opp_e_by_turn = collections.defaultdict(list)
    opp_hp_by_turn = collections.defaultdict(list)     # maxHp reveals Lively Stadium / Cape
    pz_by_turn = collections.defaultdict(list)
    last_hp = {}
    attacks_used = collections.Counter()

    def watch(obs):
        pick = inner(obs)
        try:
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if len(opts) < 1:
                return pick
            me, opp = _side(obs, 0), _side(obs, 1)
            # seats alternate; identify ours by which side holds Dragapult-family ids
            ma, mb = _bodies(me)
            if not any((x or {}).get("id") in (119, 120, 121) for x in ma + mb):
                me, opp = opp, me
                ma, mb = _bodies(me)
            oa, ob = _bodies(opp)

            S["decisions"] += 1
            if ma:
                my_e_at_their_hp.append(_energy(ma[0]))

            texts = [encode_option(o, obs) for o in opts]
            picked = set(pick if isinstance(pick, (list, tuple)) else [pick])

            # 1b. Cursed Blast is an ABILITY: it costs no energy. Against a deck that strips
            # our energy and forces 70-damage Jet Headbutts into 210 HP bodies, it is the only
            # damage source the opponent cannot deny. `clops_hold` restrains it by design -- the
            # question is whether that restraint is right HERE.
            for _cid, _nm in ((132, "dusclops"), (133, "dusknoir")):
                bl = [i for i, t in enumerate(texts) if t.startswith("ability:c%d" % _cid)]
                if bl:
                    S["blast_%s_offered" % _nm] += 1
                    if picked & set(bl):
                        S["blast_%s_used" % _nm] += 1

            # 1. Crushing Hammer -- the one card that directly cuts Myriad Leaf Shower
            hs = [i for i, t in enumerate(texts) if ("c%d" % HAMMER) in t]
            if hs:
                S["hammer_offered"] += 1
                if picked & set(hs):
                    S["hammer_played"] += 1

            # --- the same questions, per TURN, which is the unit an opportunity comes in ---
            turn = (obs.get("current") or {}).get("turn")
            key = (cur[0], turn)
            opp_energy = sum(_energy(x) for x in oa + ob)
            if oa:
                opp_e_by_turn[turn].append(_energy(oa[0]))
            for cid, nm in ((HAMMER, "hammer"), (JUDGE, "judge"), (STAMP, "stamp"),
                            (BOSS, "boss"), (FAN, "fan")):
                sel_ = [i for i, t in enumerate(texts) if ("c%d" % cid) in t]
                if not sel_:
                    continue
                # A Hammer with nothing to discard is not an opportunity missed.
                if cid == HAMMER and opp_energy == 0:
                    continue
                T[nm + "_turns"].add(key)
                if picked & set(sel_):
                    T[nm + "_played"].add(key)
            # Was a zero-energy body sitting on their bench for Boss's Orders to drag up?
            # Myriad Leaf Shower counts energy on BOTH Actives, so promoting their empty one
            # costs them the whole scaling term for a turn.
            if ob:
                if min(_energy(x) for x in ob) == 0:
                    T["boss_empty_available"].add(key)

            # 2. Phantom Dive, and where its six counters go
            pds = [i for i, o in enumerate(opts)
                   if isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE]
            if pds:
                S["pd_offered"] += 1
                if picked & set(pds):
                    S["pd_used"] += 1
                else:
                    # Declining the deck's main attack is either a real error or a correct
                    # "set up first" -- which of the two shows in WHAT was taken instead.
                    for i in picked:
                        if 0 <= i < len(texts):
                            declined[texts[i].split(":", 1)[0].split("@", 1)[0]] += 1

            # 2b. ANY attack available and declined. The energy story and the piloting story
            # make opposite predictions here: if the deck simply cannot pay, attacks are rarely
            # OFFERED; if the pilot is passing, they are offered and refused.
            atks = [i for i, o in enumerate(opts)
                    if isinstance(o, dict) and o.get("attackId")]
            if atks:
                S["attack_offered"] += 1
                if picked & set(atks):
                    S["attack_taken"] += 1
                    for i in picked:
                        if i in atks:
                            aid = opts[i].get("attackId")
                            attacks_used[aid] += 1
            # context 13/14 are DAMAGE_COUNTER / DAMAGE_COUNTER_ANY -- the key dusk_spread uses.
            # An earlier version looked for sel["type"], which does not exist, and reported zero
            # spread menus in 30 games while the deck was plainly using Phantom Dive.
            if sel.get("context") in (13, 14):
                S["spread_menus"] += 1
                for i in picked:
                    if 0 <= i < len(texts):
                        t = texts[i]
                        spread_targets["BENCH" if "BENCH" in t else "ACTIVE"] += 1
                        if "BENCH" in t and ("c%d" % OGERPON) in t:
                            S["counters_onto_benched_tera"] += 1

            # 3. THE TERA QUESTION, answered by the engine rather than by the rulebook.
            #
            # Per-slot HP tracking cannot answer it here: every Pokemon they own is the same
            # card, so a slot whose HP "fell" may simply be a different Ogerpon after a KO or a
            # new bench drop. Compare the bench TOTAL instead, and only across a pair of
            # observations where the bench COUNT is unchanged -- then a fall in the total is
            # damage that actually landed on a benched Tera body.
            tot = sum((b.get("hp") or 0) for b in ob)
            key = cur[0]
            prev = bench_seen.get(key)
            if prev is not None and prev[0] == len(ob) and len(ob) > 0:
                if tot < prev[1]:
                    S["benched_tera_total_hp_fell"] += 1
                    S["benched_tera_hp_lost"] += prev[1] - tot
            bench_seen[key] = (len(ob), tot)
            # their Active's energy, which is most of their damage
            if oa:
                S["opp_active_energy_sum"] += _energy(oa[0])
                S["opp_active_samples"] += 1
                # maxHp is the printed HP plus whatever is boosting it, so it MEASURES how
                # often Lively Stadium (+30 to Basics) and Hero's Cape (+100) are actually in
                # play instead of assuming the worst case, which is what I did on paper.
                mx = oa[0].get("maxHp") or 0
                if mx:
                    opp_hp_by_turn[turn].append(mx)
                # A heal shows up as their Active's hp RISING with the same body in place.
                hk = (cur[0], mx, _energy(oa[0]) >= 3)
                h = oa[0].get("hp") or 0
                pk = (cur[0], "oa", mx)
                if pk in last_hp and h > last_hp[pk]:
                    S["opp_heal_events"] += 1
                    S["opp_heal_amount"] += h - last_hp[pk]
                last_hp[pk] = h
            # How the game is actually being lost: prizes remaining on each side, by turn.
            mp, op = len(me.get("prize") or []), len(opp.get("prize") or [])
            if isinstance(turn, int):
                pz_by_turn[turn].append((mp, op))
        except Exception:                                    # noqa: BLE001
            S["audit_errors"] += 1
        return pick

    inner = agent
    wins = 0
    for g in range(a.games):
        cur[0] = g
        watch(_EPISODE_START)
        mine = g % 2
        s = a.seed + g // 2
        r = (play(eng, watch, opp_agent, my_ids, opp_ids, s, mirror=1) if mine == 0
             else play(eng, opp_agent, watch, opp_ids, my_ids, s, mirror=1))
        wins += 1 if r == mine else 0
        if (g + 1) % 10 == 0:
            print("  %d games, %d wins" % (g + 1, wins), flush=True)

    n = max(1, S["decisions"])
    print("\n=== %s vs %s, %d games: %d wins (%.1f%%) ==="
          % (a.deck, a.opp, a.games, wins, 100.0 * wins / max(1, a.games)))
    print("decisions observed          %d" % S["decisions"])
    print("\n-- the Tera question --")
    print("  spread menus (6 counters) %d" % S["spread_menus"])
    print("  counters aimed at BENCHED Ogerpon %d" % S["counters_onto_benched_tera"])
    print("  bench TOTAL hp fell (count unchanged) %d times, %d hp"
          % (S["benched_tera_total_hp_fell"], S["benched_tera_hp_lost"]))
    if S["counters_onto_benched_tera"] and not S["benched_tera_total_hp_fell"]:
        print("  -> the engine PREVENTS it. Every counter aimed at their bench is discarded,")
        print("     and dusk_spread/dusk_plan have no rule that knows this.")
    elif S["benched_tera_total_hp_fell"]:
        print("  -> damage DOES reach their benched Tera bodies in this engine.")
    print("\n-- feeding Myriad Leaf Shower (30 + 30 per energy on BOTH actives) --")
    if my_e_at_their_hp:
        import statistics as st
        print("  energy on OUR active     mean %.2f  median %d  max %d"
              % (st.mean(my_e_at_their_hp), st.median(my_e_at_their_hp),
                 max(my_e_at_their_hp)))
    if S["opp_active_samples"]:
        oe = S["opp_active_energy_sum"] / S["opp_active_samples"]
        me_ = (sum(my_e_at_their_hp) / len(my_e_at_their_hp)) if my_e_at_their_hp else 0
        print("  energy on THEIR active   mean %.2f" % oe)
        print("  => typical Myriad Leaf Shower = 30 + 30*(%.1f+%.1f) = %.0f damage"
              % (oe, me_, 30 + 30 * (oe + me_)))
        print("     (of which %.0f is what WE attached)" % (30 * me_))
    print("\n-- disruption, counted PER TURN (a card in hand spans every menu of a turn) --")
    print("  %-8s %10s %8s %7s   %s" % ("card", "turns able", "played", "rate", "per game"))
    for nm in ("hammer", "judge", "stamp", "boss", "fan"):
        able, done = len(T[nm + "_turns"]), len(T[nm + "_played"])
        print("  %-8s %10d %8d %6.0f%%   %.2f"
              % (nm, able, done, 100.0 * done / max(1, able), done / max(1, a.games)))
    print("  boss with an EMPTY body on their bench to drag: %d turns"
          % len(T["boss_empty_available"]))
    import statistics as _st
    ts = sorted(k for k in opp_e_by_turn if isinstance(k, int))
    if ts:
        print("  their Active energy by turn: %s"
              % " ".join("t%d:%.1f" % (t, _st.mean(opp_e_by_turn[t]))
                         for t in ts[:12]))
    th = sorted(k for k in opp_hp_by_turn if isinstance(k, int))
    if th:
        print("\n-- what their Active actually IS (measured, not assumed) --")
        print("  their Active maxHP by turn:  %s"
              % " ".join("t%d:%.0f" % (t, _st.mean(opp_hp_by_turn[t])) for t in th[:12]))
        allhp = [v for t in th for v in opp_hp_by_turn[t]]
        import collections as _c
        print("  maxHP distribution: %s" % dict(_c.Counter(allhp).most_common(6)))
        print("  (210 = bare, 240 = Lively Stadium up, 310/340 = Hero's Cape)")
    print("  heal events %d, total %d hp healed (%.2f per game)"
          % (S["opp_heal_events"], S["opp_heal_amount"],
             S["opp_heal_events"] / max(1, a.games)))
    tp = sorted(k for k in pz_by_turn if isinstance(k, int))
    if tp:
        print("\n-- how the game is lost: prizes REMAINING --")
        print("  ours:   %s" % " ".join("t%d:%.1f" % (t, _st.mean([x for x, _y in pz_by_turn[t]]))
                                        for t in tp[:12]))
        print("  theirs: %s" % " ".join("t%d:%.1f" % (t, _st.mean([y for _x, y in pz_by_turn[t]]))
                                        for t in tp[:12]))
    print("\n-- Phantom Dive: 200 into a 210 HP body --")
    print("  offered %d, used %d" % (S["pd_offered"], S["pd_used"]))
    print("  when declined, we took instead: %s" % dict(declined.most_common(6)))
    print("\n-- Cursed Blast: the only damage that needs no energy --")
    for _nm, _dmg in (("dusclops", 50), ("dusknoir", 130)):
        o_, u_ = S["blast_%s_offered" % _nm], S["blast_%s_used" % _nm]
        print("  %-9s (%3d dmg) offered %4d, used %4d (%.0f%%)"
              % (_nm, _dmg, o_, u_, 100.0 * u_ / max(1, o_)))
    print("\n-- attacking at all --")
    print("  any attack offered %d, taken %d (%.0f%%)"
          % (S["attack_offered"], S["attack_taken"],
             100.0 * S["attack_taken"] / max(1, S["attack_offered"])))
    print("  attacks used by id: %s" % dict(attacks_used.most_common(8)))
    print("  spread target split: %s" % dict(spread_targets))
    if S["audit_errors"]:
        print("\naudit errors (ignored, never piloted): %d" % S["audit_errors"])
    if a.out:
        json.dump({"wins": wins, "games": a.games, "stats": dict(S),
                   "spread": dict(spread_targets)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
