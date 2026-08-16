"""How do games actually END, and could a guard have prevented the bench-out losses?

Classified at the terminal observation:

    prize-out   the WINNER has 0 prize cards left  (it took its last prize)
    bench-out   the LOSER has no Pokemon in play   (active is None AND bench empty)
    deck-out    the LOSER's deckCount is 0         (had to draw and could not)
    unknown     none of the above matched          (a bug in this classifier, report it)

A first smoke run put bench-out at 26% of games, so the follow-up question is what a guard
could do about it. For every bench-out loss this records the pilot's LAST decision before the
end: in-play count, hand size, and how many BASIC Pokemon were in hand. That splits the losses
into

    FIXABLE     had a Basic in hand while down to one Pokemon in play -> a legality mask
                or a forced-bench rule can prevent it
    UNFIXABLE   no Basic in hand -> a draw/deckbuilding problem, no piloting fixes it

DANGER EXPOSURE counts decisions at <=1 Pokemon in play, but only AFTER the pilot has first
established an Active -- during the opening setup everyone is at 0 in play, and counting that
made the first version report 100% of games as "hit 0 in play", which is true and useless.

engine_v2 pilots both sides, so bench-out here is a LOWER BOUND for the LM: engine_v2 never
declines an optional pick, while a sampling policy can. The LM run needs the GPU.

Run:  CUDA_VISIBLE_DEVICES="" python loss_causes.py [games_per_pair] [workers]
"""
import collections
import json
import os
import random
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

GAMES_PER_PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 100
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 64

PILOTS = ["alakazam", "crustle", "dragapult", "dragapult_dusknoir",
          "marnie_grimmsnarl", "rockets_honchkrow", "rockets_mewtwo"]
OPPS = ["alakazam", "crustle", "dragapult"]
PAIRS = [(p, o) for p in PILOTS for o in OPPS]


def _in_play(pl):
    act = pl.get("active") or []
    n = 1 if (act and act[0] is not None) else 0
    return n + len(pl.get("bench") or [])


def _basics_in_hand(pl, cards):
    n = 0
    for c in (pl.get("hand") or []):
        cd = cards.get(c.get("id")) if isinstance(c, dict) else None
        if cd is not None and cd.basic:
            n += 1
    return n


def classify(cur, winner):
    loser = 1 - winner
    w, l = cur["players"][winner], cur["players"][loser]
    lip = _in_play(l)
    if len(w.get("prize") or []) == 0:
        return "prize-out", lip
    if lip == 0:
        return "bench-out", lip
    if (l.get("deckCount") or 0) == 0:
        return "deck-out", lip
    return "unknown", lip


def one_game(task):
    pilot, opp, seed = task
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    from agents.engine_v2 import _CARDS

    try:
        d_me, d_op = library.read_deck(pilot), library.read_deck(opp)
    except Exception:
        return None
    a_me = make_lm_agent(pilot, None, None)     # model=None -> engine_v2, no GPU
    a_op = make_lm_agent(opp, None, None)
    pilot_i = seed % 2
    d0, d1 = (d_me, d_op) if pilot_i == 0 else (d_op, d_me)
    obs, _ = battle_start(d0, d1)
    if obs is None:
        return None
    out = dict(pair="%s__vs__%s" % (pilot, opp), pilot=pilot, opp=opp,
               danger_dec=0, my_dec=0, min_in_play=99, turns=0,
               last=None, danger_with_basic=0)
    established = False
    try:
        for _ in range(4000):
            cur = obs.get("current")
            if cur is None:
                return None
            if cur.get("result", -1) != -1:
                w = cur["result"]
                cause, lip = classify(cur, w)
                out["turns"] = cur.get("turn", -1)
                out["won"] = 1 if w == pilot_i else 0
                out["cause"] = cause
                out["loser_in_play"] = lip
                return out
            sel = obs.get("select")
            if sel is None:
                return None
            yi = cur["yourIndex"]
            if yi == pilot_i:
                me = cur["players"][yi]
                nip = _in_play(me)
                if nip >= 1:
                    established = True
                out["my_dec"] += 1
                nb = _basics_in_hand(me, _CARDS)
                # the pilot's last state before whatever ends the game
                out["last"] = dict(turn=cur.get("turn", -1), in_play=nip,
                                   hand=len(me.get("hand") or []), basics=nb,
                                   deck=me.get("deckCount"))
                if established:
                    out["min_in_play"] = min(out["min_in_play"], nip)
                    if nip <= 1 and len(sel.get("option") or []) >= 2:
                        out["danger_dec"] += 1
                        if nb > 0:
                            out["danger_with_basic"] += 1
            obs = battle_select((a_me if yi == pilot_i else a_op)(obs))
        return None
    except Exception:
        return None
    finally:
        try:
            battle_finish()
        except Exception:
            pass


def main():
    tasks = [(p, o, s) for (p, o) in PAIRS for s in range(GAMES_PER_PAIR)]
    random.Random(0).shuffle(tasks)
    print("loss_causes: %d games over %d pairs, %d workers (engine_v2 both sides)"
          % (len(tasks), len(PAIRS), WORKERS), flush=True)
    import multiprocessing as mp
    games = []
    with mp.Pool(WORKERS) as pool:
        for i, g in enumerate(pool.imap_unordered(one_game, tasks, chunksize=1)):
            if g is not None:
                games.append(g)
            if (i + 1) % 300 == 0:
                print("  %d/%d" % (i + 1, len(tasks)), flush=True)
    with open("/root/loss_causes.jsonl", "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")
    print("\n%d games completed of %d started" % (len(games), len(tasks)))

    c = collections.Counter(g["cause"] for g in games)
    print("\n=== how games end (%d games) ===" % len(games))
    for k, v in c.most_common():
        print("  %-12s %6d   %5.2f%%" % (k, v, 100.0 * v / max(1, len(games))))

    lost = [g for g in games if not g["won"]]
    cl = collections.Counter(g["cause"] for g in lost)
    print("\n=== cause of the PILOT's losses (%d losses) ===" % len(lost))
    for k, v in cl.most_common():
        print("  %-12s %6d   %5.2f%%" % (k, v, 100.0 * v / max(1, len(lost))))

    # ---- THE question: could a guard have prevented the bench-out losses?
    bo = [g for g in lost if g["cause"] == "bench-out" and g.get("last")]
    print("\n=== the %d bench-out losses: fixable by a guard? ===" % len(bo))
    if bo:
        fix = [g for g in bo if g["last"]["basics"] > 0]
        print("  had a Basic in hand at the last decision : %d / %d  (%.1f%%)  <- FIXABLE"
              % (len(fix), len(bo), 100.0 * len(fix) / len(bo)))
        print("  had NO Basic in hand                     : %d / %d  (%.1f%%)  <- draw problem"
              % (len(bo) - len(fix), len(bo), 100.0 * (len(bo) - len(fix)) / len(bo)))
        tn = sorted(g["last"]["turn"] for g in bo)
        hs = sorted(g["last"]["hand"] for g in bo)
        print("  turn at the last decision   median %d   (p10 %d, p90 %d)"
              % (tn[len(tn) // 2], tn[len(tn) // 10], tn[min(len(tn) - 1, 9 * len(tn) // 10)]))
        print("  hand size at that decision  median %d   (p10 %d, p90 %d)"
              % (hs[len(hs) // 2], hs[len(hs) // 10], hs[min(len(hs) - 1, 9 * len(hs) // 10)]))
        ipd = collections.Counter(g["last"]["in_play"] for g in bo)
        print("  in-play count there:", dict(sorted(ipd.items())))

    print("\n=== danger exposure (after the pilot first had an Active) ===")
    ever = sum(1 for g in games if g["min_in_play"] <= 1)
    dd = sum(g["danger_dec"] for g in games)
    dwb = sum(g["danger_with_basic"] for g in games)
    md = sum(g["my_dec"] for g in games)
    print("  games ever at <=1 in play        : %d / %d  (%.1f%%)"
          % (ever, len(games), 100.0 * ever / max(1, len(games))))
    print("  decisions there / all decisions  : %d / %d  (%.2f%%)"
          % (dd, md, 100.0 * dd / max(1, md)))
    print("  ...of those, a Basic was in hand : %d  (%.1f%% of danger decisions)"
          % (dwb, 100.0 * dwb / max(1, dd)))
    print("  = the only states a guard could act in")

    print("\n=== per pilot deck ===")
    print("  %-22s %7s %8s %10s %9s %11s" % ("deck", "games", "winrate", "bench-out",
                                             "deck-out", "of BO: fixable"))
    by = collections.defaultdict(list)
    for g in games:
        by[g["pilot"]].append(g)
    for d, gs in sorted(by.items()):
        ls = [g for g in gs if not g["won"]]
        b = [g for g in ls if g["cause"] == "bench-out" and g.get("last")]
        do = sum(1 for g in ls if g["cause"] == "deck-out")
        fx = sum(1 for g in b if g["last"]["basics"] > 0)
        print("  %-22s %7d %7.1f%% %7d    %6d    %6s"
              % (d, len(gs), 100.0 * sum(g["won"] for g in gs) / max(1, len(gs)),
                 len(b), do, ("%d/%d" % (fx, len(b))) if b else "-"))


if __name__ == "__main__":
    main()
