"""Why does rockets_honchkrow bench out in half its games when it runs 12+ Basic-fetchers?

The decklist is a human top-tier Team Rocket list: Team Rocket's Proton x4 searches up to THREE
Basic Team Rocket's Pokemon, plus Poke Pad x4, Ultra Ball x1, Night Stretcher x3, and the
second-order fetchers Transceiver x4 / Petrel x4. A deck with 8 Basics and that engine should
not run out of Pokemon. So the bench-out losses point at the PILOT, not the 60.

Two suspects, from `agents/tuning.json`:
  * Proton (1220) is registered in `draw_supporters`, not `search_items` -- it is played under a
    hand-size gate (draw_threshold 5) even though its effect is a Basic search.
  * Poke Pad (1152) is in NEITHER bucket.
`tools/audit_dead_buckets.py` flags Proton statically; its own docstring says confirm with a
play-rate probe first, which is this.

MEASURED, per game, without decoding the option format:
  * play count  = copies in the pilot's DISCARD at the end. Items and Supporters go to the
    discard when played. Over-counts slightly, because Ultra Ball discards 2 cards from hand.
  * HELD-AND-DIED = decisions where the card sat in hand while the pilot was down to <=1
    Pokemon in play. This is the decisive one: holding the fix and losing anyway is a piloting
    failure, and no decklist change addresses it.

Run:  CUDA_VISIBLE_DEVICES="" python probe_honchkrow.py [deck] [games] [workers]
"""
import collections
import json
import os
import random
import sys

ROOT = "/root/ptcg/repo_fix"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

DECK = sys.argv[1] if len(sys.argv) > 1 else "rockets_honchkrow"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 300
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
OPPS = ["alakazam", "crustle", "dragapult"]

# the cards whose job is to keep Pokemon on the board
WATCH = {
    1220: "TR Proton (search 3 Basics)",
    1152: "Poke Pad (search Pokemon)",
    1121: "Ultra Ball (search Pokemon)",
    1097: "Night Stretcher (recover)",
    1134: "TR Transceiver (find TR supp)",
    1219: "TR Petrel (find Trainer)",
    1216: "TR Ariana (draw to 5/8)",
    1077: "Roto-Stick (top 4 -> supp)",
}


def _in_play(pl):
    act = pl.get("active") or []
    return (1 if (act and act[0] is not None) else 0) + len(pl.get("bench") or [])


def _ids(cards):
    out = []
    for c in (cards or []):
        i = c.get("id") if isinstance(c, dict) else None
        if i is not None:
            out.append(i)
    return out


def one_game(task):
    deck, opp, seed = task
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    from agents.engine_v2 import _CARDS

    try:
        d_me, d_op = library.read_deck(deck), library.read_deck(opp)
    except Exception:
        return None
    a_me = make_lm_agent(deck, None, None)      # model=None -> engine_v2, no GPU
    a_op = make_lm_agent(opp, None, None)
    pilot_i = seed % 2
    d0, d1 = (d_me, d_op) if pilot_i == 0 else (d_op, d_me)
    obs, _ = battle_start(d0, d1)
    if obs is None:
        return None
    out = dict(opp=opp, held=collections.Counter(), seen_hand=collections.Counter(),
               danger_dec=0, my_dec=0)
    established = False
    try:
        for _ in range(4000):
            cur = obs.get("current")
            if cur is None:
                return None
            if cur.get("result", -1) != -1:
                w = cur["result"]
                me = cur["players"][pilot_i]
                lo = cur["players"][1 - w]
                out["won"] = 1 if w == pilot_i else 0
                out["bench_out"] = (w != pilot_i and _in_play(me) == 0)
                out["turns"] = cur.get("turn", -1)
                disc = collections.Counter(_ids(me.get("discard")))
                out["discard"] = {str(k): disc.get(k, 0) for k in WATCH}
                out["final_basics_hand"] = sum(
                    1 for i in _ids(me.get("hand"))
                    if _CARDS.get(i) is not None and _CARDS[i].basic)
                out["final_hand"] = len(me.get("hand") or [])
                out["held"] = {str(k): v for k, v in out["held"].items()}
                out["seen_hand"] = {str(k): v for k, v in out["seen_hand"].items()}
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
                hand = set(_ids(me.get("hand")))
                for i in hand:
                    if i in WATCH:
                        out["seen_hand"][i] += 1
                if established and nip <= 1:
                    out["danger_dec"] += 1
                    for i in hand:
                        if i in WATCH:
                            out["held"][i] += 1        # held the fix while one KO from death
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
    tasks = [(DECK, o, s) for o in OPPS for s in range(GAMES // len(OPPS))]
    random.Random(0).shuffle(tasks)
    print("probe %s: %d games vs %s, %d workers" % (DECK, len(tasks), OPPS, WORKERS),
          flush=True)
    import multiprocessing as mp
    gs = []
    with mp.Pool(WORKERS) as pool:
        for i, g in enumerate(pool.imap_unordered(one_game, tasks, chunksize=1)):
            if g is not None:
                gs.append(g)
    with open("/root/probe_%s.jsonl" % DECK, "w") as f:
        for g in gs:
            f.write(json.dumps(g) + "\n")
    n = len(gs)
    bo = [g for g in gs if g.get("bench_out")]
    print("\n%d games | winrate %.1f%% | bench-out losses %d (%.1f%%)"
          % (n, 100.0 * sum(g["won"] for g in gs) / max(1, n), len(bo),
             100.0 * len(bo) / max(1, n)))

    print("\n=== play rate (copies in discard at game end, per game) ===")
    print("  %-32s %8s %8s %10s" % ("card", "all", "bench-out", "in deck"))
    import library
    from collections import Counter
    dcnt = Counter(library.read_deck(DECK))
    for cid, nm in WATCH.items():
        a = sum(g["discard"].get(str(cid), 0) for g in gs) / max(1, n)
        b = sum(g["discard"].get(str(cid), 0) for g in bo) / max(1, len(bo))
        print("  %-32s %8.2f %8.2f %10d" % (nm, a, b, dcnt.get(cid, 0)))

    print("\n=== HELD-AND-DIED: in hand while at <=1 Pokemon in play ===")
    print("  (decisions in that state: %d over %d games; %d in bench-out games)"
          % (sum(g["danger_dec"] for g in gs), n, sum(g["danger_dec"] for g in bo)))
    print("  %-32s %14s %14s" % ("card", "held (all)", "held (bench-out)"))
    for cid, nm in WATCH.items():
        a = sum(g["held"].get(str(cid), 0) for g in gs)
        b = sum(g["held"].get(str(cid), 0) for g in bo)
        print("  %-32s %14d %14d" % (nm, a, b))

    gbo = [g for g in bo]
    if gbo:
        fb = sum(g["final_basics_hand"] for g in gbo) / len(gbo)
        fh = sum(g["final_hand"] for g in gbo) / len(gbo)
        print("\n  in bench-out losses: final hand %.1f cards, %.2f of them Basic Pokemon"
              % (fh, fb))
        pr = sum(1 for g in gbo if g["discard"].get("1220", 0) == 0)
        print("  bench-out losses where Proton was NEVER played: %d / %d (%.1f%%)"
              % (pr, len(gbo), 100.0 * pr / len(gbo)))


if __name__ == "__main__":
    main()
