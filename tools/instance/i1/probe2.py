"""Play-rate / bench-out probe that actually passes the deck PROFILE.

Every probe I wrote earlier today called `make_lm_agent(deck, None, None)`, and
`lm/agent.py:82` does `make_policy(deck, profile or {})` -- so profile=None means the engine ran
with NO tuning at all: no bench_target, no draw_supporters (which is where Proton lives), no
search_items, no card_roles, no line. Those measurements describe an unconfigured engine_v2, not
the shipped one. `tools/rl_rollout.py` passes `profiles.get(pilot)` correctly, so the RL run
itself was never affected.

PROBE_ROOT selects the repo (so a patched copy can be A/B'd against the original) and
PROBE_PROFILE=0 reproduces the old profile-less behaviour for comparison.

Run:  PROBE_ROOT=/root/ptcg/repo CUDA_VISIBLE_DEVICES="" python probe2.py DECK [games] [workers]
"""
import collections
import json
import os
import random
import sys

ROOT = os.environ.get("PROBE_ROOT", "/root/ptcg/repo")
USE_PROFILE = os.environ.get("PROBE_PROFILE", "1") != "0"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
# library.read_deck resolves decks/ RELATIVE TO CWD -- running from /root silently returned a
# short deck list and battle_start raised "The deck must contain 60 cards."
os.chdir(ROOT)

DECK = sys.argv[1] if len(sys.argv) > 1 else "rockets_honchkrow"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 900
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 48
OPPS = ["alakazam", "crustle", "dragapult"]

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
    return [c.get("id") for c in (cards or []) if isinstance(c, dict) and c.get("id") is not None]


def one_game(task):
    deck, opp, seed = task
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    from agents.engine_v2 import _CARDS

    prof = {}
    if USE_PROFILE:
        prof = json.load(open(os.path.join(ROOT, "agents/tuning.json")))
    try:
        d_me, d_op = library.read_deck(deck), library.read_deck(opp)
    except Exception:
        return None
    a_me = make_lm_agent(deck, prof.get(deck), None)     # <- the profile, finally
    a_op = make_lm_agent(opp, prof.get(opp), None)
    pilot_i = seed % 2
    d0, d1 = (d_me, d_op) if pilot_i == 0 else (d_op, d_me)
    obs, _ = battle_start(d0, d1)
    if obs is None:
        return None
    out = dict(opp=opp, held=collections.Counter(), danger_dec=0, my_dec=0)
    established = False
    try:
        for _ in range(4000):
            cur = obs.get("current")
            if cur is None:
                return None
            if cur.get("result", -1) != -1:
                w = cur["result"]
                me = cur["players"][pilot_i]
                out["won"] = 1 if w == pilot_i else 0
                out["bench_out"] = 1 if (w != pilot_i and _in_play(me) == 0) else 0
                out["deck_out"] = 1 if (w != pilot_i and not out["bench_out"]
                                        and (me.get("deckCount") or 0) == 0) else 0
                out["turns"] = cur.get("turn", -1)
                disc = collections.Counter(_ids(me.get("discard")))
                out["discard"] = {str(k): disc.get(k, 0) for k in WATCH}
                out["held"] = {str(k): v for k, v in out["held"].items()}
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
                if established and nip <= 1:
                    out["danger_dec"] += 1
                    for i in set(_ids(me.get("hand"))):
                        if i in WATCH:
                            out["held"][i] += 1
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
    print("probe2 %s | root=%s | profile=%s | %d games | %d workers"
          % (DECK, ROOT, "ON" if USE_PROFILE else "OFF", len(tasks), WORKERS), flush=True)
    import multiprocessing as mp
    gs = []
    with mp.Pool(WORKERS) as pool:
        for g in pool.imap_unordered(one_game, tasks, chunksize=1):
            if g is not None:
                gs.append(g)
    n = len(gs)
    if not n:
        print("no games completed")
        return
    w = sum(g["won"] for g in gs)
    bo = sum(g["bench_out"] for g in gs)
    do = sum(g["deck_out"] for g in gs)
    print("RESULT %-20s games %4d | winrate %5.1f%% | bench-out %4d (%4.1f%%) | deck-out %3d (%4.1f%%)"
          % (DECK, n, 100.0 * w / n, bo, 100.0 * bo / n, do, 100.0 * do / n))
    print("  play rate per game (copies in discard at end)   |  held while at <=1 in play")
    dd = sum(g["danger_dec"] for g in gs)
    for cid, nm in WATCH.items():
        a = sum(g["discard"].get(str(cid), 0) for g in gs) / n
        h = sum(g["held"].get(str(cid), 0) for g in gs)
        print("  %-32s %6.2f   |  %6d / %d danger decisions" % (nm, a, h, dd))
    print("  turns %.1f | pilot decisions %.1f/game" % (
        sum(g["turns"] for g in gs) / n, sum(g["my_dec"] for g in gs) / n))


if __name__ == "__main__":
    main()
