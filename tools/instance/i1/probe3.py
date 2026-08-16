"""Profile ablation: which tuning.json key makes rockets_honchkrow WORSE than no tuning at all?

Measured 2026-07-30 (900 games/arm, vs alakazam/crustle/dragapult):

    profile OFF (engine defaults)   winrate 37.2%   bench-out 57.3%   deck-out  0.2%
    profile ON  (shipped)           winrate 22.0%   bench-out 40.3%   deck-out 15.1%

The shipped tuning costs ~15pt and turns the deck into a self-mill: with the profile the engine
plays Transceiver 2.30/game and Roto-Stick 2.28/game instead of 0.18 each. So the question is
not "add a rule" but "which existing key is net-negative".

Each ARM is the full profile with ONE key removed or changed, so a jump in win rate names the
culprit. Also records the deck-count trajectory and what got played in the last three turns,
because "decked out" and "decked out while drawing" are different bugs.

ARMS
    full                       the shipped profile
    empty                      no profile at all
    no:KEY                     drop that key (draw_supporters, search_items, card_roles,
                               line, l2, gust_cards, switch_cards, bench_target, main_attackers)
    set:draw_threshold=N       change the hand-size gate

Run:  PROBE_ROOT=... PROBE_ARM=full CUDA_VISIBLE_DEVICES="" python probe3.py DECK [games] [workers]
"""
import collections
import json
import os
import random
import sys

ROOT = os.environ.get("PROBE_ROOT", "/root/ptcg/repo")
ARM = os.environ.get("PROBE_ARM", "full")
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)          # library.read_deck resolves decks/ relative to cwd

DECK = sys.argv[1] if len(sys.argv) > 1 else "rockets_honchkrow"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
OPPS = ["alakazam", "crustle", "dragapult"]


def build_profile(arm, deck):
    """Arms compose with ';' so a two-defect deck can be fixed in one arm."""
    prof = json.load(open(os.path.join(ROOT, "agents/tuning.json")))
    p = dict(prof.get(deck) or {})
    if ";" in arm:
        for part in arm.split(";"):
            p = _apply(part, p)
        return p
    return _apply(arm, p)


def _apply(arm, p):
    if arm == "empty":
        return {}
    if arm == "full":
        return p
    if arm.startswith("no:"):
        for k in arm[3:].split("+"):
            p.pop(k, None)
        return p
    if arm.startswith("roles:"):
        # re-tier specific cardIds: roles:463=win+473=win+891=engine
        cr = dict(p.get("card_roles") or {})
        for kv in arm[6:].split("+"):
            k, v = kv.split("=")
            cr[str(k)] = v
        p["card_roles"] = cr
        return p
    if arm.startswith("set:"):
        for kv in arm[4:].split("+"):
            k, v = kv.split("=")
            p[k] = int(v) if v.lstrip("-").isdigit() else v
        return p
    raise SystemExit("unknown arm: %s" % arm)


def _in_play(pl):
    act = pl.get("active") or []
    return (1 if (act and act[0] is not None) else 0) + len(pl.get("bench") or [])


def one_game(task):
    deck, opp, seed = task
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent

    prof = json.load(open(os.path.join(ROOT, "agents/tuning.json")))
    try:
        d_me, d_op = library.read_deck(deck), library.read_deck(opp)
    except Exception:
        return None
    a_me = make_lm_agent(deck, build_profile(ARM, deck), None)
    a_op = make_lm_agent(opp, prof.get(opp), None)      # opponents always fully configured
    pilot_i = seed % 2
    d0, d1 = (d_me, d_op) if pilot_i == 0 else (d_op, d_me)
    obs, _ = battle_start(d0, d1)
    if obs is None:
        return None
    # deck count at the START of each of my turns, so the mill rate is visible
    traj, last_turn = [], -1
    try:
        for _ in range(4000):
            cur = obs.get("current")
            if cur is None:
                return None
            if cur.get("result", -1) != -1:
                w = cur["result"]
                me = cur["players"][pilot_i]
                bo = 1 if (w != pilot_i and _in_play(me) == 0) else 0
                do = 1 if (w != pilot_i and not bo
                           and (me.get("deckCount") or 0) == 0) else 0
                return dict(won=1 if w == pilot_i else 0, bench_out=bo, deck_out=do,
                            turns=cur.get("turn", -1), traj=traj,
                            final_deck=me.get("deckCount"))
            if obs.get("select") is None:
                return None
            yi = cur["yourIndex"]
            if yi == pilot_i:
                t = cur.get("turn", -1)
                if t != last_turn:
                    last_turn = t
                    traj.append((t, cur["players"][yi].get("deckCount")))
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
    import multiprocessing as mp
    gs = []
    with mp.Pool(WORKERS) as pool:
        for g in pool.imap_unordered(one_game, tasks, chunksize=1):
            if g is not None:
                gs.append(g)
    n = len(gs)
    if not n:
        print("ARM %-28s no games completed" % ARM)
        return
    w = sum(g["won"] for g in gs)
    bo = sum(g["bench_out"] for g in gs)
    do = sum(g["deck_out"] for g in gs)
    # mill rate: cards drawn per own turn, from the deck-count trajectory
    rates = []
    for g in gs:
        tr = [x for x in g["traj"] if x[1] is not None]
        if len(tr) >= 3:
            rates.append((tr[0][1] - tr[-1][1]) / max(1, len(tr) - 1))
    mr = sum(rates) / len(rates) if rates else float("nan")
    # how many of my turns did the game last
    myt = sum(len(g["traj"]) for g in gs) / n
    print("ARM %-28s n%5d | win %5.1f%% | bench-out %5.1f%% | deck-out %5.1f%% | "
          "my turns %4.1f | cards/turn %4.2f"
          % (ARM, n, 100.0 * w / n, 100.0 * bo / n, 100.0 * do / n, myt, mr))


if __name__ == "__main__":
    main()
