"""Would potential-based shaping with tools/eval_state.py point the policy the right way?

The user's proposal is to add a hand-crafted state evaluation (prize count, bench occupancy,
setup progress, deck-specific counters) to the reward. The bias-free form of that is
potential-based shaping, F(s,s') = gamma*Phi(s') - Phi(s), which leaves the optimal policy
unchanged (Ng/Harada/Russell 1999). So the question that decides the idea is:

    at a real branch point, does the candidate with the largest Delta-Phi actually win more?

`tools/eval_state.py` reports 51.4% winner/loser discrimination at 25% game progress -- i.e.
blind early -- but its own docstring notes that metric scores ABSOLUTE state value, while
shaping needs the DELTA across one move. A Phi can be a useless absolute ranker and still be a
usable delta detector. That is what this measures.

METHOD. Play engine_v2 vs engine_v2. At sampled pilot decisions, branch the top-K candidates
through the engine's native search tree and play each out NPLAY times. Split the playouts:

    q_sel = mean of the FIRST half     used only to rank (for the ceiling row)
    q_val = mean of the SECOND half    used only to score

For any ranker r, `E[q_val(argmax r) - mean q_val(others)]` is then UNBIASED: the selection
never sees the scoring noise. (Ranking and scoring on the same playouts is the winner's-curse
mistake that made an earlier "mean regret 0.321" uninterpretable.) Rankers compared:

    dphi        Delta of the full evaluator                     <- the proposal
    dprize      Delta of the prize count alone                  <- "basic numbers" only
    dtie        Delta of the tie-breakers alone (dphi - dprize)  <- does the soft part add?
    q_sel       an independent playout estimate                 <- honest ceiling at this budget
    random      control, must land on 0

Run:  CUDA_VISIBLE_DEVICES="" python measure_phi.py [games_per_pair] [workers]
"""
import collections
import itertools
import json
import math
import os
import random
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

GAMES_PER_PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 60
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
K = 4                 # candidates compared per branch point
NPLAY = 8             # playouts per candidate; split 4 rank / 4 score
PER_GAME = 14
ACCEPT = 0.18         # thin sampling so branch points spread over the whole game

# the 21 gate cells, so the answer is about the decks the gate actually judges
PILOTS = ["alakazam", "crustle", "dragapult", "dragapult_dusknoir",
          "marnie_grimmsnarl", "rockets_honchkrow", "rockets_mewtwo"]
OPPS = ["alakazam", "crustle", "dragapult"]
PAIRS = [(p, o) for p in PILOTS for o in OPPS]

PRIZE_TIER = 1000.0   # must match tools/eval_state.PRIZE_TIER


def _prizes(cur, yi):
    me = len(cur["players"][yi].get("prize") or [])
    op = len(cur["players"][1 - yi].get("prize") or [])
    return me, op


def _phi(raw_obs, me_idx, evaluate, to_obs):
    """Full evaluator on a RAW observation dict, from me_idx's view."""
    try:
        st = to_obs(raw_obs).current
        if not st or len(st.players or []) != 2:
            return None
        return evaluate(st, me_idx)
    except Exception:
        return None


def one_game(task):
    pilot, opp, seed = task
    import library
    import rl_branch
    import cg.api as api
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class
    from lm.agent import make_lm_agent
    from eval_state import evaluate

    rng = random.Random(seed)
    try:
        d_me, d_op = library.read_deck(pilot), library.read_deck(opp)
    except Exception:
        return []
    a_me = make_lm_agent(pilot, None, None)     # model=None -> engine_v2, no GPU
    a_op = make_lm_agent(opp, None, None)
    pilot_i = seed % 2
    d0, d1 = (d_me, d_op) if pilot_i == 0 else (d_op, d_me)
    obs, _ = battle_start(d0, d1)
    if obs is None:
        return []
    recs, taken = [], 0
    try:
        for _step in range(4000):
            cur = obs.get("current")
            if cur is None or cur.get("result", -1) != -1:
                break
            sel = obs.get("select")
            if sel is None:
                break
            yi = cur["yourIndex"]
            opts = sel.get("option") or []
            eligible = (yi == pilot_i and len(opts) >= 2
                        and sel.get("minCount", 1) == 1 and sel.get("maxCount", 1) == 1)
            if eligible and taken < PER_GAME and rng.random() < ACCEPT:
                rec = _branch_point(obs, cur, yi, opts, pilot, opp, pilot_i, d_me, d_op,
                                    a_me, a_op, rng, _step,
                                    rl_branch, api, to_observation_class, evaluate)
                if rec is not None:
                    recs.append(rec)
                    taken += 1
            obs = battle_select((a_me if yi == pilot_i else a_op)(obs))
    except Exception:
        pass
    finally:
        try:
            battle_finish()
        except Exception:
            pass
    return recs


def _branch_point(obs, cur, yi, opts, pilot, opp, pilot_i, d_me, d_op,
                  a_me, a_op, rng, step_i,
                  rl_branch, api, to_obs, evaluate):
    try:
        mu, ou = rl_branch.unseen_multisets(obs, d_me, d_op)
    except rl_branch.DeterminizationError:
        return None
    except Exception:
        return None
    kk = min(K, len(opts))
    phi0 = _phi(obs, pilot_i, evaluate, to_obs)
    if phi0 is None:
        return None
    my_p0, op_p0 = _prizes(cur, yi)

    outcomes = [[] for _ in range(kk)]      # playout results per candidate
    dphis = [[] for _ in range(kk)]         # Delta-Phi per candidate, per scenario
    dprz = [[] for _ in range(kk)]          # Delta prize lead (in Phi units)
    for _rep in range(NPLAY):
        m2, o2 = list(mu), list(ou)
        rng.shuffle(m2)
        rng.shuffle(o2)
        try:
            root = api.search_begin(api.to_observation_class(obs), m2, m2, o2, o2, o2, [])
        except Exception:
            continue
        try:
            for k in range(kk):
                st = rl_branch._raw_step(root.searchId, [k])
                if st.get("error", 0) != 0 or not st.get("state"):
                    continue
                nxt = (st["state"].get("observation") or {})
                ncur = nxt.get("current")
                if ncur is not None and ncur.get("result", -1) == -1:
                    p1 = _phi(nxt, pilot_i, evaluate, to_obs)
                    if p1 is not None:
                        dphis[k].append(p1 - phi0)
                        # prize lead is PRIZE_TIER*(op_left - my_left) from the PILOT's view
                        nyi = ncur["yourIndex"]
                        a, b = _prizes(ncur, nyi)
                        my1, op1 = (a, b) if nyi == pilot_i else (b, a)
                        dprz[k].append(PRIZE_TIER * ((op1 - my1) - (op_p0 - my_p0)))
                v = rl_branch._playout(st["state"], pilot_i, a_me, a_op)
                if v is not None:
                    outcomes[k].append(v)
        finally:
            api.search_end()

    half = NPLAY // 2
    if not all(len(v) >= NPLAY - 2 for v in outcomes):
        return None                      # need both halves populated for the split estimator
    if not all(len(v) >= 1 for v in dphis):
        return None
    return dict(pair="%s__vs__%s" % (pilot, opp),
                turn=cur.get("turn", -1), step=step_i, n_opts=len(opts),
                my_prizes=my_p0, op_prizes=op_p0,
                q_sel=[sum(v[:half]) / len(v[:half]) for v in outcomes],
                q_val=[sum(v[half:]) / max(1, len(v[half:])) for v in outcomes],
                dphi=[sum(v) / len(v) for v in dphis],
                dprize=[sum(v) / len(v) for v in dprz] if all(dprz) else None)


# ---------------------------------------------------------------- analysis

def _pick_value(recs, key_fn, rng=None):
    """E[q_val(argmax r) - mean q_val(others)], with SE. Unbiased: r never sees q_val."""
    vals = []
    for r in recs:
        qv = r["q_val"]
        if len(qv) < 2:
            continue
        sc = key_fn(r, rng)
        if sc is None:
            continue
        i = max(range(len(qv)), key=lambda j: sc[j])
        others = [qv[j] for j in range(len(qv)) if j != i]
        vals.append(qv[i] - sum(others) / len(others))
    if not vals:
        return None, None, 0
    n = len(vals)
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / max(1, n - 1)
    return mu, math.sqrt(var / n), n


def _kendall(a, b):
    n = len(a)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / max(1, c + d)


RANKERS = [
    ("dphi   (full evaluator delta)", lambda r, g: r["dphi"]),
    ("dprize (prize count only)", lambda r, g: r["dprize"]),
    ("dtie   (tie-breakers only)",
     lambda r, g: None if r["dprize"] is None
     else [p - z for p, z in zip(r["dphi"], r["dprize"])]),
    ("q_sel  (independent playouts)", lambda r, g: r["q_sel"]),
    ("random (control)", lambda r, g: [g.random() for _ in r["q_val"]]),
]


def report(tag, recs):
    print("\n=== %s  (%d branch points) ===" % (tag, len(recs)))
    if not recs:
        return
    print("  %-32s %9s %9s %8s" % ("ranker", "pick-val", "SE", "n"))
    for name, fn in RANKERS:
        g = random.Random(7)
        mu, se, n = _pick_value(recs, fn, g)
        if mu is None:
            print("  %-32s %9s" % (name, "n/a"))
            continue
        star = ""
        if se and abs(mu) > 2 * se:
            star = "  *"
        print("  %-32s %+9.4f %9.4f %8d%s" % (name, mu, se, n, star))
    ks = [_kendall(r["dphi"], r["q_val"]) for r in recs if len(r["q_val"]) > 1]
    if ks:
        print("  kendall tau(dphi, q_val) within branch point: %+.4f  (n=%d)"
              % (sum(ks) / len(ks), len(ks)))


def main():
    tasks = [(p, o, s) for (p, o) in PAIRS for s in range(GAMES_PER_PAIR)]
    random.Random(0).shuffle(tasks)
    print("measure_phi: %d games over %d pairs, K=%d, %d playouts (%d rank / %d score), "
          "%d workers" % (len(tasks), len(PAIRS), K, NPLAY, NPLAY // 2, NPLAY // 2, WORKERS))
    import multiprocessing as mp
    recs = []
    with mp.Pool(WORKERS) as pool:
        for i, rr in enumerate(pool.imap_unordered(one_game, tasks, chunksize=1)):
            recs.extend(rr)
            if (i + 1) % 100 == 0:
                print("  %d/%d games, %d branch points" % (i + 1, len(tasks), len(recs)),
                      flush=True)
    with open("/root/phi_recs.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("\nwrote /root/phi_recs.jsonl (%d branch points)" % len(recs))

    report("ALL", recs)
    # signal lives at 2-4 prizes and is 0 at <=1 (measured 2026-07-29); split accordingly
    for lo, hi, lbl in ((6, 6, "opening (6 prizes)"), (4, 5, "midgame (4-5)"),
                        (2, 3, "late (2-3)"), (0, 1, "endgame (<=1)")):
        report(lbl, [r for r in recs if lo <= r["my_prizes"] <= hi])
    by_pair = collections.defaultdict(list)
    for r in recs:
        by_pair[r["pair"]].append(r)
    print("\n=== dphi pick-value per cell ===")
    print("  %-40s %9s %9s %7s" % ("cell", "dphi", "SE", "n"))
    rows = []
    for pr, rs in by_pair.items():
        mu, se, n = _pick_value(rs, lambda r, g: r["dphi"])
        if mu is not None:
            rows.append((mu, se, n, pr))
    for mu, se, n, pr in sorted(rows, reverse=True):
        print("  %-40s %+9.4f %9.4f %7d" % (pr, mu, se, n))


if __name__ == "__main__":
    main()
