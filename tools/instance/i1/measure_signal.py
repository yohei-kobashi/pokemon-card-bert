"""How much decision-level signal is actually there?

The question RL depends on: at a branch point, does the CHOICE change the outcome, or is the
game already decided? A raw "Q spread" cannot answer it -- with n playouts of a +/-1 outcome,
candidates that are truly identical still spread by ~1/sqrt(n). So decompose the variance:

    MS_between = n * sum_k (mean_k - grand)^2 / (K-1)      signal + noise
    MS_within  = sum_k sum_i (x - mean_k)^2 / (K(n-1))     noise alone
    var_signal_hat = (MS_between - MS_within) / n          unbiased, may go negative

Pooling MS across many branch points makes this precise even though one branch point is not.
Reported per matchup and per game phase, because a decided game has no signal by construction
and mixing the two is how "20% of points differ" could mean anything.

Run:  CUDA_VISIBLE_DEVICES="" python measure_signal.py [games_per_pair] [workers]
"""
import collections
import json
import math
import os
import random
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

GAMES_PER_PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 40
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
K = 4                 # candidates compared per branch point
NPLAY = 8             # playouts per candidate
PER_GAME = 14         # cap only; the acceptance rate below sets the real count
ACCEPT = 0.18         # uniform over eligible decisions -- a low cap biases to the opening

PAIRS = [
    ("alakazam", "alakazam"),          # mirror
    ("alakazam", "dragapult"),         # engine 81.7% -- favourable
    ("dragapult", "crustle_stall"),    # engine  7.0% -- hopeless
    ("crustle_stall", "alakazam"),     # engine 63.3%
    ("rockets_mewtwo", "alakazam"),    # engine 78.3% -- our worst LM cell
]


def one_game(task):
    """Play one game; at up to PER_GAME pilot decisions, branch K candidates x NPLAY
    playouts and record the raw outcomes. Returns a list of branch-point records."""
    pilot, opp, seed = task
    import library
    import rl_branch
    import cg.api as api
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent

    rng = random.Random(seed)
    d_me, d_op = library.read_deck(pilot), library.read_deck(opp)
    a_me = make_lm_agent(pilot, None, None)
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
            # sample branch points thinly so they spread over the whole game
            if eligible and taken < PER_GAME and rng.random() < ACCEPT:
                try:
                    mu, ou = rl_branch.unseen_multisets(obs, d_me, d_op)
                except rl_branch.DeterminizationError:
                    mu = None
                if mu is not None:
                    kk = min(K, len(opts))
                    o = api.to_observation_class(obs)
                    outcomes = [[] for _ in range(kk)]
                    for _rep in range(NPLAY):
                        # a FRESH determinization per repetition (shuffled hidden pool),
                        # shared by all kk candidates -> common random numbers within a
                        # scenario, averaged across scenarios
                        m2, o2 = list(mu), list(ou)
                        rng.shuffle(m2)
                        rng.shuffle(o2)
                        try:
                            root = api.search_begin(
                                api.to_observation_class(obs), m2, m2, o2, o2, o2, [])
                        except Exception:
                            continue
                        for k in range(kk):
                            st = rl_branch._raw_step(root.searchId, [k])
                            if st.get("error", 0) != 0 or not st.get("state"):
                                continue
                            v = rl_branch._playout(st["state"], pilot_i, a_me, a_op)
                            if v is not None:
                                outcomes[k].append(v)
                        api.search_end()
                    if all(len(v) >= 2 for v in outcomes):
                        me_pl = cur["players"][yi]
                        op_pl = cur["players"][1 - yi]
                        recs.append(dict(
                            pair="%s__vs__%s" % (pilot, opp),
                            turn=cur.get("turn", -1),
                            my_prizes=len(me_pl.get("prize") or []),
                            op_prizes=len(op_pl.get("prize") or []),
                            n_opts=len(opts),
                            step=_step,
                            outcomes=outcomes))
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


def anova(outcomes):
    """(MS_between, MS_within, n_eff, K) for one branch point."""
    ks = [v for v in outcomes if len(v) >= 2]
    if len(ks) < 2:
        return None
    n = min(len(v) for v in ks)
    ks = [v[:n] for v in ks]
    kk = len(ks)
    means = [sum(v) / n for v in ks]
    grand = sum(means) / kk
    ss_b = n * sum((m - grand) ** 2 for m in means)
    ss_w = sum(sum((x - m) ** 2 for x in v) for v, m in zip(ks, means))
    ms_b = ss_b / max(1, kk - 1)
    ms_w = ss_w / max(1, kk * (n - 1))
    return ms_b, ms_w, n, kk


def report(tag, recs):
    rows = [anova(r["outcomes"]) for r in recs]
    rows = [r for r in rows if r]
    if not rows:
        print("  %-28s (no usable branch points)" % tag)
        return
    mb = sum(r[0] for r in rows) / len(rows)
    mw = sum(r[1] for r in rows) / len(rows)
    n = sum(r[2] for r in rows) / len(rows)
    var_sig = max(0.0, (mb - mw) / max(n, 1e-9))
    # share of branch points where the choice demonstrably matters (F test, p<0.05-ish)
    hits = 0
    for ms_b, ms_w, nn, kk in rows:
        if ms_w <= 1e-12:
            hits += 1 if ms_b > 1e-12 else 0
            continue
        f = ms_b / ms_w
        df1, df2 = kk - 1, kk * (nn - 1)
        if f > 2.8 and df2 >= 8:      # ~p<0.05 for df1=3, df2=28
            hits += 1
    print("  %-28s pts %4d | MS_b %.3f  MS_w %.3f | sd_signal %.3f | decisive %4.1f%%"
          % (tag, len(rows), mb, mw, math.sqrt(var_sig), 100.0 * hits / len(rows)))


def main():
    from multiprocessing import Pool
    tasks = []
    for pi, (a, b) in enumerate(PAIRS):
        for g in range(GAMES_PER_PAIR):
            tasks.append((a, b, pi * 1000 + g))
    with Pool(WORKERS) as pool:
        out = pool.map(one_game, tasks, chunksize=1)
    recs = [r for sub in out for r in sub]
    print("branch points collected: %d  (from %d games)" % (len(recs), len(tasks)))
    if not recs:
        return
    with open("/root/signal_recs.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    print("\n=== overall ===")
    report("ALL", recs)

    print("\n=== by matchup ===")
    by = collections.defaultdict(list)
    for r in recs:
        by[r["pair"]].append(r)
    for k in sorted(by):
        report(k, by[k])

    print("\n=== by game phase (prizes the PILOT still has) ===")
    ph = collections.defaultdict(list)
    for r in recs:
        ph["prizes %d" % r["my_prizes"]].append(r)
    for k in sorted(ph, reverse=True):
        report(k, ph[k])

    print("\n=== sampling check: where in the game did branch points land? ===")
    st = collections.Counter()
    for r in recs:
        st["step %s" % (min(r.get("step", 0) // 50 * 50, 300))] += 1
    for k, v in sorted(st.items(), key=lambda kv: int(kv[0].split()[1])):
        print("  %-12s %d" % (k, v))

    print("\n=== by number of legal options ===")
    op = collections.defaultdict(list)
    for r in recs:
        b = "2-3" if r["n_opts"] <= 3 else ("4-6" if r["n_opts"] <= 6 else "7+")
        op[b].append(r)
    for k in sorted(op):
        report("opts " + k, op[k])


if __name__ == "__main__":
    main()
