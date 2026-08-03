#!/usr/bin/env python3
"""Training records for energy ATTACH whose label is a measured win rate, not engine_v2's opinion.

WHY NOT MORE IMITATION DATA. `tools/attach_value.py` measured 2,519 attach decisions with
counterfactual playouts: engine_v2's target pick is worth +0.0586 +/- 0.0058 over the
alternatives, and a perfect pick is worth +0.1221 +/- 0.0074. So imitation reaches 48% of what
is on the table and no amount of extra imitation data reaches the rest -- the teacher does not
know it. The other 52% is only visible to playouts.

WHERE THE BUDGET GOES. Headroom is not spread evenly, and a uniform sample puts two thirds of
it in the cheapest cell (6 prizes remaining is 66% of attach decisions and the smallest
headroom). Measured headroom:

    targets  k=2 +0.026 (ns) | k=3 +0.040 | k=4 +0.071 | k=5 +0.104 | k>=6 +0.091
    prizes   6 +0.055 | 5 +0.078 | 4 +0.087 | 3 +0.113 | 2 +0.047 | 1 +0.058

`REPLACE_DECKS` are the six where engine_v2's attach edge is NEGATIVE -- imitating it there is
actively harmful -- so they are sampled at full rate regardless of cell.

ATTACH-OR-NOT IS PART OF THE QUESTION (2026-08-02). The first build branched attach targets
only. Measured on the resulting file, the reranker scores a NON-attach above every attach on
42.4% of decisions that have a clear best attach target -- a bigger failure than picking the
wrong target, and one a which-target-only file cannot teach. Every accepted decision now also
branches up to `--nonattach-k` of the other options, so `chosen` can come out as `end` or a
`play`, and the engine's own pick no longer has to be an attach for the decision to qualify.

THE LABEL IS SPLIT-SAMPLE. The argmax is taken on one half of the playouts and required to
still lead on the other. Taking both from the same playouts is a winner's curse: max() over
noisy estimates is biased upward, which is what produced a meaningless "chosen was best 72.4%"
in `engine-native-search-api`. A decision is also required to beat a PERMUTATION null (pool
every playout, redeal into groups of the same sizes) before it is written at all, which is how
the ~53% of attach decisions that are value-neutral get dropped rather than taught as rules.

That filter is deliberate selection: the file is NOT a sample of attach decisions, it is a
sample of attach decisions where the target demonstrably matters. It is meant to be MIXED with
the imitation pool, not to replace it.

Output is the reranker schema plus `qvals`, one entry per candidate (null where the candidate
was not branched -- every non-attach option, which stays in the list so the ordinary listwise
term still trains against it).
"""
import argparse
import collections
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# measured headroom per cell; used as a relative sampling weight, normalised to its own max
W_K = {1: 0.026, 2: 0.026, 3: 0.040, 4: 0.071, 5: 0.104}
W_K_HI = 0.091                                   # k >= 6
W_PZ = {0: 0.055, 1: 0.058, 2: 0.047, 3: 0.113, 4: 0.087, 5: 0.078, 6: 0.055}
_WMAX = 0.104 * 0.113
MIN_W = 0.35                                     # floor, see cell_weight

REPLACE_DECKS = ("mega_abomasnow_sample", "iono_bellibolt", "trevenant_control",
                 "ceruledge", "cubchoo_control", "mega_feraligatr")


def cell_weight(k, prizes, deck, uniform=False):
    """Acceptance probability for one branch point, in [0, 1].

    `uniform` turns the targeting off, for building a HELD-OUT set: a training file should be
    concentrated where the headroom is, but a file used to measure needs to represent the
    population it claims to measure.

    The weights come from the WHICH-TARGET headroom, and a floor is applied because every
    accepted decision now also carries the ATTACH-OR-NOT question, whose headroom has never
    been measured. Without the floor the cells where target choice barely matters -- k=2, six
    prizes, which is where most decisions live -- would be sampled at 2% and the new signal
    would be starved in exactly the states the pilot spends most of its time in. k=1 has no
    measurement at all (it is not a target choice), so it lands on the floor by construction.
    """
    if uniform or deck in REPLACE_DECKS:
        return 1.0
    w = (W_K.get(k, W_K_HI) * W_PZ.get(prizes, 0.055)) / _WMAX
    return max(w, MIN_W)


def permutation_null(per, rng, draws=8):
    """Mean max-min spread when the candidates are known to be worth the same.

    Redeals the SAME playouts into groups of the SAME sizes, so it is a null at matched sample
    size. An earlier version compared the two halves of each candidate instead and ran at half
    the sample size of the statistic it judged, which called real signal noise.
    """
    pool = [v for q in per for v in q]
    sizes = [len(q) for q in per]
    out = []
    for _ in range(draws):
        rng.shuffle(pool)
        off, ms = 0, []
        for s in sizes:
            ms.append(sum(pool[off:off + s]) / s)
            off += s
        out.append(max(ms) - min(ms))
    return sum(out) / len(out)


def label(per, rng):
    """-> (index of the best target, per-candidate mean Q) or (None, None) to drop.

    Four gates, each removing a different way to be fooled:
      1. the full-sample spread must beat the permutation null   (is there a decision at all)
      2. the half-A argmax must still lead on half B             (winner's curse)
      3. its half-B lead must be positive                        (sign, not just order)
      4. it must also be the UNIQUE full-sample argmax           (internal consistency)

    Gate 4 exists because the record carries two things that must not disagree: `chosen`, which
    the listwise cross-entropy trains toward, and `qvals`, which the value-margin term reads.
    `chosen` comes from the split (protected against the curse) and `qvals` are full-sample
    means, so they CAN differ -- measured at 6.1% of records on the first build, plus ~10% more
    where the best ties with a runner-up. On those the two loss terms pull in opposite
    directions. A disagreement is also a close call by construction, which is the case this
    file is meant to exclude anyway.
    """
    if any(len(q) < 4 for q in per):
        return None, None
    means = [sum(q) / len(q) for q in per]
    if max(means) - min(means) <= permutation_null(per, rng):
        return None, None
    a, b = [], []
    for q in per:
        h = len(q) // 2
        a.append(sum(q[:h]) / h)
        b.append(sum(q[h:]) / len(q[h:]))
    pick = max(range(len(a)), key=lambda i: a[i])
    others = [x for i, x in enumerate(b) if i != pick]
    if not others or b[pick] - sum(others) / len(others) <= 0:
        return None, None
    if any(means[i] >= means[pick] for i in range(len(means)) if i != pick):
        return None, None
    return pick, means


def _one_deck(job):
    deck, games, playouts, seed, target, fmt, budget, uniform, nonattach_k, max_attach = job
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.actions import encode_option
    from lm.action_token import dedup_options
    from lm.agent import make_lm_agent
    from lm.serialize import serialize_stateless
    import rl_branch

    rng = random.Random(seed)
    ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    prof = tuning.get(deck, {})
    me = make_lm_agent(ids, prof, model=None)
    opp = make_lm_agent(ids, prof, model=None)
    out, st = [], collections.Counter()
    t0 = time.time()
    for _g in range(games):
        # A per-deck wall-clock cap. Playout cost varies 10x across decks -- `slowking` needed
        # 27 s per branch point against 2 s for `zangoose` -- so without it one deck decides the
        # wall time of the whole fleet and the cheap decks sit idle having finished long ago.
        if len(out) >= target or time.time() - t0 > budget:
            break
        obs, _ = battle_start(ids, ids)
        if obs is None:
            continue
        try:
            for _ in range(4000):
                cur = obs.get("current") or {}
                if cur.get("result", -1) != -1 or obs.get("select") is None:
                    break
                yi = cur.get("yourIndex", 0)
                opts = (obs.get("select") or {}).get("option") or []
                pick = me(obs) if yi == 0 else opp(obs)
                if yi != 0 or len(opts) < 2 or not pick or pick[0] >= len(opts) \
                        or len(out) >= target or time.time() - t0 > budget:
                    obs = battle_select(pick)
                    continue
                raw = [encode_option(o, obs) for o in opts]
                cands, pos, keys = dedup_options(raw, obs)
                att = [i for i, t in enumerate(cands) if t.startswith("attach:")]
                oth = [i for i in range(len(cands)) if i not in att]
                lab = {keys[p]: n for n, p in enumerate(pos)}.get(keys[pick[0]])
                pz = len(cur["players"][0].get("prize") or [])
                # ATTACH-OR-NOT, not only WHICH-TARGET. The first build branched attach
                # candidates only, so it could not teach the model's single largest measured
                # failure: on 42.4% of decisions with a clear best attach target it scores a
                # NON-attach above every attach. A file that never values `end` or `play`
                # against an attach cannot correct that. The engine's own pick is no longer
                # required to be an attach either -- "engine declined to attach and the
                # playouts agree" is exactly as instructive as the reverse.
                if not att or len(cands) < 2:
                    obs = battle_select(pick)
                    continue
                st["seen"] += 1
                if rng.random() > cell_weight(len(att), pz, deck, uniform):
                    st["skip_cell"] += 1
                    obs = battle_select(pick)
                    continue
                extra = oth if len(oth) <= nonattach_k else rng.sample(oth, nonattach_k)
                branch = att[:max_attach] + extra
                sels = [[pos[i]] for i in branch]
                per = [[] for _ in sels]
                for _s in range(playouts):
                    q = rl_branch.branch_values(obs, ids, ids, 0, sels, me, opp,
                                                n_playouts=1, rng=rng)
                    for i, v in enumerate(q):
                        if v is not None:
                            per[i].append(v)
                best, means = label(per, rng)
                if best is None:
                    st["drop_neutral"] += 1
                else:
                    qv = [None] * len(cands)
                    for i, c in enumerate(branch):
                        qv[c] = means[i]
                    out.append({
                        "state": serialize_stateless(obs, deck_ids=ids, deck_name=deck, **fmt),
                        "candidates": cands, "chosen": branch[best], "qvals": qv,
                        "n_attach": len(att), "attach_won": cands[branch[best]].startswith("attach:"),
                        "engine_chosen": lab, "kind": "main", "deck": deck, "opp": deck,
                        "valued": True})
                    st["kept"] += 1
                    # only countable when the engine's pick was BRANCHED: it may be a
                    # non-attach that `extra` did not sample, and then agreement is impossible
                    # by construction -- which read as 8.0%, below the ~14% chance level
                    if lab in branch:
                        st["engine_branched"] += 1
                        st["engine_agreed"] += (branch[best] == lab)
                    st["attach_won"] += cands[branch[best]].startswith("attach:")
                    st["engine_attached"] += (lab in att if lab is not None else 0)
                obs = battle_select(pick)
        finally:
            battle_finish()
    return deck, out, dict(st), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=400, help="cap per deck; the record target "
                                                          "usually binds first")
    ap.add_argument("--playouts", type=int, default=16)
    ap.add_argument("--per-deck", type=int, default=300, help="record target per deck")
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--nonattach-k", type=int, default=3,
                    help="how many NON-attach options to branch alongside the attach targets, "
                         "so the record can say whether attaching was right at all. 0 "
                         "reproduces the which-target-only first build")
    ap.add_argument("--max-attach", type=int, default=6,
                    help="cap on branched attach targets, to bound cost when the menu is wide")
    ap.add_argument("--seed", type=int, default=2000,
                    help="change it to build a set disjoint from an earlier one")
    ap.add_argument("--uniform", action="store_true",
                    help="no headroom targeting -- for a held-out set that must represent the "
                         "population, not the cells worth training on")
    ap.add_argument("--deck-seconds", type=float, default=3600,
                    help="wall-clock cap per deck; playout cost varies 10x across decks")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import library
    from tools import rl_config
    fmt = dict(rl_config.PROMPT_FMT)
    decks = ([d.strip() for d in a.decks.split(",") if d.strip()]
             or sorted(library.list_decks()))
    jobs = [(d, a.games, a.playouts, a.seed + i, a.per_deck, fmt, a.deck_seconds, a.uniform,
             a.nonattach_k, a.max_attach) for i, d in enumerate(decks)]
    tot = collections.Counter()
    n = 0
    t0 = time.time()
    with gzip.open(a.out, "wt") as f:
        with mp.Pool(min(a.workers, len(jobs))) as pool:
            for deck, rows, st, dt in pool.imap_unordered(_one_deck, jobs):
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += len(rows)
                tot.update(st)
                print("  %-24s kept %4d / seen %5d  %.0fs  (total %d)"
                      % (deck, st.get("kept", 0), st.get("seen", 0), dt, n), flush=True)
    print("\nwritten %d records to %s in %.1f min" % (n, a.out, (time.time() - t0) / 60))
    print("  branch points seen        %d" % tot.get("seen", 0))
    print("  skipped by cell weight    %d (%.1f%%)"
          % (tot.get("skip_cell", 0), 100.0 * tot.get("skip_cell", 0) / max(1, tot["seen"])))
    print("  dropped as value-neutral  %d (%.1f%% of branched)"
          % (tot.get("drop_neutral", 0),
             100.0 * tot.get("drop_neutral", 0)
             / max(1, tot["seen"] - tot.get("skip_cell", 0))))
    print("  playouts chose an ATTACH  %d (%.1f%% of kept)"
          % (tot.get("attach_won", 0),
             100.0 * tot.get("attach_won", 0) / max(1, tot.get("kept", 0))))
    print("  engine chose an ATTACH    %d (%.1f%% of kept)"
          % (tot.get("engine_attached", 0),
             100.0 * tot.get("engine_attached", 0) / max(1, tot.get("kept", 0))))
    print("  label == engine's pick    %d (%.1f%% of the %d where the engine's own pick "
          "was branched)"
          % (tot.get("engine_agreed", 0),
             100.0 * tot.get("engine_agreed", 0) / max(1, tot.get("engine_branched", 0)),
             tot.get("engine_branched", 0)))


if __name__ == "__main__":
    main()
