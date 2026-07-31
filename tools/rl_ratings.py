"""Deck strength + difficulty-matched matchmaking for the RL curriculum (Stage A).

Seeds from the fleet round-robin (evaluations/roundrobin_*.json): per-deck overall
winrate = strength, and the pairwise matrix[pilot][opp] = pilot's winrate% vs opp.

WHY (the weak-deck learning problem): with terminal +1/-1 reward, a deck stuck at ~15%
vs the field produces almost only losses -- the RAE per-matchup baseline gives its rare
wins a big +advantage, but there are too FEW winning trajectories to teach good play (you
learn a skill from the CONTRAST between winning and losing lines, and 15% wins is thin
contrast). Stage-A matchmaking samples (pilot,opp) pairs biased toward an expected winrate
~50%, so EVERY deck -- strong or weak -- plays even, contrastive games and can climb its
own skill ceiling. Structurally-behind decks (fuel-bound: iono_bellibolt/zangoose/...) are
still matched to their weakest peers, but RL cannot fix a deck whose outcome is play-
independent -- that is a deck problem, not a pilot problem.

Falls back to a logistic on the strength gap when a matrix cell is missing.
"""
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path=None):
    """Return (strength: {deck: winrate%}, matrix: {pilot: {opp: winrate%}}).
    Empty dicts if no round-robin is found (callers then fall back to uniform)."""
    if path is None:
        cands = sorted(glob.glob(os.path.join(ROOT, "evaluations", "roundrobin_*.json")))
        path = cands[-1] if cands else None
    if not path or not os.path.exists(path):
        return {}, {}
    d = json.load(open(path))
    strength = {r["deck"]: float(r["winrate"]) for r in d.get("ranking", [])}
    matrix = d.get("matrix", {}) or {}
    return strength, matrix


def expected_wr(pilot, opp, strength, matrix):
    """Pilot's expected winrate% vs opp. Matrix cell first, then the symmetric cell
    (100 - opp-vs-pilot), then a logistic on the strength gap, then 50 as last resort."""
    row = matrix.get(pilot)
    if row and opp in row:
        return float(row[opp])
    orow = matrix.get(opp)
    if orow and pilot in orow:
        return 100.0 - float(orow[pilot])
    sp, so = strength.get(pilot), strength.get(opp)
    if sp is None or so is None:
        return 50.0
    return 100.0 / (1.0 + math.exp(-(sp - so) / 10.0))   # ~10 winrate-pts scale


def _wchoice(weights, rng):
    """Index ~ weights (assumes sum>0)."""
    r = rng.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def _wsample(pool, wmap, k, rng, exclude=None):
    """k DISTINCT items from pool ~ wmap (weighted, without replacement); skips `exclude`."""
    items = [x for x in pool if x != exclude]
    w = [max(0.0, (wmap or {}).get(x, 1.0)) for x in items]
    k = min(k, len(items))
    out = []
    for _ in range(k):
        if sum(w) <= 0:
            break
        i = _wchoice(w, rng)
        out.append(items[i])
        w[i] = 0.0
    return out


def sample_pairs(pilots, opps, n, rng, pilot_w=None, opp_w=None,
                 match=False, band=12.0, strength=None, matrix=None, target_wr=50.0):
    """Sample up to n DISTINCT (pilot, opp) pairs (pilot != opp). Two modes:

    match=True  (Stage A) -- weight(p,o) = pilot_w[p]*opp_w[o]*exp(-((wr-target_wr)/band)^2),
        drawn without replacement, so matchups near `target_wr` dominate. band = the
        winrate-pt std (~12 => within ~+-20pts of the target).

        ``target_wr`` is expressed in the units the ratings are IN: `expected_wr` comes from
        engine_v2-vs-engine_v2 play, so it is the win rate when BOTH sides are the engine.
        When the LEARNING pilot is weaker than the engine, an even fight for it needs a
        matchup the engine would WIN -- so the target moves UP, not down. Measured
        2026-07-28, the v37 reranker trails engine_v2 by ~12pt piloting the same deck
        (crustle_stall -6.1, mega_lucario -15.8, alakazam -16.6), hence rl_config's
        MATCH_TARGET_WR = 62. Setting it to 38 instead would pick matchups the deck already
        loses and hand the LM ~26% games -- the opposite of contrastive.
    match=False (Stage B/C) -- PER-PILOT QUOTA: allocate the n draws across pilots in exact
        proportion to pilot_w (largest-remainder), then draw each pilot's distinct opponents
        ~ opp_w. This makes the PILOT marginal exactly the target focus split (a plain weighted
        without-replacement draw under-hits it, because low-count focus pilots deplete). A
        pilot capped at its available-opp count (e.g. Stage C's single target => all opps).

    n<=0 or n>=#pairs => all pairs. pilot_w/opp_w default to 1.0 per missing key."""
    if not pilots or not opps:
        return []
    total = sum(1 for p in pilots for o in opps if p != o)
    target = total if (not n or n >= total) else n

    if match:
        pairs = [(p, o) for p in pilots for o in opps if p != o]
        pw, ow, st, mx = (pilot_w or {}), (opp_w or {}), (strength or {}), (matrix or {})
        w = [pw.get(p, 1.0) * ow.get(o, 1.0)
             * math.exp(-(((expected_wr(p, o, st, mx) - target_wr) / band) ** 2))
             for p, o in pairs]
        if sum(w) <= 0:
            w = [1.0] * len(pairs)
        chosen = []
        for _ in range(min(target, len(pairs))):
            if sum(w) <= 0:
                break
            i = _wchoice(w, rng)
            chosen.append(pairs[i]); w[i] = 0.0
        return chosen

    # weighted mode: exact pilot marginal via quota
    pw = {p: max(0.0, (pilot_w or {}).get(p, 1.0)) for p in pilots}
    tot = sum(pw.values()) or 1.0
    quota = {p: target * pw[p] / tot for p in pilots}
    base = {p: int(quota[p]) for p in pilots}
    rem = target - sum(base.values())
    for p in sorted(pilots, key=lambda p: quota[p] - int(quota[p]), reverse=True)[:max(0, rem)]:
        base[p] += 1
    out = []
    for p in pilots:
        for o in _wsample(opps, opp_w, base[p], rng, exclude=p):
            out.append((p, o))
    rng.shuffle(out)
    return out


if __name__ == "__main__":   # quick sanity / inspection
    import sys
    st, mx = load()
    print(f"decks with strength: {len(st)}; matrix rows: {len(mx)}")
    if st:
        order = sorted(st, key=st.get, reverse=True)
        print("strongest:", [(d, st[d]) for d in order[:3]])
        print("weakest:  ", [(d, st[d]) for d in order[-3:]])
        # show a weak deck's ~50% matched opponents
        weak = order[-1]
        cand = sorted(((expected_wr(weak, o, st, mx), o) for o in st if o != weak),
                      key=lambda x: abs(x[0] - 50))
        print(f"{weak} best-matched opps:", [(o, round(wr, 1)) for wr, o in cand[:5]])
