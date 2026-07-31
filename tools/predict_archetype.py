"""Predict the OPPONENT's archetype from the cards they have revealed.

Go/no-go measurement: is the archetype knowable early enough to be worth feeding to the
LM? The answer is the ACCURACY-BY-TURN curve, not a single number -- an archetype known
only at turn 15 is useless, one known at turn 3 is worth having.

Method: Bayesian deck identification, NO training. We know all 60 decklists exactly, so
the likelihood is analytic:

    log P(deck d | observed) = log prior(d) + SUM_c  k_c * log( (n_d(c) + eps) / Z )

with k_c = copies of card c seen, n_d(c) = copies deck d runs. Smoothing eps matters:
live opponents run tech cards outside our lists, and a hard zero would delete the
correct deck on one off-list card. Archetype posterior = sum over that archetype's decks,
which is why deck VARIANTS (6x mega_lucario, 4x alakazam, 3x dragapult) cost nothing --
they split the deck posterior but share the archetype, so the archetype marginal is sharp
even when the deck marginal is not.

Visible to us: the opponent's active, bench, discard, attached energy and tools, plus the
stadium. NOT their hand, deck or prizes.

Usage:
    python tools/predict_archetype.py --tar <raw.tar> --games 400
    python tools/predict_archetype.py --live logs_live --games 400
"""
import argparse, collections, glob, gzip, json, math, os, random, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import library                                        # noqa: E402
from cg.api import to_observation_class               # noqa: E402

EPS = 0.25          # smoothing: an off-list card must not zero out a deck


def load_fleet():
    """-> {deck_name: (archetype, Counter(card_id -> copies))}"""
    tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    fleet = {}
    for name, cfg in tun.items():
        if not isinstance(cfg, dict) or not cfg.get("archetype"):
            continue
        try:
            d = library.read_deck(name)
        except Exception:
            continue
        if len(d) == 60:
            fleet[name] = (cfg["archetype"], collections.Counter(d))
    return fleet


class ArchetypePredictor:
    def __init__(self, fleet, prior=None):
        self.fleet = fleet
        self.names = sorted(fleet)
        self.arch = {n: fleet[n][0] for n in self.names}
        cards = set()
        for _a, c in fleet.values():
            cards |= set(c)
        self.Z = 60.0 + EPS * len(cards)
        # log-likelihood table: card -> {deck: log P(one copy | deck)}
        self.ll = {}
        for c in cards:
            self.ll[c] = {n: math.log((fleet[n][1].get(c, 0) + EPS) / self.Z)
                          for n in self.names}
        self.miss = math.log(EPS / self.Z)      # card no deck in the fleet runs
        self.log_prior = {n: math.log((prior or {}).get(n, 1.0 / len(self.names)))
                          for n in self.names}

    def posterior(self, observed):
        """observed: Counter(card_id -> copies seen). -> (deck_post, arch_post)"""
        s = dict(self.log_prior)
        for c, k in observed.items():
            row = self.ll.get(c)
            for n in self.names:
                s[n] += k * (row[n] if row else self.miss)
        m = max(s.values())
        e = {n: math.exp(v - m) for n, v in s.items()}
        tot = sum(e.values()) or 1.0
        dpost = {n: v / tot for n, v in e.items()}
        apost = collections.defaultdict(float)
        for n, v in dpost.items():
            apost[self.arch[n]] += v
        return dpost, dict(apost)


def observed_cards(st, opp):
    """Every opponent card WE can see: board bodies, their energy and tools, discard."""
    c = collections.Counter()
    ps = st.players[opp]
    for pk in list(ps.active or []) + list(ps.bench or []):
        if pk is None:
            continue
        if getattr(pk, "id", None) is not None:
            c[pk.id] += 1
        for e in (getattr(pk, "energyCards", None) or []):
            if getattr(e, "id", None) is not None:
                c[e.id] += 1
        for t in (getattr(pk, "tools", None) or []):
            if getattr(t, "id", None) is not None:
                c[t.id] += 1
        pre = getattr(pk, "preEvolution", None) or []
        for q in pre:
            if getattr(q, "id", None) is not None:
                c[q.id] += 1
    for d in (ps.discard or []):
        if getattr(d, "id", None) is not None:
            c[d.id] += 1
    return c


def selfplay_games(tar_path, n_games, seed=0):
    tf = tarfile.open(tar_path)
    names = [m.name for m in tf.getmembers() if m.name.endswith(".jsonl.gz")]
    random.Random(seed).shuffle(names)
    got = 0
    for nm in names:
        header, steps = None, []
        for line in gzip.open(tf.extractfile(nm), "rt"):
            r = json.loads(line)
            if r.get("kind") == "game":
                if header is not None and steps:
                    yield header.get("agents") or {}, steps
                    got += 1
                    if got >= n_games:
                        return
                header, steps = r, []
            elif r.get("kind") == "step":
                steps.append(r)
        if header is not None and steps:
            yield header.get("agents") or {}, steps
            got += 1
            if got >= n_games:
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", default="data/kaggle_out/v24_full/selfplay_v24_full_raw.tar")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--live", action="store_true", help="validate on logs_live instead")
    args = ap.parse_args()

    fleet = load_fleet()
    pred = ArchetypePredictor(fleet)
    base = collections.Counter(a for a, _ in fleet.values())
    top_arch, top_n = base.most_common(1)[0]
    print(f"fleet {len(fleet)} decks, {len(base)} archetypes {dict(base)}")
    print(f"majority-class baseline: {top_arch} = {100*top_n/len(fleet):.1f}%\n")

    if args.live:
        live_eval(fleet, pred)
        return

    by_turn = collections.defaultdict(lambda: [0, 0, 0])   # turn -> [n, arch_ok, deck_ok]
    conf = collections.defaultdict(lambda: [0, 0])
    for agents, steps in selfplay_games(args.tar, args.games):
        truth = {int(k): v for k, v in agents.items()}
        if not all(v in fleet for v in truth.values()):
            continue
        seen_turn = set()
        for s in steps:
            try:
                st = to_observation_class(s["obs"]).current
            except Exception:
                continue
            if not st or len(st.players or []) != 2:
                continue
            t = int(getattr(st, "turn", 0) or 0)
            for me in (0, 1):
                opp = 1 - me
                key = (t, me)
                if key in seen_turn:            # one sample per (turn, side)
                    continue
                seen_turn.add(key)
                obs_c = observed_cards(st, opp)
                if not obs_c:
                    continue
                dpost, apost = pred.posterior(obs_c)
                pa = max(apost, key=apost.get)
                pd = max(dpost, key=dpost.get)
                ta = fleet[truth[opp]][0]
                b = by_turn[min(t, 20)]
                b[0] += 1
                b[1] += (pa == ta)
                b[2] += (pd == truth[opp])
                c = conf[round(apost[pa], 1)]
                c[0] += 1; c[1] += (pa == ta)

    print(f"{'turn':>5}{'n':>8}{'archetype acc':>15}{'exact deck acc':>16}")
    tot = [0, 0, 0]
    for t in sorted(by_turn):
        n, a, d = by_turn[t]
        tot[0] += n; tot[1] += a; tot[2] += d
        print(f"{t:>5}{n:>8}{100*a/n:14.1f}%{100*d/n:15.1f}%")
    if tot[0]:
        print(f"{'ALL':>5}{tot[0]:>8}{100*tot[1]/tot[0]:14.1f}%{100*tot[2]/tot[0]:15.1f}%")
    print("\ncalibration (predicted-archetype probability -> actual accuracy)")
    for p in sorted(conf):
        n, a = conf[p]
        if n >= 30:
            print(f"   p~{p:.1f} : {100*a/n:5.1f}%  (n={n})")




# --------------------------------------------------------------------------- #
#  LIVE validation: real ladder opponents, many running decks OUTSIDE the fleet
# --------------------------------------------------------------------------- #
def live_eval(fleet, pred, limit=0):
    """The self-play numbers are the EASY case (the opponent is always one of our 60).
    Live, the real questions are (a) do we still identify decks we do know, and (b) when
    the deck is alien, does confidence collapse -- or do we confidently pick a wrong
    archetype? (b) is the dangerous failure, so it is measured directly against the
    fraction of the opponent's revealed cards that no deck in the fleet runs."""
    pool = set()
    for _a, c in fleet.values():
        pool |= set(c)
    files = sorted(glob.glob(os.path.join(ROOT, "logs_live", "*", "*.json")))
    if limit:
        files = files[:limit]
    known_hits = [0, 0]
    side_ok = [0, 0]
    rows = []
    for f in files:
        base = os.path.basename(f)[:-5]
        if "_vs_" not in base:
            continue
        a, b = base.split("_vs_", 1)
        labels = [a.split("-", 1)[-1].lower(), b.split("-", 1)[-1].lower()]
        try:
            recs = json.load(open(f))
        except Exception:
            continue
        last = None
        for r in recs:
            cur = r.get("current")
            if not cur:
                continue
            try:
                st = to_observation_class(r).current
            except Exception:
                continue
            if not st or len(st.players or []) != 2:
                continue
            yi = getattr(st, "yourIndex", None)
            if yi is None:
                continue
            opp = 1 - yi
            oc = observed_cards(st, opp)
            if sum(oc.values()) < 3:
                continue
            last = (oc, opp, labels[yi] in fleet)
        if last is None:
            continue
        oc, opp, ours_known = last
        side_ok[0] += 1; side_ok[1] += ours_known
        alien = sum(k for c, k in oc.items() if c not in pool)
        rate = alien / max(1, sum(oc.values()))
        dpost, apost = pred.posterior(oc)
        pa = max(apost, key=apost.get)
        lab = labels[opp]
        rows.append((rate, apost[pa], pa, lab))
        if lab in fleet:
            known_hits[0] += 1
            known_hits[1] += (pa == fleet[lab][0])

    print(f"\nLIVE: {len(rows)} games "
          f"(side-assignment sanity: our own label is a fleet deck in "
          f"{100*side_ok[1]/max(1,side_ok[0]):.0f}% of files)")
    if known_hits[0]:
        print(f"  opponent label IS a fleet deck: archetype accuracy "
              f"{100*known_hits[1]/known_hits[0]:.1f}%  (n={known_hits[0]})")
    print(f"\n  {'alien-card rate':>16}{'n':>7}{'mean confidence':>17}")
    buckets = collections.defaultdict(list)
    for rate, cf, _pa, _lab in rows:
        buckets[min(int(rate * 10) / 10, 0.5)].append(cf)
    for k in sorted(buckets):
        v = buckets[k]
        print(f"  {k:>15.0%}{len(v):>7}{sum(v)/len(v):16.2f}")
    unknown = [(r, c, p, l) for r, c, p, l in rows if l not in fleet]
    if unknown:
        hi = [x for x in unknown if x[1] > 0.9]
        print(f"\n  opponent NOT a fleet deck: {len(unknown)} games, "
              f"{len(hi)} ({100*len(hi)/len(unknown):.0f}%) predicted with confidence >0.9")
        c = collections.Counter(l for _r, _c, _p, l in unknown)
        print("  most common unknown labels:", dict(c.most_common(6)))


if __name__ == "__main__":
    main()
