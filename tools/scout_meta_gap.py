"""Scout the leaderboard top-N and measure the GAP against our local 60-deck fleet.

Two questions this answers:
  1. Are competitors running deck TYPES we do not have? (archetype coverage)
  2. Are they running CARDS outside our fleet's pool? -- this is the assumption the
     Bayesian archetype predictor rests on (tools/predict_archetype.py measured 95% of
     live games with ZERO out-of-fleet cards, but on OUR OWN older logs).

Per team it reports: reconstruction completeness, the out-of-fleet card rate, the
nearest fleet deck by card overlap, and the archetype our predictor assigns.

Usage:
    PYTHONPATH=.:cg-lib python tools/scout_meta_gap.py 100 3
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scout_decks import _api, scout_team, decklist, COMP     # noqa: E402
from predict_archetype import load_fleet, ArchetypePredictor  # noqa: E402
from agents._engine import _CARDS                             # noqa: E402


def _name(cid):
    c = _CARDS.get(cid)
    return getattr(c, "name", None) or str(cid)


def leaderboard(n):
    """The kaggle client PRINTS "Next Page Token = ..." instead of exposing it on the
    response or the api object, so the only way to page is to capture stdout."""
    import io, re
    from contextlib import redirect_stdout
    api = _api()
    rows, token = [], None
    while len(rows) < n:
        buf = io.StringIO()
        with redirect_stdout(buf):
            page = list(api.competition_leaderboard_view(
                COMP, page_size=50, page_token=token) or [])
        if not page:
            break
        rows.extend(page)
        m = re.search(r"Next Page Token = (\S+)", buf.getvalue())
        if not m:
            break
        token = m.group(1)
    return rows[:n]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    fleet = load_fleet()
    pred = ArchetypePredictor(fleet)
    pool = set()
    for _a, c in fleet.values():
        pool |= set(c)
    print(f"fleet {len(fleet)} decks, {len(pool)} distinct cards in pool", flush=True)

    out = []
    alien_counter = collections.Counter()
    for rank, r in enumerate(leaderboard(n), 1):
        tid = getattr(r, "team_id", None)
        tname = getattr(r, "team_name", "?")
        score = getattr(r, "score", None)
        try:
            res = scout_team(tid, k)
        except Exception as e:
            print(f"{rank:>4} {tname[:22]:22} scout failed: {str(e)[:50]}", flush=True)
            continue
        if not res or not res.get("seen"):
            print(f"{rank:>4} {tname[:22]:22} no replays", flush=True)
            continue
        own = res["seen"].get(tname)
        if own is None:
            own = max(res["seen"].values(), key=lambda b: sum(b.values()))
        cnt = decklist(own)
        total = sum(cnt.values())
        if total < 30:                      # too partial to judge
            print(f"{rank:>4} {tname[:22]:22} only {total} cards recovered", flush=True)
            continue
        alien = {c: k2 for c, k2 in cnt.items() if c not in pool}
        for c, k2 in alien.items():
            alien_counter[c] += 1
        _dpost, apost = pred.posterior(cnt)
        arch = max(apost, key=apost.get)
        # nearest fleet deck by multiset overlap
        best, bestov = None, 0.0
        for nm, (_a, fc) in fleet.items():
            ov = sum(min(cnt[c], fc[c]) for c in set(cnt) & set(fc))
            if ov > bestov:
                best, bestov = nm, ov
        rec = {"rank": rank, "team": tname, "score": score, "cards": total,
               "alien_cards": sum(alien.values()),
               "alien_rate": sum(alien.values()) / total,
               "archetype": arch, "arch_p": round(apost[arch], 3),
               "nearest": best, "overlap": bestov / 60.0,
               "alien_list": sorted(_name(c) for c in alien)}
        out.append(rec)
        flag = "  <<< NOVEL" if rec["alien_rate"] > 0.10 or rec["overlap"] < 0.55 else ""
        print(f"{rank:>4} {tname[:22]:22} {str(score)[:6]:>7} {total:>3}c "
              f"alien {rec['alien_cards']:>2} ({100*rec['alien_rate']:4.1f}%) "
              f"{arch:9} p={rec['arch_p']:.2f} ~{(best or '?')[:20]:20} "
              f"ov={rec['overlap']:.2f}{flag}", flush=True)

    os.makedirs(os.path.join(ROOT, "evaluations"), exist_ok=True)
    dst = os.path.join(ROOT, "evaluations", f"meta_gap_top{n}.json")
    json.dump(out, open(dst, "w"), indent=1)

    print(f"\n==== {len(out)} teams reconstructed ====")
    if not out:
        return
    ac = collections.Counter(r["archetype"] for r in out)
    print("archetype mix on the leaderboard:", dict(ac.most_common()))
    print("our fleet's archetype mix:      ",
          dict(collections.Counter(a for a, _ in fleet.values()).most_common()))
    zero = sum(1 for r in out if r["alien_cards"] == 0)
    print(f"\nteams with ZERO out-of-fleet cards: {zero}/{len(out)} = {100*zero/len(out):.0f}%")
    print(f"mean out-of-fleet rate: {100*sum(r['alien_rate'] for r in out)/len(out):.1f}%")
    novel = [r for r in out if r["alien_rate"] > 0.10 or r["overlap"] < 0.55]
    print(f"flagged NOVEL (>10% alien cards or <0.55 overlap): {len(novel)}")
    for r in novel[:15]:
        print(f"   #{r['rank']} {r['team'][:20]:20} ov={r['overlap']:.2f} "
              f"alien={r['alien_cards']} {r['alien_list'][:6]}")
    if alien_counter:
        print("\nmost common out-of-fleet cards (teams running them):")
        for c, t in alien_counter.most_common(20):
            print(f"   {t:>3} teams  {_name(c)}")
    print(f"\nsaved -> {dst}")


if __name__ == "__main__":
    main()
