"""Fetch top-N leaderboard teams, reconstruct each deck, and classify the field.

Builds on tools/scout_decks.py: for each of the top-N teams we reconstruct a
(near-)60-card list from a few episode replays, then classify it against our
decks/*.csv by best multiset overlap. Prints a distribution histogram plus the
signature Pokemon for any team that doesn't match a known archetype.

Usage:
    PYTHONPATH=cg-lib python tools/leaderboard_distribution.py [N] [K]
"""
import os
import sys
import csv
import glob
import json
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents._engine import _CARDS  # noqa: E402
from tools.scout_decks import scout_team, decklist, COMP, _api  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# card ids we ignore for archetype matching (basic energy + ubiquitous staples)
BASIC_ENERGY = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}


def _load_our_decks():
    out = {}
    for path in glob.glob(os.path.join(ROOT, "decks", "*.csv")):
        name = os.path.splitext(os.path.basename(path))[0]
        cnt = defaultdict(int)
        with open(path) as f:
            for r in csv.reader(f):
                if not r or not r[0].strip().isdigit():
                    continue
                cid = int(r[0])
                n = int(r[1]) if len(r) > 1 and r[1].strip() else 1
                cnt[cid] += n
        out[name] = dict(cnt)
    return out


def _pokemon_ids(cnt):
    """Non-energy, non-trainer signature ids (POKEMON cards) sorted by count."""
    out = []
    for cid, n in cnt.items():
        c = _CARDS.get(cid)
        if c is None or cid in BASIC_ENERGY:
            continue
        # cardType: 0 == POKEMON in this dataset
        if getattr(c, "cardType", None) == 0:
            out.append((n, cid, c.name))
    return sorted(out, key=lambda x: -x[0])


def _is_pokemon(cid):
    c = _CARDS.get(cid)
    return c is not None and getattr(c, "cardType", None) == 0


def _overlap(a, b):
    """Archetype similarity = Pokemon-line multiset overlap (robust to partial
    trainer/energy reconstruction). Normalized by OUR deck's pokemon count so a
    scouted deck matches even when only a few episodes were seen."""
    pk_keys = {k for k in (set(a) | set(b)) if _is_pokemon(k)}
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in pk_keys)
    denom = max(1, sum(v for k, v in b.items() if _is_pokemon(k)))
    return inter / denom


def classify(cnt, our):
    best, score = None, 0.0
    for name, deck in our.items():
        s = _overlap(cnt, deck)
        if s > score:
            best, score = name, s
    return best, score


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    our = _load_our_decks()

    api = _api()

    def _page(tok):
        """Return (submissions, next_token) — the public wrapper drops the token."""
        from kagglesdk.competitions.types.competition_api_service import (
            ApiGetLeaderboardRequest,
        )
        with api.build_kaggle_client() as kc:
            req = ApiGetLeaderboardRequest()
            req.competition_name = COMP
            req.page_size = 50
            req.page_token = tok
            resp = kc.competitions.competition_api_client.get_leaderboard(req)
        return list(resp.submissions or []), (resp.next_page_token or None)

    rows = []
    token = None
    while len(rows) < n:
        page, token = _page(token)
        if not page:
            break
        rows.extend(page)
        if not token:
            break
    rows = rows[:n]
    print(f"leaderboard rows fetched: {len(rows)}", file=sys.stderr)

    results = []  # (rank, team_name, score, archetype, match, total, sig)
    for rank, r in enumerate(rows, 1):
        tid = getattr(r, "team_id", None)
        tname = getattr(r, "team_name", "?")
        lbscore = getattr(r, "score", None)
        try:
            res = scout_team(tid, k)
        except Exception as e:
            print(f"  #{rank} {tname}: scout failed: {e}", file=sys.stderr)
            results.append((rank, tname, lbscore, None, 0.0, 0, ""))
            continue
        own = None
        if res and res["seen"]:
            own = res["seen"].get(tname)
            if own is None:
                own = max(res["seen"].values(), key=lambda b: sum(b.values()))
        if not own:
            results.append((rank, tname, lbscore, None, 0.0, 0, ""))
            print(f"  #{rank} {tname}: no data", file=sys.stderr)
            continue
        own = dict(own)
        total = sum(own.values())
        arch, match = classify(own, our)
        pk = _pokemon_ids(own)
        sig = ", ".join(f"{nm}x{c}" for c, i, nm in pk[:4])
        if match >= 0.55:
            label = arch
        elif pk:
            label = "OTHER: " + pk[0][2]  # cluster unknowns by lead Pokemon
        else:
            label = None
        results.append((rank, tname, lbscore, label, match, total, sig))
        print(f"  #{rank} {tname}: {label or 'OTHER'} ({match:.0%}, {total}/60) [{sig}]",
              file=sys.stderr)

    # distribution
    dist = Counter(x[3] or "OTHER/unknown" for x in results)
    print("\n" + "=" * 60)
    print(f"DECK DISTRIBUTION — top {len(results)} leaderboard teams (K={k} eps each)")
    print("=" * 60)
    for name, c in dist.most_common():
        bar = "#" * c
        print(f"  {c:3d}  {name:24s} {bar}")

    # dump full detail as json for follow-up
    out_path = os.path.join(ROOT, "scratchpad_replays", "distribution.json")
    with open(out_path, "w") as f:
        json.dump([{"rank": r, "team": t, "lb_score": s, "archetype": a,
                    "match": m, "cards_seen": tot, "signature": sig}
                   for r, t, s, a, m, tot, sig in results], f, indent=2)
    print(f"\n(detail written to {out_path})")

    # OTHER breakdown by signature
    others = [x for x in results if x[3] is None or str(x[3]).startswith("OTHER")]
    if others:
        print(f"\nOTHER/unmatched ({len(others)}) — signature Pokemon:")
        for rank, tname, lbscore, _a, match, total, sig in others:
            print(f"  #{rank:3d} {tname:20s} best~{match:.0%} {total}/60  [{sig}]")


if __name__ == "__main__":
    main()
