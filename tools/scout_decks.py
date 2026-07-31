"""Reconstruct competitors' decks from the Kaggle leaderboard match history.

Kaggle sim competitions expose each episode's full replay via the API. A replay's
per-step observation reveals, from a team's OWN perspective, its complete private
state (hand / deck-as-drawn / discard / prizes). Unioning the distinct card serials
a team reveals across several games reconstructs (nearly) their full 60-card deck.

Pipeline:  team_id -> latest public submission -> list episodes -> download K
replays -> union distinct serials per (own) side -> card-id multiset.

Usage:
    PYTHONPATH=cg-lib python tools/scout_decks.py <team_id> [K]
    PYTHONPATH=cg-lib python tools/scout_decks.py --leaderboard <N> [K]   # top-N teams

Replays are cached under scratchpad/replays/ so reruns are free.
"""
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents._engine import _CARDS  # noqa: E402

COMP = "pokemon-tcg-ai-battle"
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scratchpad_replays")
os.makedirs(CACHE, exist_ok=True)


def _api():
    import kaggle
    return kaggle.api


def _cards_of_pokemon(pk):
    out = []
    if not pk:
        return out
    out.append((pk.get("serial"), pk.get("id")))
    for pe in pk.get("preEvolution") or []:
        out.append((pe.get("serial"), pe.get("id")))
    for ec in pk.get("energyCards") or []:
        out.append((ec.get("serial"), ec.get("id")))
    for t in pk.get("tools") or []:
        out.append((t.get("serial"), t.get("id")))
    return out


def _game_decklists(replay):
    """Per-game card-id counts each team reveals from its OWN perspective.

    serials are unique WITHIN a game, so counting distinct serials -> id gives a
    lower bound on each card's copies for that single game. Returns {name: {id: n}}.
    """
    teams = replay["info"]["TeamNames"]
    seen = {}  # name -> {serial: id}
    for step in replay["steps"]:
        for pidx, entry in enumerate(step):
            cur = entry["observation"].get("current")
            if not cur:
                continue
            yi = cur.get("yourIndex", pidx)
            players = cur.get("players") or []
            if yi >= len(players):
                continue
            me = players[yi]
            bucket = seen.setdefault(teams[pidx], {})
            for pk in (me.get("active") or []):
                for s, i in _cards_of_pokemon(pk):
                    if s is not None:
                        bucket[s] = i
            for pk in (me.get("bench") or []):
                for s, i in _cards_of_pokemon(pk):
                    if s is not None:
                        bucket[s] = i
            for key in ("hand", "discard", "prize"):
                for c in (me.get(key) or []):
                    if c and c.get("serial") is not None:
                        bucket[c["serial"]] = c.get("id")
    out = {}
    for name, ser in seen.items():
        cnt = defaultdict(int)
        for _s, i in ser.items():
            cnt[i] += 1
        out[name] = dict(cnt)
    return out


def _harvest(replay, best_by_team):
    """Merge a game's decklists into best_by_team[name][id] via per-id MAX across games."""
    for name, cnt in _game_decklists(replay).items():
        bucket = best_by_team.setdefault(name, defaultdict(int))
        for i, n in cnt.items():
            if n > bucket[i]:
                bucket[i] = n


def _download_episode(api, eid):
    path = os.path.join(CACHE, f"episode-{eid}-replay.json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        api.competition_episode_replay(eid, path=CACHE, quiet=True)
    with open(path) as fh:
        return json.load(fh)


def scout_team(team_id, k=5):
    api = _api()
    subs = list(api.competition_team_submissions(team_id))
    if not subs:
        return None
    sub_id = subs[0].id
    eps = list(api.competition_list_episodes(sub_id))
    eids = [e.id for e in eps][:k]
    best = {}
    used = 0
    for eid in eids:
        try:
            rep = _download_episode(api, eid)
        except Exception as e:
            print(f"    (episode {eid} failed: {e})", file=sys.stderr)
            continue
        _harvest(rep, best)
        used += 1
    return {"team_id": team_id, "submission_id": sub_id, "episodes_total": len(eps),
            "episodes_used": used, "seen": best}


def decklist(cnt_for_name):
    return dict(sorted(cnt_for_name.items(), key=lambda x: -x[1]))


def print_decklist(name, cnt):
    total = sum(cnt.values())
    print(f"  {name}: {total}/60 cards")
    for i, n in cnt.items():
        c = _CARDS.get(i)
        print(f"     x{n:2d} {i:5} {c.name if c else '?'}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--leaderboard":
        n = int(args[1]) if len(args) > 1 else 10
        k = int(args[2]) if len(args) > 2 else 5
        api = _api()
        rows = []
        token = None
        while len(rows) < n:
            page = api.competition_leaderboard_view(COMP, page_size=50, page_token=token)
            page = list(page or [])
            if not page:
                break
            rows.extend(page)
            token = getattr(api, "next_page_token", None)
            if not token:
                break
        rows = rows[:n]
        for r in rows:
            tid = getattr(r, "team_id", None)
            tname = getattr(r, "team_name", "?")
            score = getattr(r, "score", "?")
            res = scout_team(tid, k)
            print(f"\n### {tname} (team {tid}, score {score}) "
                  f"[{res['episodes_used']}/{res['episodes_total']} eps]" if res else f"\n### {tname}: no subs")
            if res:
                # the team we scouted is our own side -> its name should be tname
                own = res["seen"].get(tname)
                if own is None and res["seen"]:
                    # fall back to the fullest bucket (scouted team appears in all its games)
                    own = max(res["seen"].values(), key=lambda b: sum(b.values()))
                if own:
                    print_decklist(tname, decklist(own))
        return
    team_id = int(args[0])
    k = int(args[1]) if len(args) > 1 else 5
    res = scout_team(team_id, k)
    if not res:
        print("no submissions")
        return
    print(f"team {team_id}: submission {res['submission_id']}, "
          f"{res['episodes_used']}/{res['episodes_total']} episodes used")
    for name, seen in sorted(res["seen"].items(), key=lambda x: -len(x[1])):
        print_decklist(name, decklist(seen))


if __name__ == "__main__":
    main()
