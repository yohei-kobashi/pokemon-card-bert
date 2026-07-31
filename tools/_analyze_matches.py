"""Analyze one submission's full replays: opponent archetype, result, prize spread,
turns, and each side's attackers -> weakness profile.
    PYTHONPATH=cg-lib python tools/_analyze_matches.py <submission_id> <my_team_id>
"""
import sys, os, json
sys.path.insert(0, "tools")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kaggle
from collections import Counter, defaultdict
from scout_decks import _download_episode, _game_decklists
from leaderboard_distribution import _load_our_decks, classify, _pokemon_ids
from agents._engine import _CARDS, _ATTACKS

SUB = int(sys.argv[1]); MYTEAM = int(sys.argv[2])
api = kaggle.api
OUR = _load_our_decks()


def _side_index(rep, myname):
    return rep["info"]["TeamNames"].index(myname)


def analyze(rep, myname):
    names = rep["info"]["TeamNames"]
    mi = names.index(myname); oi = 1 - mi
    rew = rep.get("rewards", [0, 0])
    win = rew[mi] > rew[oi]
    # final state
    last = None
    for step in reversed(rep["steps"]):
        cur = step[0]["observation"].get("current")
        if cur and cur.get("players"):
            last = cur; break
    myp = last["players"][mi]; opp = last["players"][oi]
    my_taken = 6 - len(myp.get("prize", []))
    opp_taken = 6 - len(opp.get("prize", []))
    turns = last.get("turn", "?")
    my_deck_out = myp.get("deckCount", 1) == 0
    opp_deck_out = opp.get("deckCount", 1) == 0
    # attackers used (scan all steps for ATTACK actions is hard; infer from damage/discard)
    # opponent archetype
    dls = _game_decklists(rep)
    opp_cnt = dls.get(names[oi], {})
    arch, score = classify(opp_cnt, OUR)
    sig = ", ".join(f"{nm}x{c}" for c, i, nm in _pokemon_ids(opp_cnt)[:3])
    label = arch if score >= 0.5 else ("OTHER:" + (_pokemon_ids(opp_cnt)[0][2] if _pokemon_ids(opp_cnt) else "?"))
    return {"win": win, "opp": names[oi], "arch": label, "sig": sig,
            "my_prizes": my_taken, "opp_prizes": opp_taken, "turns": turns,
            "my_deckout": my_deck_out, "opp_deckout": opp_deck_out}


def main():
    eps = list(api.competition_list_episodes(SUB))
    myname = None
    rows = []
    for e in eps:
        # find my team name
        for a in (e.agents or []):
            d = a.to_dict()
            if d.get("teamId") == MYTEAM:
                myname = d.get("teamName")
        try:
            rep = _download_episode(api, e.id)
        except Exception as ex:
            continue
        if myname not in rep["info"]["TeamNames"]:
            continue
        try:
            r = analyze(rep, myname)
            r["ep"] = e.id
            rows.append(r)
        except Exception as ex:
            print("  analyze fail", e.id, ex, file=sys.stderr)
    rows.sort(key=lambda x: (x["win"], x["ep"]))
    print(f"\n=== {myname} sub {SUB}: {sum(r['win'] for r in rows)}W-{sum(not r['win'] for r in rows)}L ===")
    print(f"{'RES':4} {'opp prizes':>10} {'me':>3} {'turns':>5}  {'end':10} {'archetype':22} signature")
    for r in rows:
        end = "MY DECKOUT" if r["my_deckout"] else ("opp deckout" if r["opp_deckout"] else "prizes")
        print(f"{'WIN' if r['win'] else 'LOSS':4} {r['opp_prizes']:>10} {r['my_prizes']:>3} {str(r['turns']):>5}  {end:10} {r['arch'][:22]:22} [{r['sig']}]  vs {r['opp']}")
    # loss archetype tally
    losses = [r for r in rows if not r["win"]]
    print("\nLOSS archetypes:", dict(Counter(r["arch"] for r in losses)))
    print("avg prizes taken in LOSSES: me=%.1f opp=%.1f" % (
        sum(r["my_prizes"] for r in losses)/max(1,len(losses)),
        sum(r["opp_prizes"] for r in losses)/max(1,len(losses))))
    print("avg prizes taken in WINS:   me=%.1f opp=%.1f" % (
        sum(r["my_prizes"] for r in rows if r["win"])/max(1,sum(r["win"] for r in rows)),
        sum(r["opp_prizes"] for r in rows if r["win"])/max(1,sum(r["win"] for r in rows))))


if __name__ == "__main__":
    main()
