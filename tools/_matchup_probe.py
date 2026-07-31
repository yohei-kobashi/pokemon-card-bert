"""Probe crustle_stall v3 vs v4 (same wall-first agent, deck differs by Boss x2 vs
Cornerstone-nonex x2) against its weak matchups: winrate + avg prizes taken/given
+ how it loses (board-wipe / low-prize).
    PYTHONPATH=cg-lib python tools/_matchup_probe.py <games>
"""
import sys, os, csv, collections
sys.path.insert(0, "tools"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import library
from battle_log import load_agent
from cg.game import battle_start, battle_select, battle_finish
from multiprocessing import Pool

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
DECK = "crustle_stall"
WEAK = ["mega_lucario", "alakazam", "mega_starmie", "dragapult", "crustle", "ceruledge"]


def _v4_list():
    out = []
    with open(f"decks/{DECK}.csv") as f:
        for r in csv.reader(f):
            if r and r[0].strip().isdigit():
                out.append(int(r[0]))
    return out


def _v3_from_v4(v4):
    c = collections.Counter(v4)
    c[386] += 2; c[1182] -= 2   # v3 had Cornerstone-nonex x2, no Boss
    return [cid for cid, n in c.items() for _ in range(n)]


V4 = _v4_list()
V3 = _v3_from_v4(V4)
AGENT = "crustle_stall"


def _play(mydeck, oppname, first):
    me = load_agent(AGENT); opp = load_agent(oppname)
    od = library.read_deck(oppname)
    d0, d1, a0, a1 = (mydeck, od, me, opp) if first else (od, mydeck, opp, me)
    obs, sd = battle_start(d0, d1)
    myidx = 0 if first else 1
    last = None
    for _ in range(4000):
        cur = obs.get("current")
        if cur is None:
            break
        if cur.get("players"):
            last = cur
        if cur.get("result", -1) != -1:
            break
        sel = obs.get("select")
        if sel is None:
            break
        yi = cur["yourIndex"]
        try:
            obs = battle_select((a0 if yi == 0 else a1)(obs))
        except Exception:
            obs = None; break
    res = last.get("result") if last else None
    myp = oppp = None; mydeckout = False
    if last and last.get("players"):
        mp = last["players"][myidx]; op = last["players"][1 - myidx]
        myp = 6 - len(mp.get("prize", [])); oppp = 6 - len(op.get("prize", []))
        mydeckout = mp.get("deckCount", 1) == 0
    battle_finish()
    win = (res == myidx)
    return win, myp, oppp, mydeckout


def _task(args):
    label, deck, opp = args
    w = mp = op = dko = n = 0
    for g in range(GAMES):
        win, a, b, d = _play(deck, opp, g % 2 == 0)
        if a is None:
            continue
        n += 1; w += win; mp += a; op += b; dko += d
    return label, opp, w, n, mp, op, dko


def main():
    tasks = [("v4", V4, o) for o in WEAK] + [("v3", V3, o) for o in WEAK]
    res = {}
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for label, opp, w, n, mp, op, dko in pool.imap_unordered(_task, tasks):
            res[(label, opp)] = (w, n, mp, op, dko)
    print(f"crustle_stall vs weak matchups ({GAMES}g each, same wall-first agent)")
    print(f"{'opp':16} {'v3 WR':>7} {'v3 mine/opp prz':>16} | {'v4 WR':>7} {'v4 mine/opp prz':>16}  {'ΔWR':>5}")
    print("-" * 78)
    tot = {"v3": [0, 0], "v4": [0, 0]}
    for opp in WEAK:
        row = ""
        wr = {}
        for label in ("v3", "v4"):
            w, n, mp, op, dko = res[(label, opp)]
            wr[label] = 100 * w / n if n else 0
            tot[label][0] += w; tot[label][1] += n
        v3 = res[("v3", opp)]; v4 = res[("v4", opp)]
        print(f"{opp:16} {wr['v3']:6.1f}% {v3[4]/max(1,v3[1]):5.2f}/{v3[3]/max(1,v3[1]):.2f} (dko{v3[4] if False else v3[4]:>0}) | "
              f"{wr['v4']:6.1f}% {v4[4]/max(1,v4[1]):5.2f}/{v4[3]/max(1,v4[1]):.2f}  {wr['v4']-wr['v3']:+5.1f}")
    print("-" * 78)
    for label in ("v3", "v4"):
        w, n = tot[label]
        print(f"{label} overall vs weak: {100*w/n:.1f}% ({w}/{n})")


if __name__ == "__main__":
    main()
