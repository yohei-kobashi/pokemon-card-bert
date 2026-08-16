#!/usr/bin/env python3
"""Read exported live replays and say HOW each game was lost, not just that it was.

The live rating is the only signal we trust ([[live-alakazam-beats-us]]), and it is a single
number. This turns the games behind it back into a diagnosis: per turn, the prize race, our
draw engine, our board, and the opponent's hand size -- the last because the two decks that
beat us most, alakazam and marnie, both scale damage off resources rather than energy, so
"they were allowed to accumulate" is a loss mode that does not show up in a prize count until
it is over.

Reads the output of tools/export_live_logs.py (the local logs/ replay schema). Our seat is
taken from `current.yourIndex` per FILE rather than assumed from the filename: the exporter
orders labels by board index, but the Kaggle episode list reports our index as None for seat
0, and one wrong seat silently inverts every number in the report.

    PYTHONPATH=cg-lib python tools/live_loss_report.py --logs /root/livelogs
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _name(cid):
    try:
        from lm import vocab
        c = vocab.card(cid)
        return getattr(c, "name", None) or ("c%s" % cid)
    except Exception:
        return "c%s" % cid


def _side(p):
    """-> (prizes_left, deck, hand, n_bench, active_id, active_hp)"""
    act = p.get("active")
    act = act if isinstance(act, dict) else None
    return (len(p.get("prize") or []), p.get("deckCount"), len(p.get("hand") or []),
            len(p.get("bench") or []), (act or {}).get("id"), (act or {}).get("hp"))


def report(path, verbose=False):
    els = json.load(open(path))
    if not els:
        return None
    me = (els[0].get("current") or {}).get("yourIndex")
    me = 0 if me is None else int(me)
    op = 1 - me

    rows = []
    last_turn = None
    for el in els:
        cur = el.get("current") or {}
        pl = cur.get("players") or []
        if len(pl) < 2:
            continue
        t = cur.get("turn")
        if t == last_turn:
            rows[-1] = (t, _side(pl[me]), _side(pl[op]))    # keep the LAST state of each turn
        else:
            rows.append((t, _side(pl[me]), _side(pl[op])))
            last_turn = t
    if not rows:
        return None

    t, mine, theirs = rows[-1]
    my_taken, op_taken = 6 - mine[0], 6 - theirs[0]
    won = my_taken > op_taken if my_taken != op_taken else None

    print("\n=== %s" % os.path.basename(path))
    print("    seat %d | %d turns | prizes taken US %d - THEM %d | %s"
          % (me, t or 0, my_taken, op_taken,
             "WIN" if won else ("LOSS" if won is False else "unclear")))
    print("    turn |  our prizes  deck hand bench | their prizes  deck hand bench")
    for t, a, b in rows:
        print("    %4s | %6d      %4s %4s %5s | %7d      %4s %4s %5s"
              % (t, 6 - a[0], a[1], a[2], a[3], 6 - b[0], b[1], b[2], b[3]))

    # The loss-mode tells. Each is a different fix, and they are easy to confuse in a summary.
    tells = []
    if mine[1] is not None and mine[1] <= 3:
        tells.append("OUR DECK IS NEARLY OUT (%s left) -- deck-out risk, not a prize race" % mine[1])
    if theirs[2] >= 12:
        tells.append("THEIR HAND reached %d cards -- resource-scaling damage (Powerful Hand / "
                     "Grimmsnarl) is priced off this" % theirs[2])
    if my_taken == 0:
        tells.append("WE TOOK ZERO PRIZES -- not a race we lost, a race we never entered")
    if mine[3] == 0 and mine[4] is None:
        tells.append("OUR BOARD IS EMPTY at the end -- bench-out loss")
    peak_hand = max(a[2] for _t, a, _b in rows)
    if peak_hand <= 4:
        tells.append("OUR HAND PEAKED AT %d -- the draw engine never ran" % peak_hand)
    for s in tells:
        print("    !! " + s)
    return {"file": os.path.basename(path), "seat": me, "turns": t,
            "my_prizes": my_taken, "op_prizes": op_taken, "won": won,
            "our_deck_end": mine[1], "their_hand_end": theirs[2],
            "our_hand_peak": peak_hand, "tells": tells}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out = []
    for p in sorted(glob.glob(os.path.join(a.logs, "*.json"))):
        try:
            r = report(p)
        except Exception as e:                                    # noqa: BLE001
            print("  %s: %s" % (os.path.basename(p), e), file=sys.stderr)
            continue
        if r:
            out.append(r)

    w = [r for r in out if r["won"]]
    print("\n%d games: %d W / %d L" % (len(out), len(w), len(out) - len(w)))
    losses = [r for r in out if not r["won"]]
    if losses:
        print("in the %d losses: prizes taken mean %.1f/6, our end deck mean %.1f, "
              "their end hand mean %.1f, our hand peak mean %.1f"
              % (len(losses), sum(r["my_prizes"] for r in losses) / len(losses),
                 sum((r["our_deck_end"] or 0) for r in losses) / len(losses),
                 sum(r["their_hand_end"] for r in losses) / len(losses),
                 sum(r["our_hand_peak"] for r in losses) / len(losses)))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("-> %s" % a.out)


if __name__ == "__main__":
    main()
