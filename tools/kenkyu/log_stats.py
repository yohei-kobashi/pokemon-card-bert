#!/usr/bin/env python3
"""Summarise the collected battles: how many, against whom, won or lost, how much data.

Run this BEFORE training. It answers the two questions the training cell cannot: is there
enough data in the chosen date range, and did the human's win rate move over the days
(which is what the研究 is measuring in the first place -- the model can only learn a way of
playing that the person was actually using).

    python tools/kenkyu/log_stats.py
    python tools/kenkyu/log_stats.py --since 2026-08-16 --until 2026-08-18
    python tools/kenkyu/log_stats.py --logs "/content/drive/MyDrive/PTCG/logs" --json out.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LOG_DIR, decisions_of, list_logs, open_log, opponent_of,  # noqa: E402
                    parse_date, parse_stamp, result_of)


def collect(files, seat=0):
    games = []
    for f in files:
        try:
            log = open_log(f)
        except (OSError, ValueError) as e:
            print("  skipped %s (%s)" % (os.path.basename(f), e))
            continue
        won, turns = result_of(log, seat)
        real, total = decisions_of(log, seat)
        t = parse_stamp(f)
        games.append({"file": os.path.basename(f), "date": t.strftime("%Y-%m-%d"),
                      "time": t.strftime("%H:%M"), "opp": opponent_of(f), "won": won,
                      "turns": turns, "decisions": real, "selects": total})
    return games


def _rate(rows):
    played = [g for g in rows if g["won"] is not None]
    w = sum(1 for g in played if g["won"])
    return w, len(played), (100.0 * w / len(played) if played else 0.0)


def report(games):
    if not games:
        print("no games in range")
        return {}
    w, n, wr = _rate(games)
    dec = sum(g["decisions"] for g in games)
    print("=== 対戦履歴の集計 ===")
    print("games      : %d  (%s .. %s)" % (len(games), games[0]["date"], games[-1]["date"]))
    print("human wins : %d / %d = %.1f%%" % (w, n, wr))
    print("decisions  : %d (学習データの上限, 平均 %.1f/game)" % (dec, dec / len(games)))
    print("turns      : 平均 %.1f" % (sum(g["turns"] for g in games) / len(games)))

    print("\n--- 相手デッキ別 ---")
    by = collections.defaultdict(list)
    for g in games:
        by[g["opp"]].append(g)
    per_opp = {}
    for opp in sorted(by, key=lambda k: -len(by[k])):
        w, n, wr = _rate(by[opp])
        d = sum(x["decisions"] for x in by[opp])
        per_opp[opp] = {"games": len(by[opp]), "wins": w, "win_rate": wr, "decisions": d}
        print("  %-22s %3d games  %2d勝 %2d敗  %5.1f%%  %5d decisions"
              % (opp, len(by[opp]), w, n - w, wr, d))

    print("\n--- 日付別 ---")
    byd = collections.defaultdict(list)
    for g in games:
        byd[g["date"]].append(g)
    per_day = {}
    for day in sorted(byd):
        w, n, wr = _rate(byd[day])
        per_day[day] = {"games": len(byd[day]), "wins": w, "win_rate": wr,
                        "decisions": sum(x["decisions"] for x in byd[day])}
        print("  %s  %3d games  %2d勝 %2d敗  %5.1f%%" % (day, len(byd[day]), w, n - w, wr))
    w, n, wr = _rate(games)
    return {"games": len(games), "wins": w, "decided": n, "win_rate": wr,
            "decisions": dec, "by_opponent": per_opp, "by_day": per_day,
            "files": [g["file"] for g in games]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="*", default=[LOG_DIR], help="dirs/globs of saved games")
    ap.add_argument("--since", default="", help="2026-08-16")
    ap.add_argument("--until", default="")
    ap.add_argument("--opp", default="", help="only games against this opponent deck")
    ap.add_argument("--seat", type=int, default=0, help="the human's player index")
    ap.add_argument("--json", dest="json_out", default="", help="also write the numbers here")
    a = ap.parse_args()

    files = list_logs(a.logs, parse_date(a.since), parse_date(a.until, end=True),
                      opp=a.opp or None)
    print("%d log files in range" % len(files))
    st = report(collect(files, a.seat))
    if a.json_out and st:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
        print("\nwrote %s" % a.json_out)


if __name__ == "__main__":
    main()
