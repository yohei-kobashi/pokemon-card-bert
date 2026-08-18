"""Shared helpers for the free-research kit: log discovery, dates, Drive paths.

One place decides what counts as a collected game, so the aggregator (log_stats.py), the
uploader (sync_logs.py) and the training-row builder all see the SAME set of files for the
same --since/--until. When they disagree the training set silently stops matching the
statistics the研究 reports, which is the kind of error nobody notices.
"""
import glob
import gzip
import json
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG_DIR = os.path.join(ROOT, "logs")
# battle_log.save_battle names files {stamp}_{mode}_{p0}_vs_{p1}.json, stamp = %Y%m%d-%H%M%S.
STAMP_FMT = "%Y%m%d-%H%M%S"


def parse_stamp(path):
    """Datetime encoded in the FILENAME, or None. Cheap enough to filter thousands of
    files without opening any of them."""
    base = os.path.basename(path)
    try:
        return datetime.strptime(base[:15], STAMP_FMT)
    except ValueError:
        return None


def parse_date(s, end=False):
    """'2026-08-16', '20260816' or '2026-08-16 21:00' -> datetime.

    ``end=True`` makes a bare date mean the END of that day, so --since 2026-08-16
    --until 2026-08-17 includes everything played on the 17th. A half-open range that
    silently drops the last day's games is the classic off-by-one here."""
    if not s:
        return None
    s = s.strip().replace("/", "-")
    for fmt, is_day in (("%Y-%m-%d %H:%M:%S", False), ("%Y-%m-%d %H:%M", False),
                        ("%Y-%m-%d", True), ("%Y%m%d-%H%M%S", False), ("%Y%m%d", True)):
        try:
            d = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if is_day and end:
            d += timedelta(days=1) - timedelta(seconds=1)
        return d
    raise ValueError("cannot read a date from %r (try 2026-08-16)" % s)


def open_log(path):
    """Load a saved battle (the heroz visualize array). .json and .json.gz both."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_logs(paths=None, since=None, until=None, mode="HumanvAI", human_deck=None,
              exclude_stub=True):
    """Battle logs under ``paths`` (dirs, globs or files), filtered by name.

    ``mode`` "HumanvAI" keeps only games a person actually played -- self-play logs land in
    the same folder and are not expert data. ``exclude_stub`` drops the ``_vs_agent-`` games
    played against the random stub agent (they are not a real opponent); tools/human_rows.py
    skips those too, and the two must not disagree.
    """
    paths = paths or [LOG_DIR]
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "*.json")) + glob.glob(os.path.join(p, "*.json.gz"))
        else:
            files += glob.glob(p)
    out = []
    for f in sorted(set(files)):
        base = os.path.basename(f)
        if mode and ("_%s_" % mode) not in base:
            continue
        if exclude_stub and "_vs_agent-" in base:
            continue
        if human_deck and human_deck not in base:
            continue
        t = parse_stamp(f)
        if t is None:
            continue
        if since and t < since:
            continue
        if until and t > until:
            continue
        out.append(f)
    return out


def opponent_of(path):
    """AI side's deck name, read from the filename ( ..._vs_<agent>-<deck>.json )."""
    base = os.path.basename(path).split(".json")[0]
    tail = base.rsplit("_vs_", 1)[-1]
    return tail.split("-", 1)[1] if "-" in tail else tail


def result_of(log, seat=0):
    """(won, turns) for ``seat`` -- the Result log entry carries the winner's index."""
    res, turns = None, 0
    for e in log:
        cur = e.get("current") or {}
        turns = max(turns, cur.get("turn") or 0)
    for e in reversed(log):
        for lg in e.get("logs", []) or []:
            if lg.get("type") == "Result":
                res = lg.get("result")
                break
        if res is not None:
            break
    return (None if res is None else res == seat), turns


def decisions_of(log, seat=0):
    """(real decisions, total selects) for ``seat``.

    A "real" decision is one with at least two options -- the rest are forced and carry no
    training signal, so counting them would inflate every per-game number the研究 reports."""
    real = total = 0
    for e in log:
        cur = e.get("current") or {}
        if cur.get("yourIndex") != seat:
            continue
        opts = (e.get("select") or {}).get("option") or []
        total += 1
        if len(opts) >= 2 and isinstance(e.get("selected"), list):
            real += 1
    return real, total


def drive_candidates():
    """Likely Google Drive roots on this machine, best first.

    Drive for desktop mounts as G:\\My Drive (Windows), ~/Google Drive/My Drive (macOS),
    /content/drive/MyDrive (Colab). Linux has no official client, so a manual sync folder
    is the fallback -- sync_logs.py --zip covers that case."""
    home = os.path.expanduser("~")
    cands = [
        os.path.join(home, "Google Drive", "My Drive"),
        os.path.join(home, "Google Drive", "マイドライブ"),
        os.path.join(home, "GoogleDrive", "My Drive"),
        "/content/drive/MyDrive",
    ]
    if os.name == "nt":
        for drv in ("G:", "H:", "I:"):
            cands.append(os.path.join(drv + os.sep, "My Drive"))
            cands.append(os.path.join(drv + os.sep, "マイドライブ"))
    return [c for c in cands if os.path.isdir(c)]
