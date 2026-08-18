#!/usr/bin/env python3
"""Copy collected battles to Google Drive (or into a zip to upload by hand).

play_server already mirrors each finished game when ``play.log_mirror`` is set
(tools/kenkyu/setup_local.py --drive does that), so this script is for the two cases the
hook cannot cover: games played BEFORE the mirror was configured, and machines with no
Drive client -- mainly Linux, where --zip produces one file to drop into Drive in the
browser.

    python tools/kenkyu/sync_logs.py                       # -> the configured Drive folder
    python tools/kenkyu/sync_logs.py --drive "G:/My Drive/PTCG"
    python tools/kenkyu/sync_logs.py --zip battles.zip     # no Drive client
    python tools/kenkyu/sync_logs.py --watch 60            # keep syncing while you play
"""
import argparse
import gzip
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LOG_DIR, ROOT, drive_candidates, list_logs, parse_date  # noqa: E402


def configured_mirror():
    try:
        with open(os.path.join(ROOT, os.environ.get("PLAY_CONFIG", "config.json"))) as f:
            return (json.load(f).get("play") or {}).get("log_mirror") or ""
    except (OSError, ValueError):
        return ""


def resolve_drive(arg):
    if arg:
        return os.path.expanduser(arg)
    cfg = configured_mirror()
    if cfg:
        return cfg
    for c in drive_candidates():
        return os.path.join(c, "PTCG", "logs")
    return ""


def copy_one(src, dest_dir):
    """Gzip the game into dest_dir; skip if a copy is already there.

    Size is not compared: a battle log is written once and never edited, so presence of the
    name is enough and re-gzipping 200 games on every sync is pure waste."""
    name = os.path.basename(src)
    if not name.endswith(".gz"):
        name += ".gz"
    dst = os.path.join(dest_dir, name)
    if os.path.exists(dst):
        return 0
    data = open(src, "rb").read() if src.endswith(".gz") else None
    os.makedirs(dest_dir, exist_ok=True)
    if data is not None:
        with open(dst, "wb") as f:
            f.write(data)
    else:
        with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
            fo.write(fi.read())
    return os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default="", help="destination folder (default: play.log_mirror "
                    "from config.json, else an auto-detected Google Drive)")
    ap.add_argument("--logs", nargs="*", default=[LOG_DIR], help="source dirs/globs")
    ap.add_argument("--zip", dest="zip_path", default="", help="write one zip instead of copying")
    ap.add_argument("--since", default="", help="only games from this date (2026-08-16)")
    ap.add_argument("--until", default="")
    ap.add_argument("--all-modes", action="store_true", help="include AI-vs-AI logs too")
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds (Ctrl-C to stop)")
    a = ap.parse_args()

    since, until = parse_date(a.since), parse_date(a.until, end=True)
    mode = None if a.all_modes else "HumanvAI"

    def once():
        files = list_logs(a.logs, since, until, mode=mode)
        if a.zip_path:
            with zipfile.ZipFile(a.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    z.write(f, os.path.basename(f))
            print("%d games -> %s (%.1f MB)"
                  % (len(files), a.zip_path, os.path.getsize(a.zip_path) / 1e6))
            return
        dest = resolve_drive(a.drive)
        if not dest:
            sys.exit("no destination: pass --drive, or run setup_local.py --drive once, "
                     "or use --zip on a machine with no Drive client")
        n = b = 0
        for f in files:
            got = copy_one(f, dest)
            n += 1 if got else 0
            b += got
        print("%d new / %d total games -> %s (%.1f MB copied)" % (n, len(files), dest, b / 1e6))

    once()
    while a.watch:
        time.sleep(a.watch)
        once()


if __name__ == "__main__":
    main()
