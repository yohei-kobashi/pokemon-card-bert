#!/usr/bin/env python3
"""Build the local battle environment on Windows / macOS / Linux, then prove it works.

The repo carries no game engine: cg-lib/ is the organiser's library (competition data,
not redistributable) and is gitignored. This script puts it in place, picking the right
binary for THIS machine, and finishes by playing two real games so that "setup finished"
means the engine actually ran rather than the files merely existing.

The competition ships prebuilt binaries for every desktop platform:

    cg.dll            Windows x86-64
    libcg.so          Linux    x86-64
    libcg-arm64.so    Linux    aarch64
    libcg.dylib       macOS    arm64 (Apple Silicon)

Intel Macs have no prebuilt binary, so there the script compiles the published C++ source
with clang++ (~20 s). --build forces that path anywhere.

    python tools/kenkyu/setup_local.py                    # fetch what is missing + verify
    python tools/kenkyu/setup_local.py --from-dir D:/cg   # copy a cg/ folder you already have
    python tools/kenkyu/setup_local.py --drive auto        # + Drive (found automatically)
    python tools/kenkyu/setup_local.py --drive "G:/My Drive/PTCG"   # + Drive (explicit path)
    python tools/kenkyu/setup_local.py --check            # verify only
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, drive_candidates  # noqa: E402

COMP = "pokemon-tcg-ai-battle"
CG_DIR = os.path.join(ROOT, "cg-lib", "cg")
SRC_DIR = os.path.join(ROOT, "data", "kaggle_engine")
PY_FILES = ("__init__.py", "api.py", "game.py", "sim.py", "utils.py")
SAMPLE = "sample_submission/sample_submission/cg/"
ENGINE_PREFIX = "ptcg_engine/"


def say(msg):
    print(msg, flush=True)


def target_binary():
    """(competition file name, local file name) for this machine, or (None, None) to build.

    cg/sim.py loads "cg.dll" on Windows and "libcg.so" everywhere else, so the macOS dylib
    and the Linux arm64 build are COPIED to that name rather than loaded under their own --
    dlopen does not care about the extension, and renaming keeps sim.py untouched.
    """
    sysname, mach = platform.system(), platform.machine().lower()
    if sysname == "Windows":
        return "cg.dll", "cg.dll"
    if sysname == "Darwin":
        return ("libcg.dylib", "libcg.so") if mach in ("arm64", "aarch64") else (None, None)
    if mach in ("x86_64", "amd64"):
        return "libcg.so", "libcg.so"
    if mach in ("aarch64", "arm64"):
        return "libcg-arm64.so", "libcg.so"
    return None, None


def kaggle_api():
    try:
        import kaggle
    except ImportError:
        sys.exit("kaggle package missing -- pip install kaggle, then put kaggle.json in "
                 "~/.kaggle/ (Windows: %USERPROFILE%\\.kaggle\\)")
    except OSError as e:                       # kaggle raises this when the token is absent
        sys.exit("kaggle credentials not found (%s). Download kaggle.json from your Kaggle "
                 "account page and put it in ~/.kaggle/" % e)
    return kaggle.api


def fetch_file(api, name, dest):
    """Download one competition file into ``dest`` (unzipping if Kaggle wrapped it)."""
    os.makedirs(dest, exist_ok=True)
    api.competition_download_file(COMP, name, path=dest, quiet=True)
    base = os.path.basename(name)
    zpath = os.path.join(dest, base + ".zip")
    if os.path.exists(zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        os.remove(zpath)
    return os.path.join(dest, base)


def fetch_engine_source(api):
    """The C++ headers + Export.cpp, flattened into data/kaggle_engine/."""
    names, tok = [], None
    while True:
        r = api.competition_list_files(COMP, page_size=200, page_token=tok)
        if not r.files:
            break
        names += [f.name for f in r.files]
        tok = getattr(r, "nextPageToken", None)
        if not tok:
            break
    want = [n for n in names if n.startswith(ENGINE_PREFIX)]
    if not want:
        sys.exit("no %s files in the competition data -- have you accepted the rules?" % ENGINE_PREFIX)
    for n in want:
        fetch_file(api, n, SRC_DIR)
    say("  engine source: %d files -> %s" % (len(want), SRC_DIR))


def build_from_source(out):
    """Compile libcg from the published source (Intel Mac, or --build)."""
    if not os.path.exists(os.path.join(SRC_DIR, "Export.cpp")):
        say("  engine source missing -- fetching it")
        fetch_engine_source(kaggle_api())
    cxx = os.environ.get("CXX") or ("clang++" if platform.system() == "Darwin" else "g++")
    if not shutil.which(cxx):
        sys.exit("%s not found. macOS: run `xcode-select --install`. Linux: install g++." % cxx)
    cmd = [cxx, "-std=c++20", "-O2", "-DNDEBUG", "-fPIC", "-shared", "-fvisibility=hidden",
           "-I" + SRC_DIR, os.path.join(SRC_DIR, "Export.cpp"), "-o", out]
    say("  building: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    say("  built %s (%d bytes)" % (out, os.path.getsize(out)))


def ensure_python_lib(from_dir, api_getter):
    os.makedirs(CG_DIR, exist_ok=True)
    missing = [f for f in PY_FILES if not os.path.exists(os.path.join(CG_DIR, f))]
    if not missing:
        return
    if from_dir:
        for f in PY_FILES:
            src = os.path.join(from_dir, f)
            if not os.path.exists(src):
                sys.exit("%s has no %s -- point --from-dir at the cg/ folder itself" % (from_dir, f))
            shutil.copy2(src, os.path.join(CG_DIR, f))
        say("  python library copied from %s" % from_dir)
        return
    api = api_getter()
    for f in missing:
        fetch_file(api, SAMPLE + f, CG_DIR)
    say("  python library: %d files -> %s" % (len(missing), CG_DIR))


def ensure_binary(from_dir, api_getter, force_build):
    remote, local = target_binary()
    dest = os.path.join(CG_DIR, local or "libcg.so")
    if os.path.exists(dest) and not force_build:
        say("  binary already present: %s" % dest)
        return dest
    if force_build or remote is None:
        if remote is None and not force_build:
            say("  no prebuilt binary for %s/%s -- compiling from source"
                % (platform.system(), platform.machine()))
        build_from_source(dest)
        return dest
    if from_dir:
        src = os.path.join(from_dir, remote)
        if not os.path.exists(src):
            sys.exit("%s has no %s (this machine needs that one)" % (from_dir, remote))
        shutil.copy2(src, dest)
    else:
        got = fetch_file(api_getter(), SAMPLE + remote, CG_DIR)
        if os.path.abspath(got) != os.path.abspath(dest):
            shutil.move(got, dest)
    say("  binary: %s -> %s" % (remote, dest))
    return dest


def verify(games=2):
    """Play real games. Importing cg proves the binary loads; playing proves it runs."""
    sys.path.insert(0, os.path.join(ROOT, "cg-lib"))
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    cwd = os.getcwd()
    os.chdir(ROOT)                        # deck/agent paths in the repo are relative
    try:
        try:
            import arena
            import library
            from battle_log import load_agent
        except OSError as e:
            # ctypes raises OSError when the shared library is there but cannot be loaded --
            # wrong architecture is the usual cause, and the raw message names neither file.
            sys.exit("the engine binary did not load (%s).\n"
                     "  This is normally an architecture mismatch. Try rebuilding it:\n"
                     "    python tools/kenkyu/setup_local.py --build" % e)
        d0, d1 = library.read_deck("dragapult_dusknoir"), library.read_deck("ogerpon_mono")
        a0, a1 = load_agent("dragapult_dusknoir"), load_agent("ogerpon_mono")
        res = [arena.play(a0, a1, d0, d1) for _ in range(games)]
    finally:
        os.chdir(cwd)
    ok = sum(1 for r in res if r in (0, 1))
    say("  played %d/%d games to a result %s" % (ok, games, res))
    return ok == games


def export_drive(drive, api_getter):
    """Put a copy of the game library in Drive so Colab needs no Kaggle credentials.

    Colab is Linux x86-64 whatever the machine at home is, so the LINUX binary is exported --
    a Windows player's cg.dll would be useless there. That is the whole reason this is a
    separate step from ensure_binary()."""
    dest = os.path.join(drive, "cg")
    os.makedirs(dest, exist_ok=True)
    for f in PY_FILES:
        shutil.copy2(os.path.join(CG_DIR, f), os.path.join(dest, f))
    linux = os.path.join(dest, "libcg.so")
    if not os.path.exists(linux):
        local = os.path.join(CG_DIR, "libcg.so")
        if platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64") \
                and os.path.exists(local):
            shutil.copy2(local, linux)
        else:
            got = fetch_file(api_getter(), SAMPLE + "libcg.so", dest)
            if os.path.abspath(got) != os.path.abspath(linux):
                shutil.move(got, linux)
    say("  engine copy for Colab: %s" % dest)


def resolve_drive_arg(arg):
    """--drive "<path>" | --drive auto | "" -> the folder to use (or "").

    "auto" and a wrong path are the two ways a first-time user gets stuck here, so both are
    handled explicitly: auto picks the Drive this machine actually has, and a path whose
    PARENT does not exist is rejected instead of silently creating a stray folder that Drive
    will never sync (a mistyped drive letter otherwise looks like success until the notebook
    finds no games)."""
    if not arg:
        return ""
    if arg.strip().lower() == "auto":
        found = drive_candidates()
        if not found:
            sys.exit("--drive auto found no Google Drive folder on this machine.\n"
                     + drive_help())
        chosen = os.path.join(found[0], "PTCG")
        say("  --drive auto -> %s" % chosen)
        return chosen
    arg = os.path.expanduser(arg)
    parent = os.path.dirname(os.path.abspath(arg))
    if not os.path.isdir(parent):
        sys.exit("--drive %r: the folder %r does not exist.\n%s" % (arg, parent, drive_help()))
    return arg


def drive_help():
    """What to tell someone who has never used Drive on this machine."""
    found = drive_candidates()
    if found:
        return ("  This machine has:\n"
                + "".join("    %s\n" % f for f in found)
                + "  Use one of those, e.g.:  --drive \"%s\"\n"
                  % os.path.join(found[0], "PTCG"))
    if platform.system() == "Linux":
        return ("  Linux has no official Google Drive app. Point --drive at any local folder\n"
                "  (e.g. --drive ~/PTCG) and upload it later with:\n"
                "    python tools/kenkyu/sync_logs.py --zip battles.zip\n")
    return ("  Google Drive for desktop does not seem to be installed/running.\n"
            "  Install it from https://www.google.com/drive/download/ , sign in, then re-run\n"
            "  with --drive auto. Or skip Drive for now: the games are always saved to logs/\n"
            "  and can be zipped later with tools/kenkyu/sync_logs.py --zip.\n")


def export_repo(drive):
    """Snapshot the repo into Drive as repo.zip (~3 MB).

    The Colab notebook clones from GitHub (the repository is public), and falls back to this
    zip when the clone produces nothing -- GitHub being unreachable, or the repository having
    been made private again. Cheap insurance: the setup already has the working copy in front
    of it, and a notebook that cannot obtain the code cannot start at all."""
    out = os.path.join(drive, "repo.zip")
    try:
        subprocess.run(["git", "archive", "--format=zip", "-o", out, "HEAD"],
                       cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        say("  repo snapshot skipped (%s)" % e)
        return
    say("  repo snapshot for Colab: %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))


def setup_drive(drive):
    """Create the Drive folders and record the mirror in config.json.

    battle_log.save_battle reads ``play.log_mirror`` and drops a gzipped copy of every
    finished game there, so Drive for desktop uploads it while the next game is being
    played -- no separate step to remember after a play session."""
    logs = os.path.join(drive, "logs")
    models = os.path.join(drive, "models")
    for d in (logs, models):
        os.makedirs(d, exist_ok=True)
    cfg_path = os.path.join(ROOT, os.environ.get("PLAY_CONFIG", "config.json"))
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    cfg.setdefault("play", {})["log_mirror"] = logs
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    say("  Drive mirror: %s (recorded in %s)" % (logs, os.path.basename(cfg_path)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-dir", default="", help="a cg/ folder to copy instead of downloading "
                    "(e.g. the one this kit exported to Google Drive)")
    ap.add_argument("--build", action="store_true", help="compile the engine from source even if "
                    "a prebuilt binary exists")
    ap.add_argument("--drive", default="", help="Google Drive folder for this research (logs "
                    "and models get subfolders, and battles mirror there automatically). "
                    "Pass 'auto' to use whichever Drive folder this machine has.")
    ap.add_argument("--no-export", action="store_true", help="--drive: do not copy the game "
                    "library and a repo snapshot into Drive (Colab then needs its own "
                    "kaggle.json and a public GitHub repo)")
    ap.add_argument("--check", action="store_true", help="verify an existing setup, change nothing")
    ap.add_argument("--games", type=int, default=2, help="verification games")
    a = ap.parse_args()
    # "~/Google Drive/..." reaches here unexpanded when the caller quoted it (and the docs tell
    # them to quote it, because these paths contain spaces).
    a.drive = resolve_drive_arg(a.drive)
    a.from_dir = os.path.expanduser(a.from_dir) if a.from_dir else ""

    say("PTCG free-research setup  (%s %s, Python %s)"
        % (platform.system(), platform.machine(), platform.python_version()))
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required (3.11+ recommended)")

    if not a.check:
        say("[1/3] game library")
        ensure_python_lib(a.from_dir, kaggle_api)
        ensure_binary(a.from_dir, kaggle_api, a.build)
        if a.drive:
            say("[2/3] Google Drive")
            setup_drive(a.drive)
            if not a.no_export:
                export_drive(a.drive, kaggle_api)
                export_repo(a.drive)
        else:
            say("[2/3] Google Drive: not set up yet")
            say(drive_help().rstrip("\n"))
    say("[3/3] verification")
    if not verify(a.games):
        sys.exit("verification FAILED -- the engine did not play a full game")
    say("OK. Start playing:  python play_server.py   then open http://localhost:8000/")


if __name__ == "__main__":
    main()
