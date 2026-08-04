#!/usr/bin/env python3
"""Build a mirror-capable copy of the competition battle engine.

The engine's C++ source is competition data under LicenseRef-PTCG-ABC-Competition-Use-Only:
build-and-test only, no redistribution. THIS REPO IS PUBLIC, so nothing derived from it may be
committed. Everything this script touches lives under data/ (gitignored); the script itself
carries no engine code beyond the single token it rewrites.

    python3 tools/build_engine_mirror.py --fetch     # download the source from Kaggle
    python3 tools/build_engine_mirror.py             # patch + build

What the patch does
-------------------
Stock `Game` owns ONE std::mt19937 that both players draw from, so player 1's shuffles depend on
how many coin flips player 0 made. The patch adds a per-player stream, both seeded from the same
value, and routes every player-attributable RNG site to the acting player's stream:

    ShuffleDeck              -> playerIndex
    coin flips (x2)          -> playerIndex
    randomSelect target list -> selectPlayerIndex
    ToDeckBottomClose        -> effectPlayerIndex()

With `mirrorRand` on and the SAME 60 cards given to both players, both decks are shuffled from an
identical stream on identical input, so they get identical orders -- and stay independent of each
other afterwards. That is common random numbers between the two seats: the decklist and every
shuffle are held fixed, so a difference in outcome is a difference in play.

The Search API's own shuffle (Search.h) is deliberately left on the shared stream -- it is not
part of a live game.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "kaggle_engine")
DST = os.path.join(ROOT, "data", "kaggle_engine_mirror")
EXT = os.path.join(ROOT, "data", "kaggle_engine_ext")
COMP = "pokemon-tcg-ai-battle"

# The generator expression each file's RNG use must be re-pointed at, and how many uses to expect.
RNG_OWNER = {
    "CardMove.h": ("playerIndex", 1),
    "SelectProc.h": ("playerIndex", 2),
    "EffectProc.h": ("selectPlayerIndex", 1),
    "EffectInstant.h": ("state.effectPlayerIndex()", 1),
}
TOKEN = "state.game->rng"


def fetch():
    import kaggle
    api = kaggle.api
    tok, names = None, []
    while True:
        r = api.competition_list_files(COMP, page_size=200, page_token=tok)
        if not r.files:
            break
        names += [f.name for f in r.files]
        tok = getattr(r, "nextPageToken", None)
        if not tok:
            break
    want = [n for n in names if n.startswith("ptcg_engine/")]
    if not want:
        sys.exit("no ptcg_engine/ files in the competition data")
    os.makedirs(SRC, exist_ok=True)
    for n in want:
        api.competition_download_file(COMP, n, path=SRC, quiet=True)
    print(f"fetched {len(want)} files -> {SRC}")


def sub_once(text, pattern, repl, what, count=1):
    new, n = re.subn(pattern, repl, text)
    if n != count:
        sys.exit(f"patch anchor {what!r} matched {n} times, expected {count}")
    return new


def patch():
    if not os.path.isdir(SRC):
        sys.exit(f"{SRC} missing -- run with --fetch first")
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)

    # 1. a per-player stream on Game, plus the accessor every other patch calls.
    p = os.path.join(DST, "Game.h")
    t = open(p, encoding="utf-8").read()
    t = sub_once(t, r"\bbool deviceRand;", "bool deviceRand;\n\tbool mirrorRand;", "GameConfig.deviceRand")
    t = sub_once(
        t, r"\bstd::mt19937 rng;",
        "std::mt19937 rng;\n"
        "\tstd::mt19937 rngP[2];\n"
        "\tstd::mt19937& prng(int p) { return config.mirrorRand ? rngP[p & 1] : rng; }",
        "Game.rng")
    t = sub_once(
        t, r"\brng = std::mt19937\(this->config\.seed\);",
        "rng = std::mt19937(this->config.seed);\n"
        "\t\trngP[0] = std::mt19937(this->config.seed);\n"
        "\t\trngP[1] = std::mt19937(this->config.seed);",
        "Game::init seeding")
    open(p, "w", encoding="utf-8").write(t)

    # 2. route each live-game RNG use to the acting player's stream.
    for fname, (owner, count) in RNG_OWNER.items():
        p = os.path.join(DST, fname)
        t = open(p, encoding="utf-8").read()
        t = sub_once(t, re.escape(TOKEN), f"state.game->prng({owner})", f"{fname} rng use", count)
        open(p, "w", encoding="utf-8").write(t)

    print(f"patched -> {DST}")


def build(out, extra):
    cmd = ["g++", "-std=c++20", "-O2", "-DNDEBUG", "-fPIC", "-shared",
           "-fvisibility=hidden", f"-I{DST}",
           os.path.join(EXT, "mirror_export.cpp"), "-o", out] + extra
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"built {out} ({os.path.getsize(out):,} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the engine source from Kaggle first")
    ap.add_argument("--out", default=os.path.join(EXT, "libcg_mirror.so"))
    ap.add_argument("--cxxflags", default="", help="extra flags for g++")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    patch()
    build(a.out, a.cxxflags.split())
