"""Audit: does the glossary='none' state carry everything a decision needs?

Two questions, both answered against the REAL self-play logs the reranker trained on:

 1. Is ``ID ME <deck>`` actually OUR deck? build_rerank derives it from the filename
    (``<a>__vs__<b>`` -> {0: a, 1: b}) but the 60-card lists come from ``header['decks']``
    keyed by player index. If seats are assigned independently of filename order, the ME
    token names the OPPONENT's deck in ~half the records.

 2. Which obs fields exist but never reach the string? Collect every key at every level
    and diff against what lm/serialize.py reads.
"""
import collections
import glob
import gzip
import json
import os
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    sys.path.insert(0, p)

import library                                        # noqa: E402
from tools.build_rerank import _deck_names, _game_decks, _read_game   # noqa: E402


def q1_deck_name_alignment(files, per_file=3):
    """Compare header['decks'][p] (the real list) against read_deck(dn[p])."""
    ok = bad = unk = 0
    examples = []
    for path in files:
        n = 0
        for header, steps in _read_game(path):
            gd = _game_decks(header, steps)
            dn = _deck_names(header, path)
            if len(dn) != 2 or len(gd) != 2:
                unk += 1
                continue
            for p in (0, 1):
                try:
                    truth = collections.Counter(library.read_deck(dn[p]))
                except Exception:
                    unk += 1
                    continue
                got = collections.Counter(gd[p])
                if got == truth:
                    ok += 1
                else:
                    bad += 1
                    if len(examples) < 5:
                        other = collections.Counter(library.read_deck(dn[1 - p]))
                        examples.append((os.path.basename(path), header["game_id"], p,
                                         dn[p], "matches-the-OTHER-name" if got == other
                                         else "matches-neither"))
            n += 1
            if n >= per_file:
                break
    print(f"Q1 deck-name alignment: ok={ok} MISMATCH={bad} unresolved={unk}")
    for e in examples:
        print("   ", e)
    return bad


def _walk(d, prefix, out, depth=0):
    if depth > 3:
        return
    if isinstance(d, dict):
        for k, v in d.items():
            out[prefix].add(k)
            if isinstance(v, (dict, list)):
                _walk(v, prefix + "." + k, out, depth + 1)
    elif isinstance(d, list):
        for x in d[:3]:
            _walk(x, prefix, out, depth)


def q2_obs_fields(files, per_file=2):
    out = collections.defaultdict(set)
    vals = collections.defaultdict(collections.Counter)
    for path in files:
        n = 0
        for header, steps in _read_game(path):
            for s in steps[:400]:
                obs = s.get("obs") or {}
                cur = obs.get("current") or {}
                _walk(obs.get("select") or {}, "select", out)
                for k, v in cur.items():
                    out["current"].add(k)
                    if not isinstance(v, (dict, list)):
                        vals["current." + k][repr(v)] += 1
                for pl in (cur.get("players") or []):
                    for k, v in pl.items():
                        out["player"].add(k)
                        if not isinstance(v, (dict, list)):
                            vals["player." + k][repr(v)] += 1
                    for z in ("active", "bench"):
                        for pk in (pl.get(z) or []):
                            if not isinstance(pk, dict):
                                continue
                            for k, v in pk.items():
                                out["pokemon"].add(k)
                                if not isinstance(v, (dict, list)):
                                    vals["pokemon." + k][repr(v)] += 1
                                elif isinstance(v, list) and v:
                                    vals["pokemon." + k][f"<list len {len(v)}>"] += 1
            n += 1
            if n >= per_file:
                break
    for scope in sorted(out):
        print(f"\n== {scope} keys ==")
        print("  " + ", ".join(sorted(out[scope])))
    print("\n== value samples (non-container fields) ==")
    for k in sorted(vals):
        top = vals[k].most_common(4)
        print(f"  {k:32s} {top}")
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(ROOT, "data/selfplay/curengine_0724/*__vs__*.jsonl.gz")))
    print(f"{len(files)} matchup files")
    import random
    sel = random.Random(3).sample(files, min(40, len(files)))
    q1_deck_name_alignment(sel)
    q2_obs_fields(sel[:4])
