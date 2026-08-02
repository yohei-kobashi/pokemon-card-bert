#!/usr/bin/env python3
"""The reranker's listwise imitation pool -> the decoder's SFT schema.

instance1 and instance2 train two different heads on the SAME decisions. The reranker pool
(``state`` / ``candidates`` / ``chosen``) and the decoder pool (``prompt`` / ``target``) are two
renderings of one corpus, and only instance1 ever regenerated its copy: instance2's base was
built 2026-08-01, before ``menu_dedup`` landed, so its 193,919 rows still show one menu entry per
menu POSITION (six identical ``facedown:PRIZEn`` lines) while inference renders one entry per ACT
(``rl_config.PROMPT_FMT``, read by ``mirror_match.make_agent``). Every base row was training the
decoder on a format it never meets in play.

The conversion is a reformat, not a re-derivation. ``build_rerank._emit`` dedups the option texts
and renders the menu FROM THE DEDUPED LIST, so position i of the menu is candidate i by
construction -- verified on 200,000 rows, 0 mismatches. This tool checks that identity per record
rather than trusting it, and falls back to matching the chosen candidate by ``canon_key`` (what
``valued_to_sft`` does) when a record does not line up. Records that resolve to no menu position
are DROPPED and counted; a target index that silently points at the wrong option is invisible in
training and shows up only as a worse win rate.

Sharding is by line number so N shards can run concurrently over one gzip stream each and the
concatenation is the whole file exactly once.
"""
import argparse
import gzip
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

ACT = "[ACT]\n"
_OPT = re.compile(r"(?:^| )(\d+)=(\S+)")


def menu_of(state):
    """The rendered menu as option strings, or None when it is not numbered 0..n-1.

    Mirrors ``sft_teacher.option_texts``: the trainer reads the menu out of the prompt, so a
    prompt this tool cannot parse the same way must not be accepted with a target index derived
    some other way.
    """
    opts = _OPT.findall(state.rsplit(":: ", 1)[-1])
    if [int(i) for i, _ in opts] != list(range(len(opts))):
        return None
    return [t for _, t in opts]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True, help="comma-separated rerank .jsonl.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--drop-explored", action="store_true",
                    help="skip exploration steps (they carry a different label convention)")
    a = ap.parse_args()

    from lm.action_token import canon_key, slot_map_from_state   # noqa: E402

    n = kept = direct = fallback = 0
    drop = {"no menu": 0, "bad chosen": 0, "act not in menu": 0, "explored": 0}
    with gzip.open(a.out, "wt") as g:
        for path in [p for p in a.inp.split(",") if p]:
            with gzip.open(path, "rt") as f:
                for ln, line in enumerate(f):
                    if ln % a.nshards != a.shard:
                        continue
                    d = json.loads(line)
                    n += 1
                    if a.drop_explored and d.get("explored"):
                        drop["explored"] += 1
                        continue
                    state = d.get("state") or ""
                    cands, ch = d.get("candidates") or [], d.get("chosen")
                    if ch is None or not 0 <= ch < len(cands):
                        drop["bad chosen"] += 1
                        continue
                    menu = menu_of(state)
                    if menu is None:
                        drop["no menu"] += 1
                        continue
                    if len(menu) == len(cands) and menu[ch] == cands[ch]:
                        tgt = ch                      # the expected case: menu IS the candidate list
                        direct += 1
                    else:
                        slots = slot_map_from_state(state)
                        want = canon_key(cands[ch], slots)
                        tgt = next((i for i, t in enumerate(menu)
                                    if canon_key(t, slots) == want), None)
                        if tgt is None:
                            drop["act not in menu"] += 1
                            continue
                        fallback += 1
                    g.write(json.dumps({
                        "prompt": ACT + state, "target": str(tgt),
                        "game_id": d.get("game_id"), "i": d.get("i"),
                        "kind": d.get("kind", "main"), "mode": "act",
                        "deck": d.get("deck"), "opp": d.get("opp"),
                        "explored": bool(d.get("explored"))}, ensure_ascii=False) + "\n")
                    kept += 1
    print("[shard %d/%d] %s -> %s" % (a.shard, a.nshards, a.inp, a.out))
    print("  %d read | %d written (%d direct, %d canon-key) | dropped %s"
          % (n, kept, direct, fallback, drop))


if __name__ == "__main__":
    main()
