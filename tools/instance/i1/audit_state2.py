"""Audit part 2: quantify what the glossary='none' state omits, on real logs.

  A. duplicate candidate texts -- training dedups them, inference does NOT, so every
     duplicate is a wasted re-encode of the whole state (the reranker's only cost).
  B. special conditions -- rendered from the PLAYER dict; do they ever fire?
  C. attached energy -- `_pk` prints TYPE LETTERS, so a special energy card is
     indistinguishable from a basic. How often is an attached energy non-basic?
  D. whose turn is it -- can the model tell? (`firstPlayer` is never rendered)
  E. select.type vs select.context -- is `type` extra information?
  F. tool cards / stadium ownership / multiple stadiums
"""
import collections
import glob
import gzip
import json
import os
import random
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    sys.path.insert(0, p)

from lm import vocab                                   # noqa: E402
from lm.actions import encode_option                   # noqa: E402
from tools.build_rerank import _read_game              # noqa: E402


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "data/selfplay/curengine_0724/*__vs__*.jsonl.gz")))
    sel_files = random.Random(11).sample(files, 30)

    n_dec = n_opt = n_uniq = 0
    dup_hist = collections.Counter()
    cond = collections.Counter()
    nrg_kind = collections.Counter()
    nrg_special = collections.Counter()
    turn_owner = collections.Counter()
    ctx_type = collections.Counter()
    stad_n = collections.Counter()
    firstp = collections.Counter()
    tools_n = collections.Counter()

    for path in sel_files:
        g = 0
        for header, steps in _read_game(path):
            for s in steps:
                obs = s.get("obs") or {}
                cur = obs.get("current") or {}
                sel = obs.get("select") or {}
                opts = sel.get("option") or []
                if len(opts) < 2:
                    continue
                n_dec += 1
                enc = [encode_option(o, obs) for o in opts]
                n_opt += len(enc)
                u = len(set(enc))
                n_uniq += u
                dup_hist[len(enc) - u] += 1
                ctx_type[(sel.get("context"), sel.get("type"))] += 1
                firstp[cur.get("firstPlayer")] += 1
                stad_n[len(cur.get("stadium") or [])] += 1

                yi = cur.get("yourIndex")
                # whose turn: the engine gives the acting seat on `player`
                turn_owner[("acting=me" if s.get("player") == yi else "acting=OPPONENT",
                            bool(s.get("is_main")))] += 1

                for pl in (cur.get("players") or []):
                    for k in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
                        if pl.get(k):
                            cond[k] += 1
                    for z in ("active", "bench"):
                        for pk in (pl.get(z) or []):
                            if not isinstance(pk, dict):
                                continue
                            tools_n[len(pk.get("tools") or [])] += 1
                            for ec in (pk.get("energyCards") or []):
                                cid = ec.get("id") if isinstance(ec, dict) else ec
                                c = vocab.card(cid)
                                kind = getattr(c, "cardType", None) if c else None
                                nrg_kind[kind] += 1
                                if kind == 6:            # SP-NRG
                                    nrg_special[cid] += 1
            g += 1
            if g >= 3:
                break

    print(f"decisions with >=2 options: {n_dec}")
    print(f"A. candidate texts: {n_opt} raw -> {n_uniq} unique "
          f"({100*(n_opt-n_uniq)/max(1,n_opt):.1f}% are DUPLICATES = wasted re-encodes)")
    print(f"   duplicates per decision: {sorted(dup_hist.items())[:12]}")
    print(f"B. special conditions seen: {dict(cond) or 'NONE'}")
    print(f"C. attached energy by cardType: {dict(nrg_kind)}  (5=NRG basic, 6=SP-NRG special)")
    print(f"   top special energies: {nrg_special.most_common(8)}")
    print(f"D. acting seat: {dict(turn_owner)}")
    print(f"   firstPlayer values: {dict(firstp)}")
    print(f"E. (context,type) pairs: {sorted(ctx_type.items(), key=lambda x:-x[1])[:12]}")
    print(f"F. stadium list length: {dict(stad_n)}   tools per pokemon: {dict(tools_n)}")


if __name__ == "__main__":
    main()
