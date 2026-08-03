#!/usr/bin/env python3
"""Playout-valued attach records (reranker schema) -> the decoder's SFT schema.

`tools/attach_label.py` writes listwise records -- state, the deduped candidate list, and the
index of the candidate the playouts measured as best. The Qwen card-first trainer wants a
prompt and a TARGET INDEX INTO THE RAW MENU, because that is what `option_texts` reads and what
`label_b` turns into the answer tokens.

The bridge is the act, not the position: the winning candidate is matched back to the first raw
menu entry that performs the same act, using the same `canon_key` the dedup used. Matching by
text alone would miss every case where the surviving candidate is one of several twins, which
is most of them on attach decisions.

Records whose winner cannot be located in the menu are DROPPED and counted, not guessed at. A
target index that points at the wrong option is invisible in training and shows up only as a
worse win rate -- the failure mode this project has hit repeatedly.

The value labels are carried through in `qvals` even though the decoder does not read them: the
loss is cross-entropy over one or two answer tokens and has no margin term. Keeping them costs
nothing and leaves the door open.
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lm.action_token import canon_key, slot_map_from_state   # noqa: E402

ACT = "[ACT]\n"


def menu_of(state):
    """The raw menu as option strings, or None when the numbering is not 0..n-1.

    Mirrors sft_teacher.option_texts exactly: a prompt whose entries are renumbered (as the
    v40 menu-dedup rewrite does) must not be silently accepted here, because the trainer would
    read a different list than this converter did.
    """
    import re
    opts = re.findall(r"(?:^| )(\d+)=(\S+)", state.rsplit(":: ", 1)[-1])
    if [int(i) for i, _ in opts] != list(range(len(opts))):
        return None
    return [t for _, t in opts]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True, help="comma-separated attach_label.py outputs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n = kept = 0
    drop = {"no menu": 0, "bad chosen": 0, "act not in menu": 0}
    kinds = {}
    with gzip.open(a.out, "wt") as g:
        for path in [p for p in a.inp.split(",") if p]:
            with gzip.open(path, "rt") as f:
                for line in f:
                    d = json.loads(line)
                    n += 1
                    state = d.get("state") or ""
                    cands, ch = d.get("candidates") or [], d.get("chosen")
                    menu = menu_of(state)
                    if menu is None:
                        drop["no menu"] += 1
                        continue
                    if ch is None or not 0 <= ch < len(cands):
                        drop["bad chosen"] += 1
                        continue
                    slots = slot_map_from_state(state)
                    want = canon_key(cands[ch], slots)
                    tgt = next((i for i, t in enumerate(menu)
                                if canon_key(t, slots) == want), None)
                    if tgt is None:
                        drop["act not in menu"] += 1
                        continue
                    k = cands[ch].split(":", 1)[0]
                    kinds[k] = kinds.get(k, 0) + 1
                    g.write(json.dumps({
                        "prompt": ACT + state, "target": str(tgt),
                        "game_id": "valued/%s" % d.get("deck", "?"), "i": kept,
                        "kind": "main", "mode": "act",
                        "qvals": d.get("qvals"), "deck": d.get("deck"),
                        "valued": True}, ensure_ascii=False) + "\n")
                    kept += 1
    print("%s -> %s" % (a.inp, a.out))
    print("  %d read | %d written | dropped %s" % (n, kept, drop))
    print("  label kind: %s" % sorted(kinds.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
