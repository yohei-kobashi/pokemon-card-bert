"""What does the eps-exploration mislabel actually cost?

Two probe builds of the SAME games differing only in --label. Score a trained checkpoint on
three groups:

  not-explored        the 94% that were always labelled correctly -- the reference
  explored/executed   the old label: the move engine_v2 REFUSED. If the model has learned
                      engine_v2 at all it should score near chance here.
  explored/heuristic  the same states with the correct answer. The gap between this and the
                      reference says whether explored states are intrinsically harder or
                      just mislabelled.

Run with the two probe files as argv[2:]; records are matched by (game_id, i, kind).
"""
import collections
import gzip
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            out.setdefault((r["game_id"], r["i"], r["kind"]), []).append(r)
    return out


mdir = sys.argv[1]
exe, heur = load(sys.argv[2]), load(sys.argv[3])

groups = collections.defaultdict(list)
for k, rows in heur.items():
    for j, r in enumerate(rows):
        if not r.get("explored"):
            groups["not-explored"].append(r)
        else:
            groups["explored/heuristic"].append(r)
            alt = exe.get(k)
            if alt and j < len(alt):
                groups["explored/executed"].append(alt[j])

tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForSequenceClassification.from_pretrained(
    mdir, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()

print(f"{'group':22s} {'rows':>7s} {'top1':>7s} {'chance':>7s}")
for name in ("not-explored", "explored/heuristic", "explored/executed"):
    rows = groups[name][:4000]
    hit = n = chance = 0
    i = 0
    with torch.no_grad():
        while i < len(rows):
            grp, npairs = [], 0
            while i < len(rows) and npairs < 192:
                grp.append(rows[i]); npairs += len(rows[i]["candidates"]); i += 1
            pairs, owner = [], []
            for ri, r in enumerate(grp):
                for c in r["candidates"]:
                    pairs.append([r["state"], c]); owner.append(ri)
            enc = tok(pairs, padding=True, truncation="only_first", max_length=640,
                      return_tensors="pt").to("cuda")
            lg = model(**enc).logits.squeeze(-1).float()
            per = [[] for _ in grp]
            for k2, ri in enumerate(owner):
                per[ri].append(lg[k2])
            for r, s in zip(grp, per):
                hit += int(int(torch.stack(s).argmax()) == r["chosen"])
                chance += 1.0 / len(r["candidates"])
                n += 1
    print(f"{name:22s} {n:7d} {100.0 * hit / max(1, n):6.1f}% {100.0 * chance / max(1, n):6.1f}%")
