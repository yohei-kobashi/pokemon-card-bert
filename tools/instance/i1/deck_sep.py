"""Could the model TELL TWO DECKLISTS APART at all?

swapDECK (substituting another real deck's DECK[...]) costs v34 only 1.2pt. Two readings:
the model CHOSE to ignore the content, or it COULD NOT SEE it. This separates them at the
input layer, before any learning is involved: mean-pool the input embeddings of each deck's
DECK[] token string and measure how different two decks look.

If pairwise cosine is ~1.0 the decklists are near-identical VECTORS and no amount of training
could have used them -- the ablation was measuring a blind spot, not a preference.
"""
import itertools
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path[:0] = ["/root/ptcg/repo", "/root/ptcg/repo/cg-lib", "/root/ptcg/repo/tools"]
import library  # noqa: E402
from lm.serialize import render_my_deck  # noqa: E402

DECKS = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "crustle_stall", "alakazam", "mega_lucario", "dragapult", "hydrapple", "iono_bellibolt"]

for mdir in sys.argv[1].split(","):
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(
        mdir, trust_remote_code=True, dtype=torch.float32)
    emb = model.get_input_embeddings().weight.detach()
    vecs, oov = {}, {}
    for d in DECKS:
        # pool ONLY the c<id> tokens. The literal scaffolding ("DECK", "[", "x", the count
        # digits, the commas) is identical for every deck and is ~75% of the segment, so
        # pooling the whole string measures the scaffolding, not the decklist.
        card = [f"c{c}" for c in sorted(set(library.read_deck(d)))]
        ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in card]
        vecs[d] = emb[ids].mean(0)
        oov[d] = sum(1 for c in card if len(tok(c, add_special_tokens=False)["input_ids"]) > 1)
    cos = []
    for a, b in itertools.combinations(DECKS, 2):
        c = torch.nn.functional.cosine_similarity(vecs[a], vecs[b], dim=0).item()
        cos.append(c)
    lo, hi = min(cos), max(cos)
    print(f"{os.path.basename(mdir):22s} DECK[] pairwise cosine: mean {sum(cos)/len(cos):+.4f} "
          f"min {lo:+.4f} max {hi:+.4f}   multi-piece card tokens/deck: "
          f"{sum(oov.values())/len(oov):.1f}")
    del model
