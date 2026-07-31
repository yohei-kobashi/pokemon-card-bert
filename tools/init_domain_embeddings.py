"""Give the 3,087 domain tokens embeddings that actually differ from one another.

THE PROBLEM. ``resize_token_embeddings`` (transformers >= 4.46, ``mean_resizing=True``)
seeds every new row from the OLD embeddings' mean. Measured on the v34 checkpoint:

    BASE rows   |mu| 0.507   per-row residual 2.504   -> residual/|mu| 4.94
    ADDED rows  |mu| 0.504   per-row residual 0.058   -> residual/|mu| 0.11
    cos(mean_added, mean_base) = 0.9992

so all 2,971 ``c<cardId>`` / ``a<attackId>`` / ``d_<deck>`` / ``a_<archetype>`` rows sit in a
ball **43x tighter** than real tokens: pairwise cosine +0.998 against +0.085 for English
words, and the same after ModernBERT's embedding LayerNorm. ``c344`` and ``c1152`` are very
nearly the SAME INPUT VECTOR. A full epoch (1.2M records) moved them ~1-3% -- about half the
(already tiny) gap between rows -- so training does not dig itself out on this budget.

It matters far more with ``glossary='none'``: the state is then almost NOTHING but these
tokens. The full-glossary model scored 56.7% by reading the English rules text, whose
embeddings are fine; deleting that text removed the only well-conditioned signal it had.

THE FIX. Rebuild each domain row from the BASE vocabulary's own embeddings: tokenize a short
natural-language descriptor of the card / attack / deck and average those rows. Two wins at
once -- the rows separate, AND the descriptor carries exactly the semantics the deleted
glossary used to spell out (name, type, stage, HP, damage), for zero prompt tokens. Cards
with similar names and types land near each other, which is the prior that lets a card seen
20 times generalise.

The residual (row - global mean) is renormalised to the base vocabulary's typical residual
length, times ``--scale``. Scale 1.0 puts domain tokens on the same footing as real words;
lower values stay closer to what the partially-trained body already expects, so sweep it
against held-out top1 rather than assuming.

    PYTHONPATH=cg-lib python tools/init_domain_embeddings.py \
        --model /root/out/rerank_gte_none --out /root/out/rerank_gte_init --scale 1.0
"""
import argparse
import collections
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

_ETYPE_WORD = {0: "colorless", 1: "grass", 2: "fire", 3: "water", 4: "lightning",
               5: "psychic", 6: "fighting", 7: "darkness", 8: "metal", 9: "dragon",
               10: "rainbow any", 11: "team rocket"}
_KIND_WORD = {1: "item trainer card", 2: "pokemon tool trainer card",
              3: "supporter trainer card", 4: "stadium trainer card",
              5: "basic energy card", 6: "special energy card"}


def _card_desc(cid):
    from lm import vocab
    c = vocab.card(cid)
    if not c:
        return None
    name = (c.name or "").strip()
    if c.cardType != 0:
        return f"{name} {_KIND_WORD.get(c.cardType, 'trainer card')}"
    stage = ("mega evolution" if c.megaEx else "stage 2" if c.stage2
             else "stage 1" if c.stage1 else "basic")
    tags = " ".join(w for w, f in (("ex", c.ex), ("tera", c.tera)) if f)
    return (f"{name} {tags} {stage} {_ETYPE_WORD.get(c.energyType, '')} pokemon "
            f"{c.hp} hit points retreat {c.retreatCost}").replace("  ", " ")


def _attack_desc(aid):
    from lm import vocab
    a = vocab._ATTACKS.get(aid)
    if not a:
        return None
    cost = " ".join(_ETYPE_WORD.get(e, "") for e in (a.energies or []))
    return f"{(a.name or '').strip()} attack {a.damage} damage costing {cost}".strip()


def descriptors(tok):
    """{added token -> descriptor text}. Tokens we cannot describe are left untouched."""
    from cg.api import AreaType, OptionType, SelectContext
    from lm import vocab
    out = {}
    for cid in vocab._CARDS:
        d = _card_desc(cid)
        if d:
            out[f"c{cid}"] = d
    for aid in vocab._ATTACKS:
        d = _attack_desc(aid)
        if d:
            out[f"a{aid}"] = d
    decks, arches = vocab._fleet_names()
    for name in decks:
        out[vocab.deck_tok(name)] = "deck " + name.replace("_", " ")
    for a in arches:
        out[vocab.arch_tok(a)] = "strategy " + a.replace("_", " ")
    for cls, pre, word in ((SelectContext, "ctx_", "choice"), (OptionType, "opt_", "action"),
                           (AreaType, "area_", "zone")):
        for e in cls:
            out[f"{pre}{e.name}"] = f"{word} {e.name.lower().replace('_', ' ')}"
    return {t: d for t, d in out.items() if tok.convert_tokens_to_ids(t) is not None}


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="residual length as a multiple of the base vocabulary's typical row")
    ap.add_argument("--mode", default="idf_centered",
                    choices=("mean", "idf", "idf_centered"),
                    help="how to pool a descriptor. 'mean' is dominated by the words EVERY "
                         "descriptor shares ('pokemon', 'hit points'), leaving rows at "
                         "cosine ~0.70; 'idf' downweights them by document frequency; "
                         "'idf_centered' also removes the component common to all domain "
                         "rows, which is what actually separates 2,971 near-synonyms")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32)
    E = model.get_input_embeddings().weight.data
    nbase = tok.vocab_size
    base = E[:nbase].float()
    mu = base.mean(0)
    target = (base - mu).norm(dim=1).mean().item()
    print(f"base vocab {nbase}, added {len(tok.get_added_vocab())}, "
          f"typical base residual {target:.4f}")

    desc = descriptors(tok)
    print(f"describable domain tokens: {len(desc)}")

    def spread(ids):
        M = E[ids].float()
        M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
        C = M @ M.T
        return C[~torch.eye(len(C), dtype=torch.bool)].mean().item()

    probe = [tok.convert_tokens_to_ids(t) for t in list(desc)[:400]]
    print(f"pairwise cosine BEFORE: {spread(probe):+.4f}")

    # descriptors are plain English; drop any id in the ADDED range so a domain row can
    # never be seeded from another (equally uninformative) domain row
    enc = {t: [i for i in tok(x, add_special_tokens=False)["input_ids"] if i < nbase]
           for t, x in desc.items()}
    enc = {t: i for t, i in enc.items() if i}
    df = collections.Counter(i for ids in enc.values() for i in set(ids))
    N = len(enc)
    idf = {i: math.log(N / c) for i, c in df.items()}

    vecs, toks = [], []
    for t, ids in enc.items():
        if args.mode == "mean":
            v = base[ids].mean(0)
        else:
            w = torch.tensor([idf[i] for i in ids], dtype=torch.float32)
            if float(w.sum()) <= 0:                    # every word appears everywhere
                w = torch.ones_like(w)
            v = (base[ids] * w.unsqueeze(1)).sum(0) / w.sum()
        vecs.append(v)
        toks.append(t)
    V = torch.stack(vecs)
    centre = V.mean(0) if args.mode == "idf_centered" else mu
    R = V - centre
    R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-9) * (target * args.scale)
    with torch.no_grad():
        for t, r in zip(toks, R):
            E[tok.convert_tokens_to_ids(t)] = (mu + r).to(E.dtype)
    print(f"re-initialised {len(toks)} rows, mode={args.mode}, scale={args.scale}")
    print(f"pairwise cosine AFTER:  {spread(probe):+.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
