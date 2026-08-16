"""Did the 50%-wrong ``ID ME d_X`` label damage the deck/archetype token embeddings?

The d_*/a_* rows are SHARED between the two places a deck name appears:
  ``ID ME d_alakazam a_combo``   <- 50% wrong in the v34 'none' data
  ``OP d_alakazam:4 ...``        <- always a genuine prediction, correct by construction
so the model cannot have zeroed the embedding without also breaking the OP segment. If the
rows look healthy, the neglect (if any) is POSITIONAL -- learned in attention, not in the
embedding table -- and re-initialising the rows would destroy the OP capability for nothing.

MEASURE AFTER THE EMBEDDING LayerNorm. ModernBERT normalises straight after the token
lookup, so the large vector every row shares (all 2,971 domain rows were created by
``resize_token_embeddings``, which seeds them from the OLD embeddings' mean) is cancelled
before attention ever sees it. On the raw table every row looks identical -- |row| 0.507
for cards, decks and archetypes alike, pairwise cosine +0.999 -- which says nothing about
what the model can distinguish.
"""
import sys

sys.path.insert(0, "/root/ptcg/repo")
sys.path.insert(0, "/root/ptcg/repo/cg-lib")

import torch                                                    # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402


def _emb_norm(model):
    """The LayerNorm ModernBERT applies to the token embedding (identity if absent)."""
    base = getattr(model, "model", model)
    e = getattr(base, "embeddings", None)
    n = getattr(e, "norm", None) if e is not None else None
    return n if n is not None else (lambda x: x)


def rows(model, tok, toks, post_ln=True):
    emb = model.get_input_embeddings().weight.detach().float()
    ids, keep = [], []
    unk = tok.unk_token_id
    for t in toks:
        i = tok.convert_tokens_to_ids(t)
        if i is None or i == unk or not (0 <= i < emb.shape[0]):
            continue
        ids.append(i)
        keep.append(t)
    M = emb[ids]
    if post_ln:
        with torch.no_grad():
            M = _emb_norm(model)(M).float()
    return M, ids, keep


def stats(name, M):
    n = M.norm(dim=1)
    Mn = M / n.clamp_min(1e-9).unsqueeze(1)
    C = Mn @ Mn.T
    off = C[~torch.eye(len(C), dtype=torch.bool)]
    print(f"  {name:16s} n={len(M):5d}  |row| mean {n.mean():.3f}  "
          f"pairwise cos mean {off.mean():+.3f}  p95 {off.quantile(0.95):+.3f}  "
          f"max {off.max():+.3f}")


def main():
    from lm import vocab
    decks, arches = vocab._fleet_names()
    groups = (("deck d_*", [vocab.deck_tok(d) for d in decks]),
              ("arch a_*", [vocab.arch_tok(a) for a in arches]),
              ("card c*", [f"c{c}" for c in list(vocab._CARDS)[:400]]),
              ("english", ["the", "energy", "damage", "attack", "bench", "card",
                           "play", "turn", "deck", "prize", "water", "fire"]))
    out = {}
    for tag, d in (("BEFORE(mp)", "/root/out/rerank_gte_mp"),
                   ("AFTER(none)", "/root/out/rerank_gte_none")):
        tok = AutoTokenizer.from_pretrained(d)
        m = AutoModelForSequenceClassification.from_pretrained(d, trust_remote_code=True,
                                                               dtype=torch.float32)
        m.eval()
        print(f"\n== {tag} == (post embedding-LayerNorm)")
        got = {}
        for nm, tk in groups:
            M, ids, keep = rows(m, tok, tk)
            if len(M) < 2:
                print(f"  {nm}: only {len(M)} ids resolved -- tokens missing from vocab!")
                continue
            stats(nm, M)
            got[nm] = M
        out[tag] = got
        del m

    print("\n== relative movement during the 'none' run (post-LN) ==")
    for nm, _ in groups:
        if nm not in out["BEFORE(mp)"] or nm not in out["AFTER(none)"]:
            continue
        A, B = out["BEFORE(mp)"][nm], out["AFTER(none)"][nm]
        k = min(len(A), len(B))
        d = (B[:k] - A[:k]).norm(dim=1) / A[:k].norm(dim=1).clamp_min(1e-9)
        print(f"  {nm:16s} mean {d.mean():.4f}  p90 {d.quantile(0.9):.4f}  max {d.max():.4f}")


if __name__ == "__main__":
    main()
