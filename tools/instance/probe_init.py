#!/usr/bin/env python3
"""Which initialisation gives the added rows a usable geometry?

The default (transformers' multivariate normal with the old embeddings' mean and covariance)
measured mean pairwise cosine +0.8177 with max +1.0000, against +0.0118 for the base rows: the
new tokens sit in a narrow cone and some pairs are identical. That is a milder form of the
+0.998 collapse that made the reranker unable to read DECK[].

Candidates, measured rather than argued:
  default    what resize_token_embeddings does now
  subtoken   mean of the base tokenizer's pieces for the token's own text ("c1152" -> c,11,52)
  subtok_c   subtoken mean, then CENTRED on the base-embedding mean and rescaled to the base
             row-norm distribution -- the mean vector is what collapses cosine, so removing it
             is the direct fix
  gauss_c    centred gaussian at the base row-norm scale (no semantics, maximum spread)

Reported per candidate: mean/max pairwise cosine among new rows and mean row norm, against the
base rows as the reference geometry.
"""
import sys

for p in ("/root/ptcg/repo", "/root/ptcg/repo/cg-lib"):
    if p not in sys.path:
        sys.path.insert(0, p)


def stats(M, torch, label, ref=None):
    n = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    k = min(512, n.shape[0])
    idx = torch.arange(0, n.shape[0], max(1, n.shape[0] // k))[:k]
    g = n[idx] @ n[idx].T
    off = g[~torch.eye(len(idx), dtype=torch.bool, device=g.device)]
    print("  %-10s cos mean %+.4f  p99 %+.4f  max %+.4f | row-norm mean %.3f"
          % (label, off.mean(), off.quantile(0.99), off.max(), M.norm(dim=1).mean()), flush=True)
    return float(off.mean())


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm.vocab import special_tokens

    name = "unsloth/Qwen3.5-9B-Base"
    tk = AutoTokenizer.from_pretrained(name)
    toks = special_tokens()
    print("[probe] %d domain tokens" % len(toks), flush=True)

    # embeddings only -- loading the whole 9B is unnecessary for this question
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                             device_map="cpu", low_cpu_mem_usage=True)
    W = m.get_input_embeddings().weight.detach()
    n_base = W.shape[0]
    print("[probe] base embedding %s" % (tuple(W.shape),), flush=True)
    base_ref = stats(W[::max(1, n_base // 512)][:512], torch, "BASE")
    mu = W.mean(0)
    norms = W.norm(dim=1)
    tgt_norm = norms.median()

    # default: transformers' multivariate normal
    m.resize_token_embeddings(n_base + len(toks))
    W2 = m.get_input_embeddings().weight.detach()
    default = W2[n_base:].clone()
    stats(default, torch, "default")

    # subtoken mean
    sub = torch.stack([W[tk(t, add_special_tokens=False)["input_ids"]].mean(0) for t in toks])
    stats(sub, torch, "subtoken")

    # centred + rescaled
    sc = sub - mu
    sc = sc / sc.norm(dim=1, keepdim=True).clamp_min(1e-9) * tgt_norm
    stats(sc, torch, "subtok_c")

    g = torch.randn(len(toks), W.shape[1])
    g = g / g.norm(dim=1, keepdim=True) * tgt_norm
    stats(g, torch, "gauss_c")

    print("\n[probe] BASE is the geometry to match (%.4f). Anything far above it means the new "
          "tokens share a direction and must be pulled apart by training before they can carry "
          "card identity." % base_ref, flush=True)


if __name__ == "__main__":
    main()
