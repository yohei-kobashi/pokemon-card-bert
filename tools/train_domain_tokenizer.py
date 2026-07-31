"""Train a DOMAIN tokenizer on the reranker states (dominated by card ability text) and
report the NEW subword tokens to ADD to the ModernBERT tokenizer, plus the real token-length
reduction. Numbers are split into individual digits (pre_tokenizers.Digits) so HP/damage/
counts stay as single-digit tokens (generalization). Adds SUBWORD units (bounded vocab), NOT
long phrases -> the fresh embeddings stay learnable.

Outputs:
  <out>/domain_tokens.json : {"tokens": [ ...new subword strings not already in modernbert... ]}
Measures state token-length with modernbert-base BEFORE vs AFTER adding these tokens.

Usage:
  python tools/train_domain_tokenizer.py --data /root/data/rerank/curengine_0724.rerank.jsonl.gz \
    --vocab 3000 --out /root/data/rerank
"""
import argparse, gzip, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Alibaba-NLP/gte-reranker-modernbert-base")
    ap.add_argument("--vocab", type=int, default=3000, help="target domain-tokenizer vocab")
    ap.add_argument("--corpus-n", type=int, default=60000, help="states to train the tokenizer on")
    ap.add_argument("--out", default="/root/data/rerank")
    args = ap.parse_args()

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import AutoTokenizer

    # 1) corpus = states (ability text dominates) + candidate strings
    corpus = []
    with gzip.open(args.data, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus.append(r["state"])
            if len(corpus) >= args.corpus_n:
                break
    print(f"corpus: {len(corpus)} states", flush=True)

    # 2) train a BPE with DIGIT-SPLITTING pre-tokenization (numbers -> single-digit tokens)
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, min_frequency=5,
                                  special_tokens=["[UNK]"])
    tk.train_from_iterator(corpus, trainer)
    domain_vocab = set(tk.get_vocab().keys())
    print(f"domain tokenizer trained: {len(domain_vocab)} tokens", flush=True)

    # 3) keep only domain subwords NOT already covered by modernbert (as plain-word tokens).
    #    We test membership by tokenizing the bare word; a domain token is "new" if modernbert
    #    splits it into >1 piece (i.e. adding it would actually shorten something).
    mb = AutoTokenizer.from_pretrained(args.model)
    new_tokens = []
    for t in domain_vocab:
        w = t.strip()
        if len(w) < 3 or not any(c.isalpha() for c in w):     # skip fragments/punct/digits
            continue
        if len(mb.tokenize(" " + w)) > 1:                     # modernbert over-fragments it
            new_tokens.append(w)
    new_tokens = sorted(set(new_tokens))
    print(f"NEW subword tokens to add (modernbert over-fragments): {len(new_tokens)}", flush=True)
    print("  sample:", new_tokens[:30], flush=True)

    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "domain_tokens.json")
    json.dump({"tokens": new_tokens}, open(outp, "w"))
    print(f"-> {outp}", flush=True)

    # 4) measure real state token-length: modernbert BASE vs +domain-subwords vs +domain +IDs
    import numpy as np
    from lm.vocab import special_tokens
    sample = corpus[:3000]

    def lens(tokzr):
        return np.array([len(tokzr(s, add_special_tokens=False)["input_ids"]) for s in sample])

    base = lens(mb)
    mb2 = AutoTokenizer.from_pretrained(args.model)
    mb2.add_tokens(new_tokens)
    subw = lens(mb2)
    mb3 = AutoTokenizer.from_pretrained(args.model)
    mb3.add_tokens(special_tokens())          # the ID/attack/enum tokens (already used)
    mb3.add_tokens(new_tokens)                # + domain subwords
    both = lens(mb3)

    def rep(name, a):
        print(f"  {name:26} mean {a.mean():6.0f}  p50 {np.percentile(a,50):5.0f}  "
              f"p90 {np.percentile(a,90):5.0f}  max {a.max():5.0f}", flush=True)
    print("STATE token length:")
    rep("modernbert base", base)
    rep("+domain subwords", subw)
    rep("+IDs +domain subwords", both)
    print(f"reduction (base -> +IDs+subwords): {100*(1-both.mean()/base.mean()):.1f}% shorter")


if __name__ == "__main__":
    main()
