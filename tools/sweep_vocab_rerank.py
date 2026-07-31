"""Find every token id the reranker can ever see, so prune_vocab_rerank.py can drop the rest.

The embedding is 53,339 x 768 = 41.0M params and weight-only quantization leaves it at fp32
(it is a Gather, not a MatMul), so it dominates the submission budget. Almost all of it is
dead: with glossary='none' the prompt carries no English card text at all -- only structured
tokens (`c1152x4`, `T7.1`, `ME A[...]`, `ID ME d_alakazam a_combo`, `attach:c19@ACTIVE0`).

WHAT GETS KEPT, and why each part is not optional:
  1. every id occurring in the DATA (states + candidate texts) -- the observed distribution
  2. every id of every card in every deck under decks/, plus their attack tokens -- a card
     can sit in a 60-card list yet never be drawn in any logged game, and `DECK[...]` now
     enumerates all 60, so pruning it would render that card as [UNK] at inference
  3. every deck/archetype token for every deck -- the ID segment names the opponent's deck,
     and the predictor may name a deck that never won a logged game
  4. every single-character token -- a last-resort backstop so ANY unexpected string still
     encodes to something rather than collapsing to [UNK]
  5. the tokenizer's special ids

Re-run this after ANY prompt-format change: the format decides the token set.

    PYTHONPATH=cg-lib python tools/sweep_vocab_rerank.py \
        --data /root/data/rerank/curengine_0724_none.rerank.jsonl.gz \
        --tokenizer /root/out/rerank_gte_none --out /root/onnx/keep_ids_none.json
"""
import argparse
import gzip
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _encode_all(tok, texts, chunk=4096):
    ids = set()
    buf = []
    for t in texts:
        buf.append(t)
        if len(buf) >= chunk:
            for e in tok.encode_batch(buf):
                ids.update(e.ids)
            buf = []
    if buf:
        for e in tok.encode_batch(buf):
            ids.update(e.ids)
    return ids


def sweep_data(tok, path, limit=0):
    """Unique states + unique candidate texts. States repeat heavily across the multi-pick
    decomposition, so dedup by hash before tokenizing -- it is most of the runtime."""
    seen_state, cands, states = set(), set(), []
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            h = hashlib.blake2b(r["state"].encode(), digest_size=16).digest()
            if h not in seen_state:
                seen_state.add(h)
                states.append(r["state"])
            cands.update(r["candidates"])
            n += 1
            if limit and n >= limit:
                break
    print(f"  data: {n} records, {len(states)} unique states, {len(cands)} unique candidates",
          flush=True)
    return _encode_all(tok, states) | _encode_all(tok, sorted(cands))


def sweep_fleet(tok):
    """Card / attack / deck / archetype tokens for the WHOLE fleet, drawn or not."""
    import library
    from lm import vocab
    texts = set()
    names = list(library.list_decks())
    try:
        tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    except Exception:
        tun = {}
    for name in names:
        texts.add(vocab.deck_tok(name))
        arch = (tun.get(name) or {}).get("archetype")
        if arch:
            texts.add(vocab.arch_tok(arch))
        try:
            deck = library.read_deck(name)
        except Exception:
            continue
        for cid in set(deck):
            texts.add(vocab.card_tok(cid))
            c = vocab.card(cid)
            for aid in (getattr(c, "attacks", None) or []) if c else []:
                texts.add(vocab.attack_tok(aid))
    print(f"  fleet: {len(names)} decks -> {len(texts)} identity/card/attack tokens",
          flush=True)
    return _encode_all(tok, sorted(texts))


def sweep_singles(tok):
    """Every single-char token: the backstop that keeps unexpected text encodable."""
    v = tok.get_vocab()
    ids = {i for t, i in v.items() if len(t.lstrip("ĠĊ")) <= 1}
    print(f"  singles: {len(ids)} single-character tokens", flush=True)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="rerank jsonl.gz (repeatable via commas)")
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap records (smoke runs only)")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
    tok.no_truncation()                    # the baked 1024 cap would hide tail tokens
    hf = AutoTokenizer.from_pretrained(args.tokenizer)
    total = hf.vocab_size + len(hf.get_added_vocab())

    keep = set(int(i) for i in hf.all_special_ids)
    for path in args.data.split(","):
        print(f"sweeping {path}", flush=True)
        keep |= sweep_data(tok, path, args.limit)
    keep |= sweep_fleet(tok)
    keep |= sweep_singles(tok)

    keep = sorted(keep)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(keep, open(args.out, "w"))
    print(f"\nKEEP {len(keep)} of {total} = {len(keep) / total:.1%}  -> {args.out}")
    print(f"embedding fp32 {total * 768 * 4 / 2**20:.1f} MiB -> "
          f"{len(keep) * 768 * 4 / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
