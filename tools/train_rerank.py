"""Fine-tune a cross-encoder reranker (Alibaba-NLP/gte-reranker-modernbert-base) on the
LISTWISE game-decision data from build_rerank.py.

Objective (matches deploy AND RL): for each decision with candidates C and chosen j,
  score s_i = model(state, c_i)  (scalar logit, num_labels=1);  loss = -log softmax(s)_j.
This is exactly the policy we deploy (argmax_i s_i) and RL (softmax(s/tau)). The model
scores each (state, candidate) pair INDEPENDENTLY (bidirectional cross-encoder), so nothing
about candidate order/position leaks in.

Efficiency: FLATTENED pair batching -- gather records until ~--pair-batch (state,cand) pairs,
tokenize+forward ALL of them once, split logits back per record for per-record listwise CE.

Usage:
  python tools/train_rerank.py --data /root/data/rerank/curengine_0724.rerank.jsonl.gz \
    --model Alibaba-NLP/gte-reranker-modernbert-base --out /root/out/rerank_gte \
    --deadline-h 5 --pair-batch 192 --lr 2e-5
"""
import argparse, collections, gzip, hashlib, json, math, os, random, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def row_key(r):
    """Identity of a training record, stable across builds and files.

    (game_id, i) is NOT unique -- one multi-pick decision decomposes into several records
    sharing both. Hash what the model actually sees instead, so a fixed eval set can be
    subtracted from ANY later, larger training pool."""
    h = hashlib.blake2b(digest_size=16)
    h.update(r["state"].encode())
    h.update(b"\x00")
    h.update("\x01".join(r["candidates"]).encode())
    return h.digest()


def read_rows(path, cap, seed=1234, cap_matchup=0, cap_deck=0):
    """A UNIFORM sample of `cap` rows from the whole file (reservoir), not the first `cap`.

    build_rerank merges its shards in sorted-matchup order, so reading the head reads the
    alphabetically-first matchups. With cap=800000 of 1,523,895 that was **19 of the 62
    piloted decks**, silently excluding 43 of them -- including mega_lucario, one of the
    three fair-protocol eval decks. Every run before this one trained (and eval'd) on that
    slice, which is the likeliest reason mega_lucario was the weakest deck by far.

    Lines outside the reservoir are never json-parsed, so the extra full pass is cheap.

    ``cap_matchup`` switches to a reservoir PER (pilot deck, opponent deck) instead, so
    every one of the 3,683 matchups contributes the same number of decisions no matter how
    often that pilot won. Records are winner-only, so the natural distribution is
    proportional to wins (8,175-38,005 per deck); a matchup cap of ~90 yields ~330k records
    with all 63 decks and every pairing represented.
    """
    rng = random.Random(seed)
    paths = [p for p in str(path).split(",") if p]      # several builds -> one pool
    if cap_matchup:
        res, seen = {}, collections.Counter()
        for pth in paths:
            with gzip.open(pth, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    k = (r.get("deck"), r.get("opp"))
                    seen[k] += 1
                    b = res.setdefault(k, [])
                    if len(b) < cap_matchup:
                        b.append(r)
                    else:
                        j = rng.randrange(seen[k])
                        if j < cap_matchup:
                            b[j] = r
        rows = [r for b in res.values() for r in b]
        rng.shuffle(rows)
        decks = collections.Counter(r.get("deck") for r in rows)
        print(f"balanced: {len(rows)} rows over {len(res)} matchups / {len(decks)} decks "
              f"(per-deck min {min(decks.values())} max {max(decks.values())})", flush=True)
        if cap_deck:
            # A matchup cap is an UPPER BOUND, not a fill, so it cannot balance winner-only
            # data: a deck that loses a matchup has few winning rows there and never reaches
            # the cap. Measured, same games: winner-rows/both-rows runs 16.4% (rockets_
            # honchkrow) to 82.5% (mega_lucario), and --sides winner --cap-matchup 320 leaves
            # per-deck 7,671..19,840 = 2.59x, WORSE than the 1.50x of a small cap. Then any
            # gain on a strong deck is unattributable: more data, or better data?
            # Equalising per DECK after the per-matchup reservoir removes that confound.
            n_deck = cap_deck if cap_deck > 0 else min(decks.values())
            by = {}
            for r in rows:                       # rows are already shuffled -> prefix == sample
                b = by.setdefault(r.get("deck"), [])
                if len(b) < n_deck:
                    b.append(r)
            rows = [r for b in by.values() for r in b]
            rng.shuffle(rows)
            decks = collections.Counter(r.get("deck") for r in rows)
            print(f"deck-balanced to {n_deck}/deck: {len(rows)} rows "
                  f"(per-deck min {min(decks.values())} max {max(decks.values())})", flush=True)
        return rows[:cap] if cap else rows
    rows = []
    n = 0
    for pth in paths:
        with gzip.open(pth, "rt", encoding="utf-8") as f:
            for line in f:
                if not cap or len(rows) < cap:
                    rows.append(json.loads(line))
                else:
                    j = rng.randrange(n + 1)
                    if j < cap:
                        rows[j] = json.loads(line)
                n += 1
    return rows


def main():
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rerank jsonl.gz; comma-separate several "
                    "to pool them (e.g. a fresh build appended to the existing one)")
    ap.add_argument("--model", default="Alibaba-NLP/gte-reranker-modernbert-base")
    ap.add_argument("--out", default="/root/out/rerank_gte")
    ap.add_argument("--deadline-h", type=float, default=5.0)
    ap.add_argument("--pair-batch", type=int, default=48, help="~(state,cand) pairs per fwd")
    ap.add_argument("--accum", type=int, default=8, help="record-groups per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--max-samples", type=int, default=800000)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (slower, less mem)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cap-matchup", type=int, default=0,
                    help="reservoir size PER (deck, opponent) -- balances the 3,683 matchups")
    ap.add_argument("--cap-deck", type=int, default=0,
                    help="rows PER DECK after the matchup reservoir; -1 = equalise to the "
                         "smallest deck. Needed with --sides winner, where a matchup cap "
                         "cannot balance (a losing matchup has too few winning rows to fill "
                         "it) and the per-deck spread reaches 2.6x")
    ap.add_argument("--drop-deck", type=float, default=0.0,
                    help="P(hide DECK[...]) per record. DECK[] pins our deck exactly, so "
                         "`ID ME d_x` is redundant and gets NO gradient however correct the "
                         "label is; hiding one forces the model to read the other")
    ap.add_argument("--drop-id", type=float, default=0.0, help="P(hide `ID ME d_x a_y`)")
    ap.add_argument("--sample-seed", type=int, default=1234,
                    help="reservoir seed. Change it to draw a DIFFERENT sample at the same "
                         "--cap-matchup, i.e. add fresh records without unbalancing the mix")
    ap.add_argument("--eval-file", default="",
                    help="json list of held-out rows (tools/ablate_rerank.py --cache writes "
                         "one). Pins the eval set across runs and removes it from training")
    ap.add_argument("--emb-lr-mult", type=float, default=1.0,
                    help="LR multiplier for the token embedding. After a semantic re-init "
                         "(tools/init_domain_embeddings.py) the 3,087 domain rows must move "
                         "a long way while the body is already fitted -- so scale them apart")
    args = ap.parse_args()

    def log(m): print(m, flush=True)
    t0 = time.time()
    dev = "cuda"
    resume_seen = 0
    resuming = args.resume and os.path.exists(os.path.join(args.out, "config.json"))
    if resuming:
        # saved tokenizer already carries the domain tokens; model already resized
        tok = AutoTokenizer.from_pretrained(args.out)
        tok.truncation_side = "left"
        model = AutoModelForSequenceClassification.from_pretrained(
            args.out, trust_remote_code=True, dtype=torch.bfloat16).to(dev)
        if os.path.exists(os.path.join(args.out, "rr_progress.json")):
            resume_seen = json.load(open(os.path.join(args.out, "rr_progress.json")))["seen"]
        log(f"RESUME from {args.out}, seen {resume_seen}, vocab {len(tok)}")
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        tok.truncation_side = "left"
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.bfloat16).to(dev)
        # DOMAIN TOKENS: c<cardId>/a<attackId>/enums -> ONE token each. Cuts state+candidate
        # length several-fold -> fewer tokens per forward -> faster CPU inference (the reranker
        # re-encodes the state per candidate) + less truncation. No output-vocab bloat (cands
        # are INPUT), so cost is only ~2971 x hidden embedding rows (~2-3 MB @ INT8). Full FT
        # learns the new rows naturally (no grad-mask needed, unlike the tied-embedding decoder).
        from lm.vocab import special_tokens
        n_old = len(tok)
        n_added = tok.add_tokens(special_tokens())
        model.resize_token_embeddings(len(tok))
        log(f"domain tokens: vocab {n_old} -> {len(tok)} ({n_added} genuinely new)")
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()             # O(seq^2) global attn -> trade compute for memory
    model.train()
    emb = model.get_input_embeddings().weight
    if args.emb_lr_mult != 1.0:
        body = [p for p in model.parameters() if p is not emb]
        opt = AdamW([{"params": body, "lr": args.lr},
                     {"params": [emb], "lr": args.lr * args.emb_lr_mult}],
                    lr=args.lr, weight_decay=0.01)
        log(f"embedding LR x{args.emb_lr_mult} ({args.lr * args.emb_lr_mult:.2e})")
    else:
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    rows = read_rows(args.data, args.max_samples, seed=args.sample_seed,
                     cap_matchup=args.cap_matchup, cap_deck=args.cap_deck)
    random.Random(0).shuffle(rows)
    if args.eval_file:
        # A FIXED held-out set, so a continuation run on MORE data is still comparable and
        # still clean: rows[:eval_n] of a bigger pool is a different set, and some of it
        # would have been trained on last time.
        eval_rows = json.load(open(args.eval_file))
        held = {row_key(r) for r in eval_rows}
        train_rows = [r for r in rows if row_key(r) not in held]
        log(f"fixed eval {len(eval_rows)} rows from {args.eval_file}; "
            f"dropped {len(rows) - len(train_rows)} overlapping from train")
    else:
        eval_rows, train_rows = rows[:args.eval_n], rows[args.eval_n:]
    log(f"data: {len(train_rows)} train / {len(eval_rows)} eval  "
        f"({time.time()-t0:.0f}s)  pairs/rec avg "
        f"{statistics.mean(len(r['candidates']) for r in rows[:5000]):.1f}")

    from lm.serialize import mask_segments

    def _state(r, aug):
        if not aug:
            return r["state"]
        dd = args.drop_deck and random.random() < args.drop_deck
        di = args.drop_id and random.random() < args.drop_id
        return mask_segments(r["state"], drop_deck=dd, drop_identity=di) if (dd or di) \
            else r["state"]

    def score_batch(records, aug=False, mask=None):
        """Return list of per-record score tensors (one scalar per candidate).

        ``aug`` applies the training-time segment dropout; ``mask`` forces a fixed ablation
        (used by evaluate to report how much the model leans on each identity segment)."""
        pairs, owner = [], []
        for ri, r in enumerate(records):
            st = mask_segments(r["state"], **mask) if mask else _state(r, aug)
            for c in r["candidates"]:
                pairs.append([st, c]); owner.append(ri)
        # truncation_side is set to "left" once at tokenizer construction so an overflowing
        # state loses its HEAD, matching lm/rerank_scorer.py. Inert at the current format
        # (max pair 584 vs --max-len 640) but train and deploy must not disagree on the rule.
        enc = tok(pairs, padding=True, truncation="only_first", max_length=args.max_len,
                  return_tensors="pt").to(dev)
        logits = model(**enc).logits.squeeze(-1)          # [n_pairs]
        out = [[] for _ in records]
        for k, ri in enumerate(owner):
            out[ri].append(logits[k])
        return [torch.stack(o) for o in out]

    def listwise_loss(records):
        scores = score_batch(records, aug=True)
        losses = []
        for r, s in zip(records, scores):
            losses.append(torch.nn.functional.cross_entropy(
                s.unsqueeze(0).float(), torch.tensor([r["chosen"]], device=dev)))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def evaluate(k=None, mask=None):
        model.eval(); tl = n = top1 = 0
        ev = eval_rows[:k] if k else eval_rows
        i = 0
        while i < len(ev):
            grp, npairs = [], 0
            while i < len(ev) and npairs < args.pair_batch:
                grp.append(ev[i]); npairs += len(ev[i]["candidates"]); i += 1
            scores = score_batch(grp, mask=mask)
            for r, s in zip(grp, scores):
                tl += float(torch.nn.functional.cross_entropy(
                    s.unsqueeze(0).float(), torch.tensor([r["chosen"]], device=dev)))
                top1 += int(s.argmax().item() == r["chosen"]); n += 1
        model.train(); return tl / max(1, n), 100.0 * top1 / max(1, n)

    def ablate(k=500):
        """How much does the model actually USE each way of naming our deck? A segment the
        model ignores costs nothing to hide -- that is the measurement, not an argument."""
        out = []
        for nm, mk in (("full", None), ("-DECK[]", {"drop_deck": True}),
                       ("-ID ME", {"drop_identity": True}),
                       ("-both", {"drop_deck": True, "drop_identity": True})):
            out.append(f"{nm} {evaluate(k, mk)[1]:.1f}%")
        return "  ".join(out)

    def save(seen):
        model.save_pretrained(args.out); tok.save_pretrained(args.out)
        json.dump({"seen": seen}, open(os.path.join(args.out, "rr_progress.json"), "w"))

    l0, a0 = evaluate(500)
    log(f"eval(base) loss {l0:.3f}  top1 {a0:.1f}%")
    log(f"ablation(base) {ablate()}")

    def length_batches(rows, chunk=8192):
        """Group records into ~pair_batch-sized batches of SIMILAR length.

        The tokenizer pads each batch to its longest pair, and pair length varies 312 (p50)
        to 606 (max), so random batching spends 46% of its compute on padding (measured:
        padded/real 1.46x at pair-batch 256). Sorting inside a shuffled chunk and then
        shuffling the BATCHES keeps the optimizer's view close to i.i.d. while collapsing the
        padding -- length correlates with turn and deck, so sorting the WHOLE list instead
        would hand the optimizer a curriculum of one deck at a time."""
        out = []
        for c0 in range(0, len(rows), chunk):
            block = sorted(rows[c0:c0 + chunk], key=lambda r: len(r["state"]))
            j = 0
            while j < len(block):
                grp, npairs = [], 0
                while j < len(block) and npairs < args.pair_batch:
                    grp.append(block[j]); npairs += len(block[j]["candidates"]); j += 1
                out.append(grp)
        return out

    deadline = args.deadline_h * 3600
    seen = 0; step = 0; opt.zero_grad(); last_ckpt = time.time(); losses = []; n_ckpt = 0
    i = 0
    batches = length_batches(train_rows)
    random.Random(0).shuffle(batches)
    log(f"length-bucketed into {len(batches)} batches "
        f"(~{sum(len(b) for b in batches)/max(1,len(batches)):.1f} records each)")
    while True:
        grp = batches[i]; i += 1
        if i >= len(batches):
            i = 0
            random.Random(step).shuffle(train_rows)
            batches = length_batches(train_rows)
            random.Random(step).shuffle(batches)
        if seen < resume_seen:                            # fast-forward on resume
            seen += len(grp); continue
        loss = listwise_loss(grp) / args.accum
        loss.backward(); losses.append(float(loss) * args.accum)
        seen += len(grp); step += 1
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if step % 100 == 0:
            el = time.time() - t0
            log(f"  step {step} seen {seen} loss {statistics.mean(losses[-100:]):.3f} "
                f"{el:.0f}s ({seen/max(1,el):.1f} rec/s)")
        if time.time() - last_ckpt > 900:
            vl, va = evaluate(500); save(seen)
            log(f"  [eval] loss {vl:.3f} top1 {va:.1f}%  [ckpt] {time.time()-t0:.0f}s")
            n_ckpt += 1
            if n_ckpt % 4 == 0:                  # 4x an eval -- watch the trend, not each tick
                log(f"  [ablation] {ablate(300)}")
            last_ckpt = time.time()
        if time.time() - t0 > deadline or seen >= args.max_samples:
            break

    vl, va = evaluate(); save(seen)
    log(f"FINAL eval loss {vl:.3f}  top1 {va:.1f}%  seen {seen}  steps {step}")
    log(f"FINAL ablation {ablate(min(2000, args.eval_n))}")
    log("RR_TRAIN_DONE")


if __name__ == "__main__":
    main()
