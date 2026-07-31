"""SFT (completion-only loss) for the LM agent, then IMMEDIATELY play it against the
heuristic engine and report win rate. Designed to run on a vast.ai GPU box.

Loss (revised, agreed): mask the PROMPT (labels=-100), compute loss ONLY on the target
(the chosen option) + EOS -- the model learns the DECISION, not to reproduce the board.

After training, the trained model is wrapped as a ScoringModel and dropped into the SAME
lm.agent adapter used at submission time: every real decision is made by scoring the
legal candidates and taking the argmax (single-pick argmax; multi-pick one-at-a-time).
The LM agent then plays full games vs engine_v2 (the shipped heuristic) via arena.play,
and the win rate is printed + saved.

Usage (on vast.ai, from the repo root):
    python tools/sft_train_eval.py --data /path/v31_full.jsonl.gz \
        --deadline-h 6 --eval-decks mega_lucario,alakazam_nz_fez,crustle_stall \
        --eval-games 30
    # eval only from a saved adapter:
    python tools/sft_train_eval.py --skip-train --adapter out/lora_adapter --eval-games 30
"""
import argparse, glob, gzip, json, os, random, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_DEFAULT = "Qwen/Qwen3.5-0.8B-Base"


# --------------------------------------------------------------------------- #
#  Training  (completion-only loss)
# --------------------------------------------------------------------------- #
def _pilot_of(line):
    """Extract the PILOT deck from a raw SFT line's game_id ('pilot__vs__opp#n') via
    string slicing -- 10M json.loads would cost minutes, this is a few finds."""
    k = line.find('"game_id"')
    if k < 0:
        return "?"
    a = line.find('"', k + 9) + 1        # opening quote of the value
    b = line.find('__vs__', a)
    return line[a:b] if (a > 0 and b > a) else "?"


def _read_rows(path, cap=0, seed=0, balance=False, per_deck=0):
    """Stream the file and reservoir-sample raw LINES (parse only survivors). The file is
    ordered by matchup, so head/stride would be biased; reservoir sampling is unbiased.

    balance=True: sample PER PILOT DECK (one reservoir each, cap=per_deck). Row-uniform
    sampling over-represents long-game decks (control/mill emit 10-70x more decision-rows
    than aggro), so a short training horizon leaves fast decks with ~50 rows. Per-deck
    reservoirs give every deck ~equal coverage for the same compute."""
    op = gzip.open if path.endswith(".gz") else open
    rng = random.Random(seed)
    if not balance:
        keep, n = [], 0
        with op(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                if not cap or len(keep) < cap:
                    keep.append(line)
                else:
                    j = rng.randrange(n)
                    if j < cap:
                        keep[j] = line
        return [json.loads(l) for l in keep], n
    # balanced: independent reservoir per pilot deck, each capped at per_deck
    res, cnt, n = {}, {}, 0
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            d = _pilot_of(line)
            k = cnt.get(d, 0) + 1
            cnt[d] = k
            r = res.setdefault(d, [])
            if len(r) < per_deck:
                r.append(line)
            else:
                j = rng.randrange(k)
                if j < per_deck:
                    r[j] = line
    keep = [l for r in res.values() for l in r]
    return [json.loads(l) for l in keep], n


def _encode(r, tok, maxlen):
    """prompt masked (-100), loss only on target+EOS. Left-truncate the prompt so the
    MENU (end of the prompt) and the target always survive."""
    pp = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    tt = tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"][:maxlen]
    keep = maxlen - len(tt)
    pp = pp[-keep:] if keep > 0 else []
    return pp + tt, [-100] * len(pp) + tt


def _batch_loss(model, tok, batch, device):
    """Loss for a padded batch, computing logits ONLY at the supervised positions.

    The prompt is ~99% of every sequence and is fully masked (-100), yet a plain
    `model(labels=...)` call still runs lm_head over EVERY position -- B x L x 251k
    floats, which is what pinned us to batch 1. Here we run the backbone, gather the
    handful of positions that actually carry a label, and apply lm_head to those only.
    Right padding (no left-pad) keeps the Gated-DeltaNet / causal-conv1d kernels on the
    path they were validated on."""
    import torch
    import torch.nn.functional as F
    pad = tok.pad_token_id
    L = max(len(ids) for ids, _ in batch)
    B = len(batch)
    inp = torch.full((B, L), pad, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    bpos, tpos, tgt = [], [], []
    for b, (ids, lab) in enumerate(batch):
        inp[b, :len(ids)] = torch.tensor(ids)
        att[b, :len(ids)] = 1
        for t, y in enumerate(lab):
            if y != -100:                 # token t is predicted from position t-1
                bpos.append(b); tpos.append(t - 1); tgt.append(y)
    inp = inp.to(device); att = att.to(device)
    core = model.base_model.model            # the ForCausalLM under the peft wrapper
    h = core.model(input_ids=inp, attention_mask=att).last_hidden_state
    hs = h[torch.tensor(bpos, device=device), torch.tensor(tpos, device=device)]
    logits = core.lm_head(hs).float()
    return F.cross_entropy(logits, torch.tensor(tgt, device=device)), len(tgt)


def _make_batches(rows, tok, maxlen, batch_size, rng):
    """Length-bucketed batches: sort a chunk by encoded length so padding is minimal,
    then cut it into batches and shuffle the batch order."""
    enc = [_encode(r, tok, maxlen) for r in rows]
    enc.sort(key=lambda e: len(e[0]))
    batches = [enc[i:i + batch_size] for i in range(0, len(enc), batch_size)]
    rng.shuffle(batches)
    return batches


def train(args, log):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW

    t0 = time.time()
    gpu = torch.cuda.get_device_properties(0)
    log(f"GPU {gpu.name} {gpu.total_memory/1e9:.0f}GB  torch {torch.__version__}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    # --- domain tokenizer extension: c<cardId>/a<attackId>/enums become ONE token each.
    #     Cuts prompt length several-fold (the whole point of the stateless format).
    n_old = model.get_input_embeddings().weight.shape[0]
    from lm.vocab import special_tokens
    toks = special_tokens()
    tok.add_tokens(toks)
    model.resize_token_embeddings(len(tok), mean_resizing=True)   # mean-init new rows
    # ids are NOT contiguous from n_old (some domain strings already exist in the vocab)
    new_ids = sorted({i for i in tok.convert_tokens_to_ids(toks) if i is not None and i >= n_old})
    log(f"tokenizer: {n_old} -> {len(tok)} ({len(new_ids)} genuinely new of {len(toks)} domain strings)")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    _resume_dir = args.adapter if (getattr(args, "resume", False)
                                   and os.path.exists(os.path.join(args.adapter, "adapter_config.json"))) else None
    if _resume_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, _resume_dir, is_trainable=True)
        log(f"RESUME: loaded LoRA adapter from {_resume_dir}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM"))
    for pp in model.parameters():
        if pp.requires_grad:
            pp.data = pp.data.float()

    # --- train ONLY the new embedding rows (old rows frozen via a grad mask).
    #     Embeddings are tied, so this also teaches lm_head to emit the new tokens.
    #     weight_decay MUST be 0 or decoupled decay shrinks the frozen rows at grad=0.
    #  NOTE: keep it bf16 -- casting it to fp32 makes hidden_states fp32 and every
    #  downstream bf16 Linear then dies with "mat1 and mat2 have the same dtype".
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)                      # peft froze it; undo for this one tensor
    row_mask = torch.zeros(emb.shape[0], 1, device=emb.device, dtype=emb.dtype)
    row_mask[new_ids] = 1.0
    emb.register_hook(lambda g: g * row_mask)
    if _resume_dir and os.path.exists(os.path.join(_resume_dir, "new_embeddings.pt")):
        ne = torch.load(os.path.join(_resume_dir, "new_embeddings.pt"), map_location="cpu")
        emb.data[ne["new_ids"]] = ne["rows"].to(emb.device, emb.dtype)   # restore trained rows
        log("RESUME: restored trained embedding rows")

    tr = sum(x.numel() for x in model.parameters() if x.requires_grad)
    log(f"model+LoRA r{args.lora_r} trainable {tr/1e6:.1f}M "
        f"(+{len(new_ids)} embedding rows) in {time.time()-t0:.0f}s")

    rows, n_total = _read_rows(args.data, cap=args.max_samples,
                               balance=args.balance_decks, per_deck=args.per_deck)
    random.Random(0).shuffle(rows)
    if args.balance_decks:
        import collections as _c
        _dist = _c.Counter(r["game_id"].split("__vs__")[0] for r in rows)
        _mc = _dist.most_common()
        log(f"balanced: {len(_dist)} decks, per-deck {_mc[-1][1]}..{_mc[0][1]} "
            f"(min={_mc[-1][0]}, max={_mc[0][0]})")
    log(f"data {os.path.basename(args.data)}: sampled {len(rows)} of {n_total} rows "
        f"in {time.time()-t0:.0f}s")
    eval_data = [_encode(r, tok, args.maxlen) for r in rows[:2000]]
    train_rows = rows[2000:]
    log(f"avg tokens/sample {statistics.mean(len(a) for a,_ in eval_data):.0f} (maxlen {args.maxlen})")

    opt = AdamW([x for x in model.parameters() if x.requires_grad], lr=args.lr, weight_decay=0.0)
    resume_seen = resume_step = 0; resume_elapsed = 0.0
    if _resume_dir and os.path.exists(os.path.join(_resume_dir, "sft_progress.json")):
        pr = json.load(open(os.path.join(_resume_dir, "sft_progress.json")))
        resume_seen, resume_step, resume_elapsed = pr["seen"], pr["step"], pr.get("elapsed", 0.0)
        osp = os.path.join(_resume_dir, "opt_state.pt")
        if os.path.exists(osp):
            opt.load_state_dict(torch.load(osp, map_location=model.device))
        log(f"RESUME: step {resume_step}, seen {resume_seen}, elapsed {resume_elapsed:.0f}s")

    @torch.no_grad()
    def ev(k=400):
        model.eval(); tl = tn = 0
        for ids, lab in eval_data[:k]:
            x = torch.tensor([ids], device=model.device); y = torch.tensor([lab], device=model.device)
            loss = model(input_ids=x, labels=y).loss
            nt = sum(1 for t in lab if t != -100); tl += float(loss) * nt; tn += nt
        model.train(); return tl / max(tn, 1)

    def save(tag):
        model.save_pretrained(args.adapter); tok.save_pretrained(args.adapter)
        # the trained embedding rows live OUTSIDE the LoRA adapter -- save them too
        torch.save({"new_ids": new_ids, "rows": emb.data[new_ids].to(torch.bfloat16).cpu()},
                   os.path.join(args.adapter, "new_embeddings.pt"))
        # RESUME state: optimizer + progress, so a killed run continues from here
        torch.save(opt.state_dict(), os.path.join(args.adapter, "opt_state.pt"))
        json.dump({"seen": seen, "step": step, "elapsed": resume_elapsed + (time.time() - t0)},
                  open(os.path.join(args.adapter, "sft_progress.json"), "w"))
        log(f"[ckpt {tag}] {time.time()-t0:.0f}s")

    log(f"eval(base) loss/token {ev():.3f}")
    model.train(); losses = []; step = resume_step; opt.zero_grad(); last_ckpt = time.time()
    dt0 = time.time() - t0                         # exclude load/eval time from samp/s
    deadline = args.deadline_h * 3600
    rng = random.Random(1)
    seen = resume_seen; stop = False
    chunk = args.batch * 512                       # bucket window: sort ~512 batches at a time
    for c0 in range(0, len(train_rows), chunk):
        if stop:
            break
        if c0 + chunk <= resume_seen:              # RESUME: chunk finished before the kill
            continue
        for bi, batch in enumerate(_make_batches(train_rows[c0:c0 + chunk], tok,
                                                 args.maxlen, args.batch, rng)):
            loss, _ = _batch_loss(model, tok, batch, model.device)
            (loss / args.accum).backward(); losses.append(float(loss))
            seen += len(batch)
            if (bi + 1) % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = args.lr * min(1.0, step / max(1, args.warmup))
                opt.step(); opt.zero_grad(); step += 1
                if step % 50 == 0:
                    el = time.time() - t0
                    log(f"  step {step} seen {seen} loss {sum(losses[-200:])/len(losses[-200:]):.3f} "
                        f"{el:.0f}s ({seen/max(1,el-dt0):.1f} samp/s)")
            if time.time() - last_ckpt > 1800:
                log(f"  [eval] {ev():.3f} @step {step}"); save(f"s{step}"); last_ckpt = time.time()
            if time.time() - t0 + resume_elapsed > deadline:
                log(f"DEADLINE at seen {seen}"); stop = True; break
    log(f"FINAL eval loss/token {ev():.3f}  steps {step}  epoch {seen/max(1,len(train_rows)):.2f}")
    save("final")
    return model, tok


# --------------------------------------------------------------------------- #
#  ScoringModel  (score(prompt, candidates) -> per-candidate mean-token logprob)
# --------------------------------------------------------------------------- #
class ScoringModel:
    """Wraps the trained model for lm.agent: length-normalized log-prob of each
    candidate's tokens given the prompt (teacher forcing), batched over candidates."""
    def __init__(self, model, tok, maxlen):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.tok = tok
        self.maxlen = maxlen
        self.device = model.device

    def score(self, prompt, candidates, obs=None):
        torch = self.torch
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = [self.tok(c, add_special_tokens=False)["input_ids"] or [self.tok.eos_token_id]
                 for c in candidates]
        cap = self.maxlen - max(len(c) for c in c_ids)
        if cap > 0 and len(p_ids) > cap:
            p_ids = p_ids[-cap:]                       # keep the tail (menu + board)
        seqs = [p_ids + c for c in c_ids]
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        inp = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        inp = inp.to(self.device); att = att.to(self.device)
        # EFFICIENCY: gather ONLY the candidate positions before lm_head (avoid B x L x 260k
        # vocab softmax; same trick as _batch_loss). Backbone runs once for the batch.
        core = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        start = len(p_ids)
        bpos, ppos = [], []
        for i, c in enumerate(c_ids):
            for t in range(start, start + len(c)):
                bpos.append(i); ppos.append(t - 1)
        with torch.no_grad():
            h = core.model(input_ids=inp, attention_mask=att).last_hidden_state
            bt = torch.tensor(bpos, device=self.device); pt = torch.tensor(ppos, device=self.device)
            lp = torch.log_softmax(core.lm_head(h[bt, pt]).float(), -1)
            tgt = inp[bt, pt + 1]
            tok_lp = lp[torch.arange(len(bpos), device=self.device), tgt]
        out = []; k = 0
        for i, c in enumerate(c_ids):
            out.append(float(tok_lp[k:k + len(c)].sum()) / max(1, len(c))); k += len(c)   # length-normalized
        return out


# --------------------------------------------------------------------------- #
#  Eval  (LM agent vs heuristic engine_v2 -> win rate)
# --------------------------------------------------------------------------- #
def evaluate(scoring_model, decks, opp_decks, games, log):
    import arena, library
    from battle_log import load_agent
    from lm.agent import make_lm_agent
    tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    results = {}
    for d in decks:
        dl = library.read_deck(d)
        lm_agent = make_lm_agent(dl, profile=tun.get(d, {}), model=scoring_model)
        for opp in (opp_decks or [d]):               # default: mirror (LM vs heuristic, same deck)
            ol = library.read_deck(opp); oa = load_agent(opp)
            w = n = 0
            for g in range(games):
                mine = g % 2
                r = (arena.play(lm_agent, oa, dl, ol) if mine == 0
                     else arena.play(oa, lm_agent, ol, dl))
                n += 1; w += (r == mine)
            wr = 100 * w / max(1, n)
            results[f"{d} (LM) vs {opp} (heuristic)"] = {"win": w, "games": n, "win_rate": wr}
            log(f"  {d:20} (LM) vs {opp:20} (heuristic): {w}/{n} = {wr:.1f}%")
    ov = [v for v in results.values()]
    overall = 100 * sum(v["win"] for v in ov) / max(1, sum(v["games"] for v in ov))
    log(f"OVERALL LM win rate vs heuristic: {overall:.1f}%")
    return results, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="", help="SFT jsonl(.gz) from build_sft (v31_full)")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--adapter", default=os.path.join(ROOT, "out", "lora_adapter"))
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--balance-decks", action="store_true",
                    help="per-pilot-deck reservoirs (each capped at --per-deck) so fast "
                         "decks aren't drowned out by long-game decks in a short horizon")
    ap.add_argument("--per-deck", type=int, default=2000,
                    help="rows kept per pilot deck when --balance-decks")
    ap.add_argument("--max-samples", type=int, default=400000,
                    help="reservoir-sample this many rows (0 = all; 9M would OOM)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8, help="length-bucketed batch size")
    ap.add_argument("--accum", type=int, default=2, help="effective batch = batch * accum")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--deadline-h", type=float, default=6.0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint already in --adapter (LoRA + embedding rows + optimizer + progress); chunk-granular")
    ap.add_argument("--skip-train", action="store_true", help="eval only, load --adapter")
    ap.add_argument("--eval-decks", default="mega_lucario,alakazam_nz_fez,crustle_stall")
    ap.add_argument("--eval-opp", default="", help="comma list; empty = mirror (same deck)")
    ap.add_argument("--eval-games", type=int, default=30)
    args = ap.parse_args()

    LOG = []
    def log(s): print(s, flush=True); LOG.append(str(s))

    import torch
    if args.skip_train:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        # tokenizer + domain vocab: the SFT adapter carries them, but an RL policy adapter
        # (fresh zero-init on an ALREADY-merged base, e.g. the gate's policy_init) does NOT --
        # then the domain tokens live in --model. Prefer the adapter's tokenizer, fall back to
        # the model's, and only resize when the vocab actually differs.
        _tok_src = (args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json"))
                    else args.model)
        tok = AutoTokenizer.from_pretrained(_tok_src)
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        if len(tok) != base.get_input_embeddings().weight.shape[0]:
            base.resize_token_embeddings(len(tok), mean_resizing=False)
        ne = os.path.join(args.adapter, "new_embeddings.pt")
        if os.path.exists(ne):
            blob = torch.load(ne, map_location="cpu")
            w = base.get_input_embeddings().weight
            w.data[blob["new_ids"]] = blob["rows"].to(w.device, w.dtype)
            log(f"restored {len(blob['new_ids'])} trained embedding rows")
        model = PeftModel.from_pretrained(base, args.adapter).eval()
    else:
        assert args.data, "--data required for training"
        model, tok = train(args, log)
    model.config.use_cache = True

    sm = ScoringModel(model, tok, args.maxlen)
    log("\n=== LM (scoring) vs heuristic engine_v2 ===")
    results, overall = evaluate(sm, args.eval_decks.split(","),
                                [x for x in args.eval_opp.split(",") if x],
                                args.eval_games, log)
    os.makedirs(os.path.dirname(args.adapter) or ".", exist_ok=True)
    json.dump({"results": results, "overall_win_rate": overall},
              open(os.path.join(os.path.dirname(args.adapter) or ".", "eval_winrate.json"), "w"), indent=2)
    open(os.path.join(os.path.dirname(args.adapter) or ".", "train_eval_log.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
