"""SFT the Qwen3.5-9B-Base teacher on engine_v2 demonstrations, via Unsloth bf16 LoRA.

Follows Unsloth's published Qwen3.5 fine-tuning recipe (FastLanguageModel, load_in_16bit,
explicit target_modules, max_seq_length passed to get_peft_model too, adamw_8bit, warmup_steps,
dataset_num_proc=1) and deviates only where this task requires it -- each deviation is annotated.

NOT 4-bit: Unsloth's guide says "It is not recommended to do QLoRA (4-bit) training on the
Qwen3.5 models", and bf16 LoRA for 9B needs ~22 GB against this card's 48 GB.

STACK REQUIREMENT (learned the hard way). Qwen3.5's linear-attention layers only take the fast
path when ALL FOUR of these import (transformers/models/qwen3_5/modeling_qwen3_5.py:205):
`causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`,
`fused_recurrent_gated_delta_rule` -- i.e. BOTH `causal-conv1d` AND `flash-linear-attention`.
fla's cpp extensions additionally need torch >= 2.11. On a CUDA-12.8 driver that means
torch==2.11.0+cu128 specifically (the default `pip install unsloth` pulls a cu13 build the driver
cannot run). Without the fast path the DeltaNet layers fall back to a pure-torch recurrence and a
single step did not finish in 3.5 minutes.

Data: tools/_legacy_decoder/build_sft.py --target-mode index, so the completion is the chosen
option's MENU INDEX ("0".."51"), 1-2 tokens instead of 7-10 for the action string. Verified on all
3,088,497 records that menu[index] == the action string and prompts are byte-identical to the
action-target build.

Prompt/EOS contract: the data is a RAW COMPLETION format, which is why -Base is used -- no chat
template is applied anywhere, and the `prompt` string is exactly what lm/serialize renders at
inference. Loss is completion-only.

Smoke:  --limit 2000 --steps 20
Real:   --limit 200000 --epochs 1
"""
import argparse
import collections
import gzip
import json
import os
import re
import time

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")   # keep fused CE; 248k-vocab logits are 1.3 GB

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# One dedicated token per menu index, so EVERY answer is exactly one token.
#
# Written as digits, an index of 10 or more costs two tokens ("14" -> "1","4"), and measured on
# the training mix that is 15.5% of all decisions -- the many-option ones, i.e. the hard ones.
# Three things follow from that, and all three are fixed by a single-token answer:
#   * eval_top1 could only score decisions with <= 10 options, so 16% of the held-out set went
#     unmeasured and the reported top1 was an average over the EASY subset;
#   * inference has to walk a trie of digit prefixes -- measured at 1.27 forward passes per
#     decision (tools/instance/eval_teacher.py), against 1.00 here;
#   * the loss on a 2-digit answer is split over a first digit that only narrows the range.
# The menu in the PROMPT keeps its plain digits, so the prompt format is untouched and this
# stays comparable with v39 and with the reranker; only the answer alphabet changes.
#
# 128 covers the widest menu observed (62 options, max index 49) with room to spare, and
# `index_completion` refuses rather than truncating if a menu ever exceeds it.
N_INDEX_TOKENS = 128
INDEX_TOKENS = ["<i%d>" % k for k in range(N_INDEX_TOKENS)]


def load_action_vocab(path):
    """The frozen action-token list a model is trained and served with. -> (list, set)"""
    d = json.load(open(path))
    toks = d["tokens"]
    print("[action] %d tokens from %s (built on %d decisions)"
          % (len(toks), path, d.get("decisions", -1)), flush=True)
    return toks, set(toks)


def option_texts(prompt):
    """The rendered menu, as option strings in menu order."""
    opts = _RE_OPT.findall(prompt.rsplit(":: ", 1)[-1])
    if [int(i) for i, _ in opts] != list(range(len(opts))):
        return None
    return [t for _, t in opts]


def index_completion(t):
    k = int(t)
    if not 0 <= k < N_INDEX_TOKENS:
        raise SystemExit("[data] target index %d has no single-token form (only 0..%d exist). "
                         "Raise N_INDEX_TOKENS -- do NOT fall back to digits, or part of the "
                         "data would silently train on a different answer alphabet."
                         % (k, N_INDEX_TOKENS - 1))
    return INDEX_TOKENS[k]


def load_pairs(path, limit, skip=0, index_tokens=False, action_vocab=None,
               card_first=None, scheme_b=False):
    """`skip` records are dropped first, so the held-out slice never overlaps training.

    With `action_vocab` the label is the token naming the ACT (`A|attach|c6@BENCH1`) instead of
    the menu position. A record whose chosen option is outside the frozen vocabulary is DROPPED
    rather than relabelled: the model cannot emit that token, so training on it would teach a
    target it can never produce. Measured at 0.010% of decisions, so the loss of data is
    immaterial -- but it is counted and reported, because a large number here would mean the
    vocabulary was built from the wrong file.
    """
    from lm.action_token import (action_token, first_token, sub_index, SUB_TOKENS,
                                 to_scheme_b, label_b, groups)
    P, C = [], []
    dropped = forced = 0
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            d = json.loads(line)
            t = d.get("target")
            if not t:
                continue
            if card_first is not None:
                opts = option_texts(d["prompt"])
                k = int(t)
                if opts is None or k >= len(opts):
                    dropped += 1
                    continue
                ft = first_token(opts[k])
                if ft not in card_first:
                    dropped += 1
                    continue
                if scheme_b:
                    # Forced decisions are dropped. After collapsing equivalent acts, 8.82% of
                    # records offer exactly ONE act -- overwhelmingly "choose a face-down prize"
                    # (16,950) and "there is nothing left to do but end" (10,985). The answer is
                    # not a choice, so the record teaches nothing, spends 8.8% of the budget, and
                    # adds ~8.8pp of free correctness to every accuracy number. At inference the
                    # LM need not be called for them at all.
                    #
                    # This makes scheme-B accuracy NOT comparable with scheme-A's: A counts these
                    # as wins and B does not measure them.
                    g_ = groups(opts, d["prompt"])
                    if len(g_) == 1 and len(g_[0][1]) == 1:
                        forced += 1
                        continue
                    # The prompt is REWRITTEN here rather than in a converted copy of the pool:
                    # one source of truth, and no chance of training on a file that was built
                    # from a different version of the renderer.
                    P.append(to_scheme_b(d["prompt"]))
                    a_, b_ = label_b(d["prompt"], k, opts)
                    C.append(a_ + (b_ or ""))
                    if limit and len(P) >= limit:
                        break
                    continue
                si = sub_index(d["prompt"], opts, k)
                c = ft if si is None else ft + SUB_TOKENS[si]
            elif action_vocab is not None:
                opts = option_texts(d["prompt"])
                k = int(t)
                if opts is None or k >= len(opts):
                    dropped += 1
                    continue
                tok = action_token(opts[k])
                if tok not in action_vocab:
                    dropped += 1
                    continue
                c = tok
            else:
                c = index_completion(t) if index_tokens else t
            P.append(d["prompt"])
            C.append(c)
            if limit and len(P) >= limit:
                break
    if forced:
        print("[data] skipped %d forced decisions (%.2f%%): only one act was available"
              % (forced, 100.0 * forced / max(1, forced + len(P))), flush=True)
    if dropped:
        print("[data] dropped %d records (%.3f%%) whose chosen option is outside the frozen "
              "action vocabulary" % (dropped, 100.0 * dropped / max(1, dropped + len(P))),
              flush=True)
    return {"prompt": P, "completion": C}


_RE_OPT = re.compile(r"(?:^| )(\d+)=(\S+)")


def n_options(prompt):
    """How many numbered options the rendered menu offers."""
    menu = prompt.rsplit(":: ", 1)[-1]
    return len(re.findall(r"(?:^| )(\d+)=", menu))


_RE_IDX_TOK = re.compile(r"^<i(\d+)>$")


def answer_index(completion):
    """The menu index a completion names, whichever answer alphabet produced it. -> int or None"""
    m = _RE_IDX_TOK.match(completion)
    if m:
        return int(m.group(1))
    return int(completion) if completion.isdigit() else None


def eval_top1(model, tok, torch, pairs, maxlen, bsz=16, index_tokens=False,
              action_vocab=None, card_first=None, scheme_b=False):
    """Top-1 on held-out decisions, scored the way inference will score.

    The whole decision is a single argmax over the tokens that name a legal option -- which is
    also the deployment contract: constrain to legal indices, so an illegal move is impossible.

    With `index_tokens` every answer is one token, so EVERY decision is scored. Without them the
    answer alphabet is digits, an index of 10 or more needs two tokens, and those decisions are
    skipped rather than scored by a first-digit-only rule that would be simply wrong. That skip
    covered 16% of the held-out set, so the two numbers are NOT comparable: the digit version
    reports an average over the easy, few-option subset.
    """
    model.eval()
    # Qwen3.5 is a VLM, so `tok` is a PROCESSOR: a positional call makes the first argument
    # `images`, and our prompts came back as "Incorrect image source ... Got 0". Use the
    # underlying text tokenizer.
    tk = getattr(tok, "tokenizer", tok)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"          # last position must be the answer slot
    idx_tok = {}
    if index_tokens:
        for k, t in enumerate(INDEX_TOKENS):
            i = tk.convert_tokens_to_ids(t)
            if i is None or i == tk.unk_token_id:
                raise SystemExit("[eval] %r is not in the tokenizer -- the index tokens were "
                                 "never added, so the targets and the scoring disagree." % t)
            idx_tok[k] = i
    else:
        for k in range(10):
            ids = tk(str(k), add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                idx_tok[k] = ids[0]
    ok = tot = skipped = 0
    by_n = collections.defaultdict(lambda: [0, 0])   # menu size -> (hits, total)
    by_k = collections.defaultdict(lambda: [0, 0])   # answer bucket -> (hits, total)
    by_kind = collections.defaultdict(lambda: [0, 0])  # action kind -> (hits, total)
    if action_vocab is not None:
        from lm.action_token import action_token
        counts = action_vocab if isinstance(action_vocab, dict) else {}
    if card_first is not None:
        return eval_card_first(model, tok, tk, torch, pairs, maxlen, bsz, card_first,
                               scheme_b)
    P, C = pairs["prompt"], pairs["completion"]
    with torch.no_grad():
        for i in range(0, len(P), bsz):
            bp, bc = P[i:i + bsz], C[i:i + bsz]
            if action_vocab is None:
                gold = [answer_index(c) for c in bc]
                keep = [j for j in range(len(bp))
                        if gold[j] is not None and gold[j] in idx_tok
                        and (index_tokens or n_options(bp[j]) <= 10)]
            else:
                gold = list(bc)                       # the label IS the token
                keep = [j for j in range(len(bp)) if tk.convert_tokens_to_ids(bc[j]) is not None]
            skipped += len(bp) - len(keep)
            if not keep:
                continue
            texts = [bp[j] for j in keep]
            enc = tk(texts, return_tensors="pt", padding=True, truncation=True,
                     max_length=maxlen).to(model.device)
            out = model(**enc).logits[:, -1, :].float()
            for r, j in enumerate(keep):
                n = n_options(bp[j])
                if action_vocab is None:
                    cand = [(k, idx_tok[k]) for k in range(n) if k in idx_tok]
                else:
                    # Score the tokens naming the LEGAL options, deduplicated: two options that
                    # share a token are the same act, so picking either is correct and they must
                    # not compete with each other.
                    opts = option_texts(bp[j]) or []
                    seen = {}
                    for o in opts:
                        t_ = action_token(o)
                        i_ = tk.convert_tokens_to_ids(t_)
                        if i_ is not None and t_ not in seen:
                            seen[t_] = i_
                    cand = list(seen.items())
                if not cand:
                    skipped += 1
                    continue
                best = max(cand, key=lambda kt: out[r, kt[1]].item())[0]
                hit = int(best == gold[j])
                ok += hit
                tot += 1
                by_n[min(n, 24)][0] += hit
                by_n[min(n, 24)][1] += 1
                # index scheme: how far down the menu. action scheme: how often the correct
                # token was seen in training -- the long tail is where an action vocabulary is
                # supposed to hurt, so it is bucketed by exactly that.
                b = (min(gold[j] // 4, 6) if action_vocab is None
                     else min(4, len("%d" % max(1, counts.get(gold[j], 0)))))
                by_k[b][0] += hit
                by_k[b][1] += 1
                if action_vocab is not None:
                    # Per action kind, because the weaknesses on record are per kind, not
                    # global: energy attachment sits at chance while everything else is
                    # +25-79pt (`attach-decisions-at-chance`). A factored <card>+<kind@slot>
                    # answer is the specific remedy for that one, so whether it is worth its
                    # extra forward pass is decided by this row and nothing else.
                    kd = gold[j].split("|")[1] if "|" in gold[j] else "?"
                    by_kind[kd][0] += hit
                    by_kind[kd][1] += 1
    model.train()
    # Is the answer alphabet itself costing anything? An index names a POSITION in the menu, so
    # the model has to look up what sits there -- a pointer step that a content-based output
    # (the card id) would not need. The objection to a content-based output is that the card id
    # alone leaves the action undetermined on 20.2% of decisions (measured; mostly "attach to the
    # active or to the bench"), so the index stays. These two breakdowns are what would show the
    # pointer step failing anyway: accuracy falling as the menu grows, or falling for options far
    # down the list. Flat means the encoding is not the bottleneck and there is nothing to trade.
    if tot:
        print("[eval] by option count: " + "  ".join(
            "%s:%.0f%%(%d)" % ("%d" % n if n < 24 else "24+", 100.0 * o / t, t)
            for n, (o, t) in sorted(by_n.items()) if t >= 20), flush=True)
        lab = ("by answer index: " if action_vocab is None
               else "by how often the correct token was seen in training: ")
        fmt = ((lambda k: "%d-%d" % (4 * k, 4 * k + 3)) if action_vocab is None
               else (lambda k: ("<10", "<100", "<1k", "<10k", ">=10k")[min(k - 1, 4)]))
        print("[eval] " + lab + "  ".join(
            "%s:%.0f%%(%d)" % (fmt(k), 100.0 * o / t, t)
            for k, (o, t) in sorted(by_k.items()) if t >= 20), flush=True)
        if by_kind:
            print("[eval] by action kind: " + "  ".join(
                "%s:%.0f%%(%d)" % (k, 100.0 * o / t, t)
                for k, (o, t) in sorted(by_kind.items(), key=lambda kv: -kv[1][1])
                if t >= 20), flush=True)
    return ok, tot, skipped


def warm_start(model, tk, torch, src, n_base):
    """Continue from an earlier checkpoint whose vocabulary is NOT the same one.

    Scheme B keeps every card and attack token but replaces the 64 <sN> tie-breakers with 69
    K|kind@slot tokens, so the added rows sit at different indices than they did in the run being
    resumed. Copying by position would leave every card token pointing at another card's vector,
    train perfectly well, and be discoverable only from the win rate -- the failure already on
    record for the 9B scorer. Rows are therefore matched BY NAME, and the counts are printed so a
    near-empty restore is visible instead of silent.

    The LoRA is restored separately, from adapter_model.safetensors, and only when the shapes
    line up; a rank or target-module change makes the tensors incompatible and is refused rather
    than partially applied.
    """
    import os
    from safetensors.torch import load_file
    rep = {"emb": 0, "lora": 0, "skipped": 0}

    embp = os.path.join(src, "domain_embeddings.pt")
    if os.path.exists(embp):
        blob = torch.load(embp, map_location="cpu")
        rows, n_src = blob["rows"], blob["n_base"]
        # The name->id map comes from the SAVED TOKENIZER, not from added_tokens.json: a fast
        # tokenizer keeps its added tokens inside tokenizer.json and writes no such file, so
        # reading the file would find nothing, trip the refusal below, and stop the chain.
        from transformers import AutoTokenizer
        old = AutoTokenizer.from_pretrained(src).get_added_vocab()
        w = model.get_input_embeddings().weight
        with torch.no_grad():
            for name, oid in old.items():
                j = oid - n_src
                if not 0 <= j < rows.shape[0]:
                    continue
                nid = tk.convert_tokens_to_ids(name)
                if nid is None or nid < n_base:
                    rep["skipped"] += 1
                    continue
                w[nid] = rows[j].to(w.dtype).to(w.device)
                rep["emb"] += 1
    print("[warm] embedding rows restored by name: %d (%d in the old vocabulary have no home in "
          "the new one)" % (rep["emb"], rep["skipped"]), flush=True)
    if rep["emb"] < 1000:
        raise SystemExit("[warm] REFUSING: only %d rows were restored. The card tokens are the "
                         "whole point of resuming; a restore this small means the checkpoint or "
                         "its added_tokens.json is not the one it claims to be." % rep["emb"])

    ad = os.path.join(src, "adapter_model.safetensors")
    if os.path.exists(ad):
        sd = load_file(ad)
        cur = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in sd.items():
                for c in (k, k.replace("base_model.model.", ""), "base_model.model." + k):
                    if c in cur and cur[c].shape == v.shape:
                        cur[c].copy_(v.to(cur[c].dtype).to(cur[c].device))
                        rep["lora"] += 1
                        break
        print("[warm] LoRA tensors restored: %d of %d" % (rep["lora"], len(sd)), flush=True)
        if rep["lora"] == 0:
            raise SystemExit("[warm] REFUSING: the adapter loaded but nothing matched. Rank or "
                             "target modules differ from this run's; resuming would silently "
                             "train a fresh LoRA while claiming to continue.")
    else:
        print("[warm] no adapter at %s -- embeddings only" % ad, flush=True)
    return rep


def _split_answer(c, scheme_b):
    """A completion string -> (first token, tie token or None), for either scheme."""
    if scheme_b:
        i = c.find("K|")
        return (c[:i], c[i:]) if i > 0 else (c, None)
    i = c.find("<s")
    return (c[:i], c[i:]) if i > 0 else (c, None)


def eval_card_first(model, tok, tk, torch, pairs, maxlen, bsz, counts, scheme_b=False):
    """Top-1 for the card-first answer, scored the way inference will score it.

    Two numbers, because they fail for different reasons and a single average would hide both:
      * FIRST -- argmax over the card tokens of the legal options. This is the decision proper.
      * TIE   -- on the subset that needs a tie-breaker, argmax over <s0..sK> with the correct
                 card token forced. Teacher-forcing the first token is deliberate: it measures
                 the tie-break in isolation instead of compounding it with a first-token miss.
    JOINT is what actually reaches the game.
    """
    from lm.action_token import (first_token, sub_index, tie_group, sort_key, parse_board,
                                 equivalent, SUB_TOKENS, parse_menu_b)
    model.eval()
    sub_ids = [tk.convert_tokens_to_ids(t) for t in SUB_TOKENS]
    P, C = pairs["prompt"], pairs["completion"]
    ok = tot = 0
    tie_ok = tie_tot = 0
    by_kind = collections.defaultdict(lambda: [0, 0])
    by_freq = collections.defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for i in range(0, len(P), bsz):
            bp = P[i:i + bsz]
            enc = tk(bp, return_tensors="pt", padding=True, truncation=True,
                     max_length=maxlen).to(model.device)
            out = model(**enc).logits[:, -1, :].float()
            for r, p in enumerate(bp):
                menu = parse_menu_b(p.rsplit(" :: ", 1)[-1]) if scheme_b else None
                opts = None if scheme_b else option_texts(p)
                if not menu and not opts:
                    continue
                gold_c = C[i + r]
                gold_first = _split_answer(gold_c, scheme_b)[0]
                cand = {}
                heads = [h for h, _ in menu] if scheme_b else [first_token(o) for o in opts]
                for t_ in heads:
                    if t_ not in cand:
                        j = tk.convert_tokens_to_ids(t_)
                        if j is not None:
                            cand[t_] = j
                if not cand:
                    continue
                best = max(cand.items(), key=lambda kv: out[r, kv[1]].item())[0]
                hit = int(best == gold_first)
                ok += hit
                tot += 1
                if scheme_b:
                    sec = _split_answer(gold_c, True)[1]
                    kd = (sec.split("|")[1].split("@")[0] if sec
                          else (gold_first.split("|")[1] if gold_first.startswith("A|") else "-"))
                else:
                    kd = "-"
                    for o in opts:                  # the kind of the GOLD option
                        if first_token(o) == gold_first:
                            kd = o.split(":")[0]
                            break
                by_kind[kd][0] += hit
                by_kind[kd][1] += 1
                b = min(4, len("%d" % max(1, counts.get(gold_first, 0))))
                by_freq[b][0] += hit
                by_freq[b][1] += 1
    # the tie-break, teacher-forced, on the subset that needs one
    idx = [j for j in range(len(P)) if _split_answer(C[j], scheme_b)[1]]
    for i in range(0, len(idx), bsz):
        chunk = idx[i:i + bsz]
        if not chunk:
            break
        texts = [P[j] + _split_answer(C[j], scheme_b)[0] for j in chunk]
        with torch.no_grad():
            enc = tk(texts, return_tensors="pt", padding=True, truncation=True,
                     max_length=maxlen).to(model.device)
            out = model(**enc).logits[:, -1, :].float()
        for r, j in enumerate(chunk):
            gf, gs = _split_answer(C[j], scheme_b)
            if scheme_b:
                secs = [ss for h, ss in parse_menu_b(P[j].rsplit(" :: ", 1)[-1]) if h == gf]
                cand = [(t_, tk.convert_tokens_to_ids(t_)) for t_ in (secs[0] if secs else [])]
                cand = [c for c in cand if c[1] is not None]
                gold_sub = gs
            else:
                opts = option_texts(P[j]) or []
                k = len(tie_group(opts, gf))
                cand = [(n_, sub_ids[n_]) for n_ in range(min(k, len(sub_ids)))
                        if sub_ids[n_] is not None]
                gold_sub = int(gs.split("<s")[1].rstrip(">")) if gs else -1
            if not cand:
                continue
            best = max(cand, key=lambda kv: out[r, kv[1]].item())[0]
            tie_ok += int(best == gold_sub)
            tie_tot += 1
    model.train()
    rate = 100.0 * ok / max(1, tot)
    trate = 100.0 * tie_ok / max(1, tie_tot)
    share = tie_tot / max(1, tot)
    print("[eval] FIRST %d/%d = %.2f%% | TIE %d/%d = %.2f%% (%.1f%% of decisions) | "
          "JOINT approx %.2f%%"
          % (ok, tot, rate, tie_ok, tie_tot, trate, 100.0 * share,
             rate * ((1 - share) + share * trate / 100.0)), flush=True)
    print("[eval] by action kind: " + "  ".join(
        "%s:%.0f%%(%d)" % (k, 100.0 * o / t, t)
        for k, (o, t) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]) if t >= 20), flush=True)
    print("[eval] by how often the card was on a menu: " + "  ".join(
        "%s:%.0f%%(%d)" % (("<10", "<100", "<1k", "<10k", ">=10k")[min(k - 1, 4)],
                           100.0 * o / t, t)
        for k, (o, t) in sorted(by_freq.items()) if t >= 20), flush=True)
    return ok, tot, 0


def add_domain_tokens(model, tok, torch, index_tokens=False, action_tokens=None,
                      extra=None):
    """Extend the tokenizer with the SAME 2,971 domain tokens the reranker uses, and report
    whether the new rows are actually distinct vectors.

    History: on the reranker, 3,087 added tokens ended up at pairwise cosine +0.998 -- every
    card was the same vector to three decimals, so no training could ever have used DECK[].
    That came from initialising new rows at the embedding MEAN. transformers now seeds them
    from a multivariate normal matching the old embeddings' mean and covariance, which should
    give distinct vectors -- but that is the exact thing that failed before, so it is measured
    here rather than assumed.
    """
    import sys
    for p in ("/root/ptcg/repo", "/root/ptcg/repo/cg-lib"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from lm.vocab import special_tokens

    tk = getattr(tok, "tokenizer", tok)
    n_old = len(tk)
    # The index tokens are added HERE rather than in lm.vocab.special_tokens(), because that
    # function defines the reranker's vocabulary too and a checkpoint trained against a different
    # token set cannot be loaded. Only the decoder needs an answer alphabet.
    new = list(special_tokens()) + (INDEX_TOKENS if index_tokens else []) \
        + (list(action_tokens) if action_tokens else []) + (list(extra) if extra else [])
    n_added = tk.add_tokens(new)
    model.resize_token_embeddings(len(tk))
    print("[tokens] vocab %d -> %d (%d genuinely new)" % (n_old, len(tk), n_added), flush=True)
    if n_added == 0:
        return None

    w = model.get_input_embeddings().weight
    # Re-initialise the new rows: RANDOM DIRECTIONS at the base row-norm scale. Measured
    # (tools/instance/probe_init.py) against the base rows' own geometry, mean pairwise cosine:
    #     BASE +0.0117 | default +1.0000 | subtoken +0.8587 | subtoken-centred +0.8583
    #     random-centred -0.0000            <- the only one that matches BASE
    # transformers' default leaves every new row the same near-zero vector (row-norm 0.092 vs
    # 0.855), i.e. the +0.998 collapse that made the reranker unable to read DECK[].
    # Sub-token means do not fix it: every card token shares the pieces "c" and digits, so they
    # share a direction no centring removes -- the earlier "semantic re-init" reached only
    # +0.7096 for the same reason. Random loses the (unusable) semantics and buys a geometry the
    # rows can actually be trained apart from, which is the point of training them at all.
    with torch.no_grad():
        # .cpu(): the generator is seeded on CPU for reproducibility, so the scale must come
        # back from the GPU to meet it
        tgt = w[:n_old].float().norm(dim=1).median().cpu()
        gen = torch.Generator(device="cpu").manual_seed(3407)
        g = torch.randn(n_added, w.shape[1], generator=gen, dtype=torch.float32)
        g = g / g.norm(dim=1, keepdim=True) * tgt
        w[n_old:] = g.to(device=w.device, dtype=w.dtype)
    new = w[n_old:].detach().float()
    nrm = new / new.norm(dim=1, keepdim=True).clamp_min(1e-9)
    # sample 512 rows: the full 2971^2 gram is affordable but the sample is enough and keeps
    # this cheap if the token set grows
    idx = torch.arange(0, new.shape[0], max(1, new.shape[0] // 512))[:512]
    g = nrm[idx] @ nrm[idx].T
    off = g[~torch.eye(len(idx), dtype=torch.bool, device=g.device)]
    base = w[:n_old].detach().float()
    bn = base[::max(1, n_old // 512)][:512]
    bn = bn / bn.norm(dim=1, keepdim=True).clamp_min(1e-9)
    bg = bn @ bn.T
    boff = bg[~torch.eye(len(bn), dtype=torch.bool, device=bg.device)]
    print("[tokens] new-row pairwise cosine  mean %+.4f  max %+.4f   (base rows: mean %+.4f)"
          % (off.mean(), off.max(), boff.mean()), flush=True)
    # Threshold is set from the BASE rows measured in the same run, not a guessed constant.
    # The first version used 0.9 and would have waved through the +0.8177 the smoke produced.
    if off.mean() > boff.mean() + 0.15:
        raise SystemExit("[tokens] REFUSING: new rows sit at mean cosine %.4f against %.4f for "
                         "the base rows. That is the degeneracy that made the reranker unable "
                         "to read DECK[]; training from a collapsed geometry wastes the run."
                         % (off.mean(), boff.mean()))
    return n_old


def unfreeze_new_rows(model, n_base, torch, index_tokens=False):
    """Train ONLY the added embedding rows.

    The input embedding is large (248,320 x 4,096 = 1.02B on the 9B); training all of it would
    cost more than the LoRA it is meant to support, and the base rows do not need to move.
    A gradient hook zeroes every base row, so the optimizer sees gradient only where the new
    tokens are.

    TIED WEIGHTS. On Qwen3.5-9B `tie_word_embeddings` is False, so `lm_head` is a separate
    tensor and freezing it is both safe and right: under the constrained-output rule the model
    only ever emits digits, never a card token. On Qwen3-4B it is TRUE -- lm_head.weight IS
    embed_tokens.weight. Freezing the head there would freeze the input embedding through the
    same tensor and the 2,971 new rows would never receive a gradient, silently: training would
    complete, the run would look normal, and every card token would stay at its random
    initialisation. That is the failure already on record in `domain-token-embedding-degeneracy`,
    so the tensors are compared by identity rather than the config trusted.
    """
    emb = model.get_input_embeddings()
    emb.weight.requires_grad_(True)

    def _mask_grad(grad):
        grad = grad.clone()
        grad[:n_base] = 0
        return grad

    emb.weight.register_hook(_mask_grad)
    head = model.get_output_embeddings()
    tied = head is not None and head.weight.data_ptr() == emb.weight.data_ptr()
    if head is not None and not tied:
        head.weight.requires_grad_(False)
    print("[emb] training rows %d.. (%d of %d) | lm_head %s"
          % (n_base, emb.weight.shape[0] - n_base, emb.weight.shape[0],
             "TIED to the embedding -- left trainable, or the new rows would get no gradient"
             if tied else "frozen (separate tensor)"), flush=True)
    if not emb.weight.requires_grad:
        raise SystemExit("[emb] REFUSING: the input embedding ended up frozen, so the added "
                         "domain tokens cannot learn. Training would finish and mean nothing.")
    if index_tokens and not tied and head is not None:
        # The index tokens are the ONLY thing the model ever emits, and what decides which one it
        # emits is the OUTPUT row, not the input row. Card tokens are the opposite case: they are
        # read from the prompt and never generated, which is why freezing lm_head is right for
        # them. With an untied head (Qwen3.5-9B) the two needs conflict, and freezing the head
        # would leave every answer token at its random initialisation -- the model could not
        # learn to answer at all, while the loss curve looked ordinary.
        raise SystemExit(
            "[emb] REFUSING: --index-tokens with an UNTIED lm_head. The answer tokens are "
            "generated, so they need their OUTPUT rows trained, but lm_head is frozen here to "
            "keep the card tokens cheap. Train the new lm_head rows (with the same masking hook) "
            "before using index tokens on this model.")
    # keep a SAMPLE of base rows (the full 248k x 4096 copy would be 4 GB) to prove afterwards
    # that the hook actually held them
    bidx = torch.arange(0, n_base, max(1, n_base // 4096))[:4096].to(emb.weight.device)
    return (emb.weight[n_base:].detach().float().cpu().clone(),
            (bidx.cpu(), emb.weight[bidx].detach().float().cpu().clone()))


def report_embedding_movement(model, n_base, emb0, torch, out_dir, base0=None):
    """Did the new rows actually LEARN, and did the base rows stay put?

    On the reranker the added rows shifted 0.42% of their norm in a full epoch even with
    --emb-lr-mult 3, i.e. they stayed at their hand-written initialisation. If that repeats
    here, the tokens were added for nothing and the run should not be read as a test of them.
    """
    w = model.get_input_embeddings().weight.detach().float().cpu()
    new = w[n_base:]
    d = (new - emb0).norm(dim=1) / emb0.norm(dim=1).clamp_min(1e-9)
    moved = (d > 1e-6)
    # Rows for tokens that never appeared in the sampled data get no gradient, so the median over
    # ALL rows reads 0.000% and hides whether the ones that DID appear learned anything. Report
    # the share that moved, and the distribution among those.
    print("[emb] new rows that moved: %d/%d (%.1f%%)"
          % (moved.sum(), len(d), 100.0 * moved.sum() / len(d)), flush=True)
    if moved.any():
        dm = d[moved]
        print("[emb]   among them: median %.3f%%  p90 %.3f%%  max %.3f%%  "
              "(the reranker's added rows managed 0.42%% and stayed unlearned)"
              % (100 * dm.median(), 100 * dm.quantile(0.9), 100 * dm.max()), flush=True)
    # The other half of the check: the gradient hook is supposed to hold every BASE row fixed.
    # Nothing so far has verified that, and a leak would mean a 1B-parameter finetune nobody
    # asked for, quietly degrading the base model.
    if base0 is not None:
        bd = (w[base0[0]] - base0[1]).abs().max()
        print("[emb] base rows max abs change %.3e  %s"
              % (bd, "OK (frozen)" if bd < 1e-6 else "*** LEAK: the grad hook is not holding ***"),
              flush=True)
    path = os.path.join(out_dir, "domain_embeddings.pt")
    torch.save({"n_base": n_base, "rows": w[n_base:].clone()}, path)
    print("[emb] saved %d new rows -> %s" % (new.shape[0], path), flush=True)


def strip_trailing_eos(ds, eos_id, name):
    """Remove the end-of-sequence token TRL appends to every completion.

    With a single-token answer there is nothing to terminate: inference runs one forward pass and
    takes an argmax over the legal index tokens, so the position after the answer is never read.
    Trained on, it is half of the two supervised tokens -- and a token that is perfectly
    predictable from the one before it, so the reported loss settles at roughly HALF the real
    decision loss and stops meaning -log P(correct option).

    TRL appends it unconditionally (`add_eos` fires whenever the completion does not already end
    with the eos string) and exposes no switch: setting SFTConfig.eos_token only changes WHICH
    token is appended, which here would append a real index token to every other example. So the
    tokenised dataset is edited after TRL has built it, which also means the edit is verifiable --
    tools/instance/check_loss_mask.py reads the labels the collator actually produces.
    """
    def f(ex):
        if ex["input_ids"] and ex["input_ids"][-1] == eos_id:
            ex["input_ids"] = ex["input_ids"][:-1]
            for k in ("completion_mask", "attention_mask", "assistant_masks"):
                if k in ex and ex[k]:
                    ex[k] = ex[k][:-1]
        return ex

    before = len(ds)
    n_eos = sum(1 for i in range(min(2000, before)) if ds[i]["input_ids"][-1] == eos_id)
    ds = ds.map(f, desc="stripping the trailing EOS from %s" % name)
    print("[eos] %s: trailing EOS present in %d/%d sampled rows -> stripped"
          % (name, n_eos, min(2000, before)), flush=True)
    if n_eos < min(2000, before):
        print("[eos] WARNING: %d sampled rows did NOT end in EOS. They are left as they are, but "
              "a mixed answer format is worth understanding before trusting the run."
              % (min(2000, before) - n_eos), flush=True)
    return ds


def make_trainer_class(SFTTrainer, _unused, torch, group=True, emb_lr_mult=1.0):
    """SFTTrainer that draws length-grouped batches.

    Every batch is padded to its own longest member, so with a random order that length is close
    to the global maximum every time. Measured on this data (tools/instance/measure_lengths.py,
    20k v39 prompts, mean 349 / p90 437 / max 814): a random order costs 1.32x the useful tokens
    at batch 8 and 1.56x at batch 32, i.e. a third to a half of the forward pass is spent on pad.
    Sorting by length drops that to 1.00x.

    trl 0.24 dropped `group_by_length` from SFTConfig, and its replacement `padding_free` needs
    flash-attention varlen, which is not installed here. transformers still ships the sampler the
    old flag used, so it is wired up directly. LengthGroupedSampler shuffles into megabatches of
    50*batch_size and sorts only within them, so the order stays random at the scale that matters
    for optimisation while the padding waste disappears.
    """
    from transformers.trainer_pt_utils import LengthGroupedSampler

    def measure(ds):
        """Sequence lengths, taken from TRL's OWN tokenised dataset.

        Tokenising the prompts a second time to get them cost minutes on 40k rows and would cost
        hours on 2.5M -- a length-grouping optimisation that spends more than the padding it
        saves. The trainer has already produced `input_ids`, and Arrow can report the length of
        every list in one C pass, so this is effectively free.
        """
        try:
            import pyarrow.compute as pc
            return pc.list_value_length(ds.data.column("input_ids")).to_pylist()
        except Exception as e:                     # column missing, or a non-Arrow dataset
            print("[len] arrow path unavailable (%s); measuring row by row" % e, flush=True)
            return [len(x) for x in ds["input_ids"]]

    class TunedSFTTrainer(SFTTrainer):
        def create_optimizer(self):
            """Give the embedding its own learning rate.

            Every answer token is a NEW row starting from a random direction, so unlike the 9B
            run -- whose targets were pre-trained digit tokens -- the output side has to be
            built from nothing. At a shared LoRA learning rate the reranker's added rows moved
            0.42% of their norm in a full epoch and stayed effectively unlearned
            (`domain-token-embedding-degeneracy`), which is the failure this exists to avoid.
            """
            if emb_lr_mult == 1.0 or self.optimizer is not None:
                return super().create_optimizer()
            emb = self.model.get_input_embeddings().weight
            decay, no_decay, embs = [], [], []
            for n_, p_ in self.model.named_parameters():
                if not p_.requires_grad:
                    continue
                (embs if p_ is emb else (no_decay if p_.ndim < 2 else decay)).append(p_)
            cls_, kw = SFTTrainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            groups = [{"params": decay, "weight_decay": self.args.weight_decay},
                      {"params": no_decay, "weight_decay": 0.0},
                      {"params": embs, "weight_decay": 0.0,
                       "lr": self.args.learning_rate * emb_lr_mult}]
            self.optimizer = cls_([g for g in groups if g["params"]], **kw)
            print("[opt] embedding lr %.2e (x%.1f), everything else %.2e"
                  % (self.args.learning_rate * emb_lr_mult, emb_lr_mult,
                     self.args.learning_rate), flush=True)
            return self.optimizer

        def _get_train_sampler(self, train_dataset=None):
            if not group:
                return super()._get_train_sampler(train_dataset)
            ds = train_dataset if train_dataset is not None else self.train_dataset
            lens = measure(ds)
            print("[len] grouping %d rows: mean %.0f max %d"
                  % (len(lens), sum(lens) / max(1, len(lens)), max(lens)), flush=True)
            g = torch.Generator()
            g.manual_seed(self.args.seed)
            return LengthGroupedSampler(
                batch_size=self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps,
                dataset=ds,
                lengths=lens,
                generator=g,
            )

    return TunedSFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-9B-Base")
    ap.add_argument("--data", default="/root/ptcg/repo/data/sft/teacher_0730_index.jsonl.gz")
    ap.add_argument("--out", default="/root/out/teacher9b")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=0, help="max_steps; 0 = use --epochs")
    ap.add_argument("--epochs", type=float, default=1.0)
    # 1024 not the recipe's 2048: measured p99 739 / max 897 tokens on this data, so 2048 would
    # pad-waste half of every batch.
    ap.add_argument("--maxlen", type=int, default=1024)
    # The recipe uses bsz 1 x accum 4 to fit 27B. 9B on 48 GB has room, and our sequences are
    # short; raise it for throughput and keep the same effective batch shape.
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)          # recipe: r=16, alpha=16
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--emb-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for the embedding rows; the answer "
                         "tokens are new random rows and may need to move much further "
                         "than the LoRA does")
    ap.add_argument("--group-by-length", action="store_true",
                    help="length-grouped batches: removes the 32-56%% padding waste measured "
                         "on this data (see make_trainer_class)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="drop gradient checkpointing -- faster, but activations must fit")
    ap.add_argument("--num-proc", type=int, default=1,
                    help="dataset tokenisation workers; the one-off cost is large at 2.4M rows")
    ap.add_argument("--keep-eos", action="store_true",
                    help="keep the EOS TRL appends after the answer (pointless with a "
                         "single-token answer, but here to fall back on)")
    ap.add_argument("--init-from", default="",
                    help="continue from this checkpoint: its embedding rows are "
                         "matched BY NAME and its LoRA copied in, so a vocabulary "
                         "change between runs does not scramble them")
    ap.add_argument("--card-first", default="",
                    help="path from tools/instance/build_cardfirst_vocab.py: answer with the\n"
                         "CARD token, plus a sorted <sN> only where the card does not settle it")
    ap.add_argument("--action-vocab", default="",
                    help="path from tools/instance/build_action_vocab.py: label each decision "
                         "with the token naming the ACT (card + board slot) instead of the "
                         "menu position. Overrides --index-tokens.")
    ap.add_argument("--index-tokens", action="store_true",
                    help="one dedicated token per menu index, so every answer is exactly one "
                         "token (15.5%% of decisions otherwise need two); requires "
                         "--domain-tokens, since the rows are trained by the same mechanism")
    ap.add_argument("--domain-tokens", action="store_true",
                    help="add lm.vocab.special_tokens() and train ONLY those embedding rows")
    ap.add_argument("--eval-n", type=int, default=4000,
                    help="held-out records taken from the FRONT of the file; training skips them")
    a = ap.parse_args()

    t0 = time.time()
    from unsloth import FastLanguageModel               # noqa: E402  (must precede transformers)
    import torch                                        # noqa: E402
    from datasets import Dataset                        # noqa: E402
    from trl import SFTTrainer, SFTConfig               # noqa: E402

    # The DeltaNet fast-path gate applies to Qwen3.5 ONLY. Qwen3 is a plain dense transformer
    # with ordinary attention -- there is no linear-attention kernel to be missing, so running
    # the check would refuse a model that is perfectly fine.
    is_q35 = "qwen3.5" in a.model.lower() or "qwen3_5" in a.model.lower()
    print("[stack] torch %s cuda=%s | model %s" % (torch.__version__, torch.cuda.is_available(),
                                                   a.model), flush=True)
    if is_q35:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M   # noqa: E402
        print("[stack] qwen3_5 FAST PATH = %s" % M.is_fast_path_available, flush=True)
        if not M.is_fast_path_available:
            print("[stack] REFUSING: without the fast path the DeltaNet layers run a pure-torch "
                  "recurrence and a single step takes minutes. Install causal-conv1d + "
                  "flash-linear-attention on torch>=2.11 (cu128 build for a CUDA 12.8 driver).",
                  flush=True)
            raise SystemExit(2)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model,
        max_seq_length=a.maxlen,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    print("[load] %.1fs %s" % (time.time() - t0, type(model).__name__), flush=True)

    cf_set = cf_extra = cf_counts = None
    scheme_b = False
    if a.card_first:
        _cf = json.load(open(a.card_first))
        cf_set = set(_cf["first_tokens"])
        scheme_b = _cf.get("scheme") == "b"
        tie = _cf["second_tokens"] if scheme_b else _cf["sub_tokens"]
        cf_extra = list(_cf["new_tokens"]) + list(tie)
        cf_counts = _cf.get("counts", {})
        print("[cardfirst] scheme %s | %d first tokens (%d need a new row) + %d tie tokens"
              % ("B (act)" if scheme_b else "A (sorted rank)", len(cf_set),
                 len(_cf["new_tokens"]), len(tie)), flush=True)
        if not a.domain_tokens:
            raise SystemExit("--card-first needs --domain-tokens: the card tokens it answers "
                             "with live in lm.vocab.special_tokens().")

    act_list = act_set = act_counts = None
    if a.action_vocab:
        act_list, act_set = load_action_vocab(a.action_vocab)
        act_counts = json.load(open(a.action_vocab)).get("counts", {})
        if not a.domain_tokens:
            raise SystemExit("--action-vocab needs --domain-tokens: the action tokens are new "
                             "rows and are trained by that same mechanism.")

    n_base_vocab = None
    if a.domain_tokens:
        n_base_vocab = add_domain_tokens(model, tok, torch, a.index_tokens,
                                         act_list, cf_extra)

    model = FastLanguageModel.get_peft_model(
        model,
        r=a.rank,
        target_modules=TARGETS,
        lora_alpha=a.alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False if a.no_grad_ckpt else "unsloth",
        random_state=3407,
        max_seq_length=a.maxlen,
    )
    if a.init_from:
        warm_start(model, getattr(tok, "tokenizer", tok), torch, a.init_from, n_base_vocab or 0)

    emb0 = base0 = None
    if n_base_vocab is not None:
        emb0, base0 = unfreeze_new_rows(model, n_base_vocab, torch,
                                        a.index_tokens or bool(a.action_vocab)
                                        or bool(a.card_first))
    print("[peft] trainable %.1fM"
          % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6), flush=True)

    ev = (load_pairs(a.data, a.eval_n, index_tokens=a.index_tokens, action_vocab=act_set,
                     card_first=cf_set, scheme_b=scheme_b) if a.eval_n else None)
    d = load_pairs(a.data, a.limit, skip=a.eval_n, index_tokens=a.index_tokens,
                   action_vocab=act_set, card_first=cf_set, scheme_b=scheme_b)
    ds = Dataset.from_dict(d)
    print("[data] train %d | held-out %d | completions %r"
          % (len(ds), len(ev["prompt"]) if ev else 0, d["completion"][:8]), flush=True)
    if ev:
        ok, tot, sk = eval_top1(model, tok, torch, ev, a.maxlen,
                                index_tokens=a.index_tokens, action_vocab=act_counts,
                                card_first=cf_counts, scheme_b=scheme_b)
        print("[eval BEFORE] top1 %d/%d = %.2f%%  (skipped %d with >10 options)"
              % (ok, tot, 100.0 * ok / max(1, tot), sk), flush=True)

    cfg = SFTConfig(
        max_length=a.maxlen,
        per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=a.accum,
        warmup_steps=a.warmup,
        max_steps=a.steps if a.steps else -1,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        logging_steps=1,
        output_dir=a.out,
        optim="adamw_8bit",
        seed=3407,
        dataset_num_proc=a.num_proc,
        save_steps=a.save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        bf16=True,
        completion_only_loss=True,   # never train on the prompt
        packing=False,               # one decision per sample; packing would blur boundaries
        # `padding_free=True` would remove padding outright rather than merely grouping it, but
        # it concatenates sequences and relies on flash-attention varlen to keep them from
        # attending to each other -- and flash_attn is not installed in this image. Without it
        # the examples in a concatenated block WOULD see each other, which is silent corruption
        # rather than an error. `--group-by-length` gets most of the same win with none of that
        # risk, so it is the default lever here. `packing` is off for the same reason.
        report_to="none",
    )
    TrainerCls = SFTTrainer
    if a.group_by_length or a.emb_lr_mult != 1.0:
        TrainerCls = make_trainer_class(SFTTrainer, None, torch,
                                        group=a.group_by_length, emb_lr_mult=a.emb_lr_mult)
    tr = TrainerCls(model=model, train_dataset=ds, processing_class=tok, args=cfg)
    if (a.index_tokens or a.action_vocab or a.card_first) and not a.keep_eos:
        tk_eos = getattr(tok, "tokenizer", tok)
        tr.train_dataset = strip_trailing_eos(tr.train_dataset, tk_eos.eos_token_id, "train")
    print("[train] start (+%.1fs)" % (time.time() - t0), flush=True)
    r = tr.train()
    print("[done] %s" % r.metrics, flush=True)
    if ev:
        ok, tot, sk = eval_top1(model, tok, torch, ev, a.maxlen,
                                index_tokens=a.index_tokens, action_vocab=act_counts,
                                card_first=cf_counts, scheme_b=scheme_b)
        print("[eval AFTER ] top1 %d/%d = %.2f%%   GATE: beat the reranker's 69.7%%"
              % (ok, tot, 100.0 * ok / max(1, tot)), flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    if a.card_first:
        import shutil
        shutil.copyfile(a.card_first, os.path.join(a.out, "cardfirst_vocab.json"))
        print("[cardfirst] vocabulary copied into the checkpoint", flush=True)
    if a.action_vocab:
        # The vocabulary is part of the model: every added row means something only relative
        # to this list, so serving a checkpoint against a list rebuilt from other data would
        # mis-map every action. Ship it in the checkpoint directory, like domain_embeddings.
        import shutil
        shutil.copyfile(a.action_vocab, os.path.join(a.out, "action_vocab.json"))
        print("[action] vocabulary copied into the checkpoint", flush=True)
    if n_base_vocab is not None and emb0 is not None:
        report_embedding_movement(model, n_base_vocab, emb0, torch, a.out, base0)
    print("[peak] VRAM %.1f GiB allocated / %.1f GiB reserved"
          % (torch.cuda.max_memory_allocated() / 2**30,
             torch.cuda.max_memory_reserved() / 2**30), flush=True)
    print("[saved] %s | total %.1f min" % (a.out, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
