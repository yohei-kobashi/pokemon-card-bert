#!/usr/bin/env python3
"""Head-to-head pilot comparison on IDENTICAL decklists, with sequential stopping.

Why the mirror. Comparing two pilots by each one's win rate against a field needs TWO
measurements and compares them, so the noise of both lands in the difference -- and the baseline
half of it re-scores 2.6pt apart between runs on the same checkpoint
([[rl-gate-is-noisier-than-assumed]]). Playing them against EACH OTHER on the same decklist
removes that entirely: by symmetry the null is exactly 0.500, so there is no baseline to measure.

    detect 10pt at 2 sigma      96 decisive games
    detect  5pt                384
    detect  3pt              1,068

Seats alternate every game (arena.match already does this), which cancels any first-player
advantage without needing to know its size.

SPRT stops as soon as the evidence crosses a bound instead of always paying the fixed N -- the
standard practice in engine testing. Expected decisive games at the default bounds
(p0 0.50, p1 0.55, alpha = beta = 0.05), and the wall clock at the 9B's ~15 s/game:

    B truly 70%     ~84 games     ~21 min      -> B_STRONGER
    B truly 60%    ~196          ~49 min      -> B_STRONGER
    B truly 50%    ~290          ~73 min      -> NO_DIFFERENCE

Raise --p1 when only a large effect matters; it resolves faster but stops distinguishing small
real edges from none.

--mirror goes one step further: the same decklist AND the same shuffle order for both seats, and
each seed replayed once per seat. Needs the patched engine from tools/build_engine_mirror.py.
Two consequences:

  * With identical pilots every pair splits, so p is exactly 0.500 with SE exactly 0 -- the null
    is structural, not merely expected.
  * The two games of a pair are dependent, so the SPRT runs on PAIRS (B won both / A won both /
    split, which carries no information about piloting and is skipped). Treating the games as
    independent Bernoulli trials would be wrong by a factor 2(1-split_rate), i.e.
    anti-conservative below a 50% split rate and conservative above it.

WHERE THE GAIN IS, measured on engine vs engine-weakened-15%:

  * Re-running the SAME comparison is bit-identical (fixed seeds, deterministic pilots). That is
    the point for the gate: the same checkpoint re-scores 2.6pt apart today
    ([[rl-gate-is-noisier-than-assumed]]), and here it re-scores at 0.0pt.
  * At FIXED N (600 games) the paired estimator needs 0.48x the games on dragapult, 0.42x on
    alakazam, 0.93x on crustle_stall -- never worse, ~2x on two of three. It buys this by
    measuring a BIGGER gap, not a smaller variance.
  * Under SPRT EARLY STOPPING most of that is given back: dragapult reaches WORSE in 92 games
    against stock's 101. The sign test conditions on decisive pairs, so the ~40% of games that
    land in split pairs pay for nothing. (Conditioning is the right test -- the split rate is a
    nuisance parameter -- but the fixed-N paired t uses those splits as exact zeros, which is
    where the 2x came from.) So: prefer --mirror with a fixed N, not with aggressive stopping.

CALIBRATION. Pair-level effects run about 1.75x the game-level gap (measured 2.01 / 1.70 / 1.53),
so BOTH thresholds are conservative on pairs: --p1 0.55 -> ~0.59 (superiority, the BETTER side)
and --margin 0.05 -> ~0.09 (non-inferiority, which is what actually decides WORSE and so is the
one that matters to the screening loop). Leaving them at the game-level defaults gives the 2x
back to the stopping rule. They are NOT changed automatically: the verdict strings feed
dagger_loop_*.sh, and silently redefining WORSE would corrupt comparisons across rounds.

SCOPE. This answers "who pilots THIS deck better in the mirror", not "who wins on the ladder".
A mirror has no matchup asymmetry, so a pilot can be good here and bad against the field; the
live-weighted protocol remains the thing to judge submissions on. Run it per deck -- the fleet
average moves with a single deck ([[lm-below-engine-baseline]]).

    python tools/mirror_match.py --deck crustle_stall --a engine --b qwen:/root/out/teacher9b_v39
"""
import argparse
import json
import hashlib
import math
import statistics
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_deck(name):
    import library
    with open(library.deck_path(name)) as f:
        return [int(x) for x in f if x.strip()]


class QwenScorer:
    """Adapts the 9B to lm/agent's scorer contract: score(prompt, cands, obs) -> [float].

    Returns each candidate's full token-sequence log-probability under the constrained-output
    rule (indices only), which is the same quantity tools/instance/eval_teacher.py measures.
    """

    def __init__(self, adapter, base=None, maxlen=1024, merge=True, kv=True, backend="hf",
                 fp8=False, compile_mode=""):
        # The base was hard-coded to the 9B. A Qwen3-4B checkpoint loaded on that base would
        # either fail on shapes or, worse, load an adapter trained for a different model. Read it
        # from the adapter's own config.
        if base is None:
            cfgp = os.path.join(adapter, "adapter_config.json")
            base = "unsloth/Qwen3.5-9B-Base"
            if os.path.exists(cfgp):
                base = json.load(open(cfgp)).get("base_model_name_or_path") or base
                base = base.replace("-unsloth-bnb-4bit", "")
            print("[qwen] base model from the checkpoint: %s" % base, flush=True)
        sys.path.insert(0, os.path.join(ROOT, "tools", "instance"))
        from eval_teacher import score_decision
        import torch
        self._score_decision = score_decision
        self.torch = torch
        self.maxlen = maxlen
        # unsloth is a TRAINING optimiser and its patched forward is a liability at inference: it
        # ignores `logits_to_keep` (so every forward computes logits at all ~368 positions over a
        # 251,048 vocabulary) and returns no KV cache when asked for one (so the tie-break pass
        # has to re-run the whole prompt). Its inference path additionally asserts q_len == 1 the
        # moment past_key_values is passed. Plain transformers supports both. Measured, 120 real
        # decisions: unsloth 134.1 ms/decision with both fast paths REFUSED by the probe.
        if backend == "unsloth":
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(
                model_name=base, max_seq_length=maxlen,
                load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                base, dtype=torch.bfloat16, device_map="cuda")
            tok = AutoTokenizer.from_pretrained(base)
        # A --domain-tokens checkpoint was trained on a RESIZED vocabulary (248,320 -> 251,048)
        # and its 2,971 new embedding rows live in domain_embeddings.pt, not in the LoRA. Load
        # the checkpoint's tokenizer, resize to match, and restore those rows BEFORE attaching
        # the adapter -- otherwise every card token either fails to map or maps to a row the
        # model never trained, and nothing errors.
        emb = os.path.join(adapter, "domain_embeddings.pt")
        if os.path.exists(emb):
            import torch as _t
            from transformers import AutoTokenizer
            tk_new = AutoTokenizer.from_pretrained(adapter)
            base_tk = getattr(tok, "tokenizer", tok)
            if len(tk_new) != len(base_tk):
                model.resize_token_embeddings(len(tk_new))
                blob = _t.load(emb, map_location="cpu")
                n_base, rows = blob["n_base"], blob["rows"]
                w = model.get_input_embeddings().weight
                if w.shape[0] != n_base + rows.shape[0]:
                    raise SystemExit("domain_embeddings.pt does not fit this model: %d rows "
                                     "on top of %d, but the embedding is %d"
                                     % (rows.shape[0], n_base, w.shape[0]))
                with _t.no_grad():
                    w[n_base:] = rows.to(device=w.device, dtype=w.dtype)
                if hasattr(tok, "tokenizer"):
                    tok.tokenizer = tk_new
                else:
                    tok = tk_new
                print("[qwen] restored %d domain-token rows" % rows.shape[0], flush=True)
        if os.path.exists(os.path.join(adapter, "adapter_config.json")):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        elif not os.path.isdir(adapter):
            raise SystemExit("no adapter at %r" % adapter)
        model.eval()
        # Folding the LoRA into the base weights removes a B(A(x)) branch from every linear.
        # Measured 134.1 -> 71.3 ms per decision, the single largest win available here. It is
        # NOT bit-exact (W + BA is rounded once into bf16), so tools/check_merge_equiv.py gates
        # it on argmax agreement: the screen's paired statistic compares checkpoints, and a
        # scorer that quietly picks differently would show up as the checkpoint moving.
        if merge and hasattr(model, "merge_and_unload"):
            try:
                model = model.merge_and_unload()
                model.eval()
                print("[qwen] LoRA merged into the base weights", flush=True)
            except Exception as e:
                print("[qwen] merge failed, keeping the adapter live: %s" % e, flush=True)
        # FP8 is the one lever whose arithmetic clearly favours this workload. The prefill is
        # compute-bound at ~52 TFLOP/s against ~91 TFLOPS of bf16 peak, and Ada (sm89) has FP8
        # tensor cores at roughly twice bf16 -- so it raises the ceiling rather than chasing
        # overhead. It also halves the weights, which matters more than the speed: the screen
        # runs 3 shards because 12.3 GiB x 3 fills a 47.4 GiB card, and the shard count is what
        # limits how much of instance2's 13.44 cores the game loops can use (3 shards are
        # single-threaded Python, so they occupy about 3).
        if fp8:
            try:
                from torchao.quantization import quantize_
                cfg = None
                for name in ("Float8DynamicActivationFloat8WeightConfig",
                             "float8_dynamic_activation_float8_weight"):
                    try:
                        import torchao.quantization as _q
                        obj = getattr(_q, name, None)
                        if obj is not None:
                            cfg = obj()
                            break
                    except Exception:
                        continue
                if cfg is None:
                    raise RuntimeError("no float8 config in this torchao")
                # lm_head is 154,733 x 2560 and its output IS the score, so it is quantised only
                # if the equivalence check still passes; keep it out by default.
                quantize_(model.model, cfg)
                print("[qwen] FP8 dynamic-activation weights (body only)", flush=True)
            except Exception as e:
                print("[qwen] FP8 unavailable (%s) -- staying in bf16" % e, flush=True)
        if compile_mode:
            try:
                model.forward = torch.compile(model.forward, mode=compile_mode, dynamic=True)
                print("[qwen] torch.compile mode=%s" % compile_mode, flush=True)
            except Exception as e:
                print("[qwen] compile failed (%s)" % e, flush=True)
        self.model = model
        self.tk = getattr(tok, "tokenizer", tok)
        if self.tk.pad_token is None:
            self.tk.pad_token = self.tk.eos_token
        self.n = 0
        self.t = 0.0
        # Both fast paths are PROBED, not assumed. unsloth patches the model, and its inference
        # path asserts q_len == 1 whenever past_key_values is passed -- so handing it a cache on
        # the PREFILL raises AssertionError, while handing it one for a single token is exactly
        # what it wants. A capability that silently misbehaves would corrupt every score, so each
        # is checked against the plain forward before it is switched on.
        self._klast = {}
        self.kv = False
        try:
            # A REAL prompt, not [1,2,3,4]: both paths are checked on the shape of input they
            # will actually see, and the tolerance is on LOG-PROBS, which are bounded, rather
            # than on raw logits. Neither path is bit-exact -- computing logits at one position
            # instead of 64 changes the reduction order, and bf16 rounds differently. Measured
            # here: logits_to_keep 0.0625 max on raw logits, KV reuse 0.125 max on log-probs,
            # with the same argmax. The tolerance below admits that and the ARGMAX is what the
            # screen reads; tools/check_scorer_equiv.py measures the picks that actually change.
            probe = self.tk("[ACT]\nDECK win[c743x4] eng[c13] T3.2 ME A[c5:100/100] pz3 dk20 "
                            "bm5 H[c7,c9] || SEL MAIN n1-1 :: 0=attach:c7@ACTIVE 1=end",
                            add_special_tokens=False)["input_ids"]
            t = torch.tensor([probe], device=model.device)
            with torch.no_grad():
                full = self.model(input_ids=t, use_cache=False)
                ref = torch.log_softmax(full.logits[0, -1, :].float(), -1)
                try:
                    o = self.model(input_ids=t, use_cache=False, logits_to_keep=1)
                    lp = torch.log_softmax(o.logits[0, -1, :].float(), -1)
                    if o.logits.shape[1] == 1 and int(lp.argmax()) == int(ref.argmax()) \
                            and float((lp - ref).abs().max()) < 0.5:
                        self._klast = {"logits_to_keep": 1}
                    else:
                        print("[qwen] logits_to_keep rejected by the probe", flush=True)
                except Exception as e:
                    print("[qwen] logits_to_keep unsupported (%s)" % type(e).__name__, flush=True)
                if kv:
                    o1 = self.model(input_ids=torch.tensor([probe[:-1]], device=model.device),
                                    use_cache=True, **self._klast)
                    pkv = getattr(o1, "past_key_values", None)
                    if pkv is None or not hasattr(pkv, "crop"):
                        print("[qwen] backend returns no croppable KV cache -- reuse disabled",
                              flush=True)
                    else:
                        o2 = self.model(input_ids=torch.tensor([[probe[-1]]],
                                                               device=model.device),
                                        past_key_values=pkv, use_cache=True, **self._klast)
                        lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), -1)
                        d = float((lp2 - ref).abs().max())
                        self.kv = int(lp2.argmax()) == int(ref.argmax()) and d < 0.5
                        if not self.kv:
                            print("[qwen] KV reuse disagrees with the plain forward by %.3f "
                                  "-- disabled" % d, flush=True)
        except Exception as e:
            print("[qwen] fast-path probe failed (%s) -- using the plain forward" % e,
                  flush=True)
        print("[qwen] logits_to_keep=%s  kv_reuse=%s"
              % (bool(self._klast), self.kv), flush=True)
        # A card-first checkpoint answers with a CARD token, not a menu index, so the index-trie
        # scorer would score a vocabulary the model never emits. The scheme is read from the
        # vocabulary the training run shipped inside the checkpoint, so a model and its decoder
        # cannot be paired wrongly.
        self.cf = None
        cfp = os.path.join(adapter, "cardfirst_vocab.json")
        if os.path.exists(cfp):
            self.cf = json.load(open(cfp))
            self.scheme_b = self.cf.get("scheme") == "b"
            print("[qwen] card-first decoding, scheme %s" % ("B" if self.scheme_b else "A"),
                  flush=True)

    def _score_card_first(self, prompt, cands):
        """logP for each candidate under the card-first answer.

        One forward gives every candidate's first token. Only groups that share a first token and
        are not the same act need a second -- measured at 46.7% of real decisions, not the 14.8%
        this docstring used to claim.

        Three optimisations, all measured on 120 real decisions of the v40 pool (RTX 5880 Ada),
        starting from 134.1 ms per decision:

          merged LoRA        134.1 -> 71.3   folded into the base weights in __init__
          logits_to_keep=1    71.3 -> 66.6   a plain forward computes logits at every position
                                             and only the last is read: 368 x 251,048 x 2560 x 2
                                             of lm_head plus a 740 MB fp32 upcast, per forward
          one gather/sync     66.6 -> 69.2*  float(tensor[i]) per candidate is a device sync;
                                             one index_select + one .tolist() replaces ~6
          KV for the 2nd tok  69.2 -> ~51    the tie-break re-ran the WHOLE prompt to score one
                                             extra position

        (*the gather measured slightly slower in isolation and is kept because it removes syncs
        that matter once the tie-break shares the same cache.)

        BATCHING WAS TESTED AND DOES NOT HELP: batch 4 is 1.05x over batch 1 and batch 32 is
        WORSE (53.6 vs 45.2 ms/decision). A 368-token prefill of a 4B model is ~2.9 TFLOP, which
        at 45 ms is most of this card's bf16 throughput -- there is no idle GPU to fill. That
        also removes vLLM's main lever here; continuous batching and paged attention optimise
        memory-bound decode, and this workload is compute-bound prefill.
        """
        import sys as _s
        _s.path.insert(0, ROOT)
        from lm.action_token import (first_token, second_token, sub_index, groups, to_scheme_b,
                                     equivalent, SUB_TOKENS)
        torch = self.torch
        dev = self.model.device
        p = to_scheme_b(prompt) if self.scheme_b else prompt
        heads = [first_token(c) for c in cands]
        ids = self.tk(p, add_special_tokens=False, truncation=True,
                      max_length=self.maxlen)["input_ids"]
        hid = {h: self.tk.convert_tokens_to_ids(h) for h in set(heads)}

        # which groups need a tie-break at all -- decided BEFORE the forward, so the prompt's KV
        # cache is only kept when something is going to use it
        need = {}
        for h in set(heads):
            grp = [i for i, x in enumerate(heads) if x == h]
            if len(grp) > 1 and not all(equivalent(cands[i], cands[grp[0]]) for i in grp):
                need[h] = grp
        want_kv = bool(need) and self.kv

        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=dev),
                             use_cache=want_kv, **self._klast)
            lp1 = torch.log_softmax(out.logits[0, -1, :].float(), -1)
            order = sorted(hid)
            sel = torch.tensor([hid[h] if hid[h] is not None else 0 for h in order], device=dev)
            vals = lp1[sel].tolist()                      # ONE sync for the whole decision
        got = {h: (vals[k] if hid[h] is not None else -1e9) for k, h in enumerate(order)}
        base = [got[h] for h in heads]

        cache = getattr(out, "past_key_values", None) if want_kv else None
        for h, grp in need.items():
            with torch.no_grad():
                if cache is not None:
                    o2 = self.model(input_ids=torch.tensor([[hid[h]]], device=dev),
                                    past_key_values=cache, use_cache=True, **self._klast)
                    try:
                        cache.crop(len(ids))              # put it back for the next group
                    except Exception:
                        cache = None                      # no crop -> re-prefill the rest
                else:
                    o2 = self.model(input_ids=torch.tensor([ids + [hid[h]]], device=dev),
                                    use_cache=False, **self._klast)
                lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), -1)
                if self.scheme_b:
                    toks = [second_token(cands[i]) for i in grp]
                else:
                    toks = []
                    for i in grp:
                        r = sub_index(prompt, cands, i)
                        toks.append(SUB_TOKENS[r] if r is not None else None)
                jj = [self.tk.convert_tokens_to_ids(t) if t else None for t in toks]
                sel2 = torch.tensor([j if j is not None else 0 for j in jj], device=dev)
                v2 = lp2[sel2].tolist()
            for k, i in enumerate(grp):
                base[i] += v2[k] if jj[k] is not None else -1e9
        return base

    def score(self, prompt, cands, obs=None):
        # The prompt format is part of the model. build_sft trained on "[ACT]\n<state>"; if the
        # live serializer stops emitting that tag the model is off-distribution and nothing
        # fails, so check rather than silently prepending it.
        if not prompt.startswith("[ACT]"):
            raise ValueError("prompt does not start with [ACT] -- the trained format was "
                             "'[ACT]\\n<state>'. Refusing to score off-distribution: %r"
                             % prompt[:60])
        t0 = time.time()
        if self.cf is not None:
            lp = self._score_card_first(prompt, cands)
            self.t += time.time() - t0
            self.n += 1
            return lp
        lp, _mass, _ = self._score_decision(self.model, self.tk, self.torch, prompt,
                                            len(cands), self.maxlen)
        self.t += time.time() - t0
        self.n += 1
        return lp


class HFRerankScorer:
    """Cross-encoder straight from a training checkpoint -- no ONNX export needed to evaluate."""

    def __init__(self, path, maxlen=768):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.dev).eval()
        self.maxlen = maxlen
        self.n = 0
        self.t = 0.0

    def score(self, prompt, cands, obs=None):
        t0 = time.time()
        with self.torch.no_grad():
            enc = self.tok([[prompt, c] for c in cands], padding=True,
                           truncation="only_first", max_length=self.maxlen,
                           return_tensors="pt").to(self.dev)
            s = self.model(**enc).logits.squeeze(-1).float().tolist()
        self.t += time.time() - t0
        self.n += 1
        return s if isinstance(s, list) else [s]


_SCORERS = {}


def make_noisy(agent, q, salt=0):
    """`agent`, but on a q-fraction of states it plays a random legal move instead.

    A pilot of KNOWN, tunable weakness, for testing the harness itself without a GPU. The coin is
    a hash of the observation, not a live RNG: a pilot carrying its own randomness re-randomises
    between the two games of a --mirror pair and destroys the common random numbers the pairing
    exists to exploit.
    """
    import hashlib
    import random as _random

    def f(obs):
        d = hashlib.blake2b(json.dumps(obs["current"], sort_keys=True).encode(),
                            digest_size=8, salt=str(salt).encode()[:16]).digest()
        n_ = int.from_bytes(d, "big")
        if n_ / 2**64 < q:
            sel = obs["select"]
            n = len(sel["option"])
            k = min(max(sel["minCount"], min(sel["maxCount"], 1)), n)
            return _random.Random(n_).sample(range(n), k) if k > 0 else []
        return agent(obs)
    return f


def make_no_kind(agent, banned):
    """`agent`, but never plays `banned`. When it would, substitute `end` if that is on the menu
    and otherwise the first other option -- which is how the LM actually substitutes it on
    ns_zoroark (retreat -> end 49%, retreat -> play 39%)."""
    from lm.actions import encode_option

    def f(obs):
        pick = agent(obs)
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if not pick or len(pick) != 1 or sel.get("minCount", 1) > 1 or pick[0] >= len(opts):
            return pick
        kind = lambda i: encode_option(opts[i], obs).split(":", 1)[0].split("@", 1)[0]
        if kind(pick[0]) != banned:
            return pick
        alt = None
        for i in range(len(opts)):
            if kind(i) == "end":
                alt = i
                break
            if kind(i) != banned and alt is None:
                alt = i
        return [alt] if alt is not None else pick
    return f


def make_defer(lm_agent, ref_agent, kinds):
    """The LM pilots, except at decisions where any candidate is one of `kinds` -- those go to
    engine_v2. Isolates how much of a deficit ONE action kind accounts for (the method that
    localised the attach deficit to +11.4pt in [[attach-decisions-at-chance]])."""
    from lm.actions import encode_option
    want = set(kinds)

    def f(obs):
        opts = (obs.get("select") or {}).get("option") or []
        for o in opts:
            if encode_option(o, obs).split(":", 1)[0].split("@", 1)[0] in want:
                return ref_agent(obs)
        return lm_agent(obs)
    return f


def make_agent(spec, deck_name, deck_ids, profile):
    """The scorer is CACHED across decks. It does not depend on the deck -- only the prompt
    does -- and rebuilding it per deck loaded a fresh 9B for every one of 63 decks without
    freeing the last, which filled the card by deck three and left accelerate offloading
    parameters to the meta device ('Tensor.item() cannot be called on meta tensors'). At 149M
    that was merely wasteful; at 9B it is fatal."""
    from lm.agent import make_lm_agent
    from tools import rl_config
    fmt = dict(rl_config.PROMPT_FMT)
    if spec == "engine":
        return make_lm_agent(deck_ids, profile, model=None), None
    if spec.startswith("noisy:"):   # harness self-test: engine_v2 weakened by a known amount
        return make_noisy(make_lm_agent(deck_ids, profile, model=None),
                          float(spec.split(":", 1)[1])), None
    if spec.startswith("defer:"):
        # defer:<kind>[,<kind>]:<modelspec>
        _, kinds, sub = spec.split(":", 2)
        lm, sc = make_agent(sub, deck_name, deck_ids, profile)
        return make_defer(lm, make_lm_agent(deck_ids, profile, model=None),
                          kinds.split(",")), sc
    if spec.startswith("nokind:"):
        # engine_v2 with ONE action kind ablated, substituted the way the LM substitutes it.
        # Turns "the LM never does X" from a correlation into a causal test: take X away from
        # the pilot that wins, and see whether it stops winning.
        return make_no_kind(make_lm_agent(deck_ids, profile, model=None),
                            spec.split(":", 1)[1]), None
    if spec in _SCORERS:
        sc = _SCORERS[spec]
        return make_lm_agent(deck_ids, profile, model=sc, deck_name=deck_name, **fmt), sc
    kind, _, path = spec.partition(":")
    if kind == "qwen":
        sc = QwenScorer(path)
    elif kind == "hf":
        sc = HFRerankScorer(path)
    elif kind == "rerank":
        from lm.rerank_scorer import OnnxRerankerScorer
        sc = OnnxRerankerScorer(os.path.join(path, "model.onnx"), path)
    else:
        raise SystemExit("unknown agent spec %r (engine | qwen:<dir> | hf:<dir> | rerank:<onnx dir>)" % spec)
    _SCORERS[spec] = sc
    return make_lm_agent(deck_ids, profile, model=sc, deck_name=deck_name, **fmt), sc


def sprt(w, l, p0, p1, alpha, beta, margin=0.05):
    """-> (llr_superiority, llr_non_inferiority, verdict). Draws carry no information about p.

    TWO tests run together, because a single superiority SPRT answers the wrong question when
    the challenger is not expected to win. With H0 p<=0.50 / H1 p>=0.55 alone, a true 40% pilot
    and a true 50% pilot BOTH come back "NO_DIFFERENCE" -- and the 40% one resolves FASTER (118
    games vs 290), so the worse result looks like the more confident one.

      superiority      H0: p <= p0        H1: p >= p1        is B actually better?
      non-inferiority  H0: p <= p0-margin H1: p >= p0        is B at least not meaningfully worse?

    WORSE      non-inferiority rejected -- B is more than `margin` behind
    EQUIVALENT non-inferior AND not superior -- the "same strength" answer, stated positively
    BETTER     superiority accepted
    """
    def _llr(hi_p, lo_p):
        return w * math.log(hi_p / lo_p) + l * math.log((1 - hi_p) / (1 - lo_p))
    hi = math.log((1 - beta) / alpha)
    lo = math.log(beta / (1 - alpha))
    sup = _llr(p1, p0)
    ni = _llr(p0, p0 - margin)
    if ni <= lo:
        v = "WORSE"
    elif ni >= hi and sup >= hi:
        v = "BETTER"
    elif ni >= hi and sup <= lo:
        v = "EQUIVALENT"
    elif ni >= hi:
        # Non-inferiority is settled, superiority is not -- the true rate sits inside the
        # indifference region (p0, p1). Collapsing this to "undecided" would throw away the
        # conclusion that was actually reached, which is the one that matters when the
        # challenger is not expected to win.
        v = "NOT_WORSE"
    else:
        v = "undecided"
    return sup, ni, v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", action="append", required=True, help="repeatable")
    ap.add_argument("--a", default="engine", help="reference pilot")
    ap.add_argument("--b", required=True, help="challenger")
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--p0", type=float, default=0.50)
    ap.add_argument("--p1", type=float, default=0.55)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="equivalence margin: B counts as WORSE only below p0-margin")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--out", default="")
    ap.add_argument("--mirror", action="store_true",
                    help="same decklist AND the same shuffle order for both seats, and replay "
                         "each seed once per seat. Needs the patched engine from "
                         "tools/build_engine_mirror.py.")
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--seed", type=int, default=1, help="--mirror: base seed for the shuffles")
    args = ap.parse_args()

    eng = None
    if args.mirror:
        import zlib

        from tools.mirror_env import DEFAULT_SO, MirrorEngine
        from tools.mirror_env import play as mplay
        so_path = args.mirror_so or DEFAULT_SO
        eng = MirrorEngine(so_path)
        # Reproducibility is guaranteed per (seed, .so), NOT per seed alone: the permutation
        # std::shuffle produces from a given mt19937 state is implementation-defined, so a .so
        # built with a different libstdc++ can deal a different game from the same seed. Record
        # which binary produced these numbers so a cross-round comparison can be checked.
        so_sha = hashlib.sha256(open(so_path, "rb").read()).hexdigest()[:16]
        if abs(args.p1 - 0.55) < 1e-9 or abs(args.margin - 0.05) < 1e-9:
            print("[mirror] --p1 0.55 / --margin 0.05 are calibrated for game-level effects; on "
                  "pairs the gap runs ~1.75x, so ~0.59 and ~0.09 are the equivalents. The "
                  "defaults still work, they just stop later than they need to (--margin is the "
                  "one that decides WORSE).", flush=True)

    from tools.arena import play
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    results = {}
    for deck in args.deck:
        ids = load_deck(deck)
        prof = tuning.get(deck, {})
        if args.mirror and not globals().get("_FP_DONE"):
            # Fingerprint a FIXED deck, never this shard's first one: a sharded screen would
            # otherwise stamp a different fingerprint per shard, and the merged file would look
            # like it changed engines the moment --shards changed.
            import library

            from tools.mirror_env import engine_fingerprint
            fp_deck = sorted(library.list_decks())[0]
            fp_ids = load_deck(fp_deck)
            fp = engine_fingerprint(eng, fp_ids)
            print("[mirror] engine %s\n[mirror] binary sha %s | shuffle fingerprint %s (deck %s)"
                  % (so_path, so_sha, fp, fp_deck), flush=True)
            globals()["_FP_DONE"] = fp
        agentA, _ = make_agent(args.a, deck, ids, prof)
        agentB, scB = make_agent(args.b, deck, ids, prof)
        w = l = d = 0
        # per-seat tally: the aggregate hid that mega_lucario's LM went 8-7 when it always
        # moved first (tools/diag_pilot.py) but 4-36 with seats alternating. A pilot that
        # collapses in one seat is a different bug from a pilot that is simply weaker.
        seat = {0: [0, 0], 1: [0, 0]}
        pair_vals = []          # --mirror: B's share of each pair, in {0, 0.5, 1}
        pw = pl = 0             # --mirror: pairs B won/lost from BOTH seats
        t0 = time.time()
        verdict = "undecided"
        base_seed = args.seed + (zlib.crc32(deck.encode()) & 0xFFFF if args.mirror else 0)

        def record(r, b_seat):
            nonlocal w, l, d
            if r is None:
                d += 1
                return None
            b_won = (r == 1) if b_seat == 1 else (r == 0)
            seat[b_seat][0 if b_won else 1] += 1
            if b_won:
                w += 1
            else:
                l += 1
            return b_won

        # --mirror plays each seed TWICE, once per seat, and the SPRT then runs on those pairs
        # rather than on games. The two games of a pair are strongly dependent -- when the pilots
        # agree the mirror guarantees they split -- so feeding them to a Bernoulli SPRT as
        # independent trials is wrong in a direction that depends on the deck: with a split rate
        # s the true variance is 2(1-s)x the independent assumption, i.e. anti-conservative below
        # s=0.5 and conservative above it. A pair is one trial: B won both, A won both, or split
        # (which carries no information about piloting and is skipped).
        units = args.max_games // 2 if args.mirror else args.max_games
        for g in range(units):
            if args.mirror:
                s = base_seed + g
                b1 = record(mplay(eng, agentA, agentB, ids, ids, s, mirror=1), 1)
                b2 = record(mplay(eng, agentB, agentA, ids, ids, s, mirror=1), 0)
                got = [x for x in (b1, b2) if x is not None]
                pair_vals.append(sum(got) / len(got) if got else 0.5)
                if b1 is not None and b2 is not None and b1 == b2:
                    pw += b1
                    pl += not b1
                sw, sl, ngames = pw, pl, 2 * (g + 1)
            else:
                # swap seats every game: first-player advantage is a systematic bias, and
                # alternating cancels it without needing to know how large it is
                b_seat = 1 if g % 2 == 0 else 0
                if b_seat == 1:
                    record(play(agentA, agentB, ids, ids), 1)
                else:
                    record(play(agentB, agentA, ids, ids), 0)
                sw, sl, ngames = w, l, g + 1
            if sw + sl >= 20:
                sup, ni, verdict = sprt(sw, sl, args.p0, args.p1, args.alpha, args.beta,
                                        args.margin)
                if verdict != "undecided":
                    break
            if ngames % 20 == 0:
                sup, ni, _v = sprt(sw, sl, args.p0, args.p1, args.alpha, args.beta, args.margin)
                extra = ("  pairs %d-%d split %.0f%%"
                         % (pw, pl, 100.0 * (len(pair_vals) - pw - pl) / max(1, len(pair_vals)))
                         if args.mirror else "")
                print("  %-22s %3d games  B %3d-%-3d (%.1f%%) draws %d  sup %+.2f ni %+.2f%s  %.0fs"
                      % (deck, ngames, w, l, 100.0 * w / max(1, w + l), d, sup, ni, extra,
                         time.time() - t0), flush=True)
        n = w + l
        p = w / max(1, n)
        # The independent-trial SE is wrong once games are paired. Take it from the spread of the
        # pair values instead: with identical pilots every pair is exactly 0.5, so this correctly
        # reports SE 0 where sqrt(0.25/n) would claim residual uncertainty that does not exist.
        if args.mirror and len(pair_vals) > 1:
            p = sum(pair_vals) / len(pair_vals)
            se = statistics.stdev(pair_vals) / math.sqrt(len(pair_vals))
        else:
            se = math.sqrt(0.25 / max(1, n))
        sup, ni, verdict = sprt(pw if args.mirror else w, pl if args.mirror else l,
                                args.p0, args.p1, args.alpha, args.beta, args.margin)
        print("%-24s B %d-%d = %.1f%% (95%% CI %.1f-%.1f)  draws %d  sup %+.2f ni %+.2f -> %s   %.0fs"
              % (deck, w, l, 100 * p, 100 * (p - 1.96 * se), 100 * (p + 1.96 * se), d,
                 sup, ni, verdict, time.time() - t0), flush=True)
        if args.mirror:
            npair = len(pair_vals)
            split = npair - pw - pl
            print("  pairs: B %d - A %d, %d split (%.1f%%)  paired SE %.3f  seed base %d"
                  % (pw, pl, split, 100.0 * split / max(1, npair), se, base_seed), flush=True)
            if pw + pl == 0 and npair:
                print("  NOTE: every pair split -- on this deck the two pilots never diverged "
                      "into a different result, so the mirror says they are interchangeable "
                      "here, not that the test lacked games.", flush=True)
        for si in (0, 1):
            sw, sl = seat[si]
            if sw + sl:
                print("  B as player %d: %d-%d = %.1f%%" % (si, sw, sl,
                                                            100.0 * sw / (sw + sl)), flush=True)
        if d > 0.05 * (n + d):
            print("  NOTE: %d/%d games were draws/timeouts and are excluded from the test; a "
                  "pilot that stalls games would hide here." % (d, n + d), flush=True)
        if scB is not None and getattr(scB, "n", 0):
            print("  challenger scored %d decisions, %.3f s each"
                  % (scB.n, scB.t / scB.n), flush=True)
        results[deck] = {"w": w, "l": l, "d": d, "p": p, "se": se,
                         "verdict": verdict, "sup": sup, "ni": ni,
                         "seat0": seat[0], "seat1": seat[1]}
        if args.mirror:
            # NEW fields only -- w/l/d/p keep their meaning so mirror and non-mirror rounds stay
            # comparable in the loop scripts that diff `p` across rounds.
            results[deck].update({"mirror": True, "pair_w": pw, "pair_l": pl,
                                  "pairs": len(pair_vals), "seed_base": base_seed,
                                  # binary sha is advisory (it differs between machines that
                                  # nonetheless deal identical games); the fingerprint is the
                                  # invariant that must match for numbers to be comparable.
                                  "engine_sha": so_sha,
                                  "shuffle_fp": globals().get("_FP_DONE", "")})
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"a": args.a, "b": args.b, "decks": results}, f, indent=1)
        print("-> %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
