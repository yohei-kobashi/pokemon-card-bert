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

SCOPE. This answers "who pilots THIS deck better in the mirror", not "who wins on the ladder".
A mirror has no matchup asymmetry, so a pilot can be good here and bad against the field; the
live-weighted protocol remains the thing to judge submissions on. Run it per deck -- the fleet
average moves with a single deck ([[lm-below-engine-baseline]]).

    python tools/mirror_match.py --deck crustle_stall --a engine --b qwen:/root/out/teacher9b_v39
"""
import argparse
import json
import math
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

    def __init__(self, adapter, base="unsloth/Qwen3.5-9B-Base", maxlen=1024):
        sys.path.insert(0, os.path.join(ROOT, "tools", "instance"))
        from eval_teacher import score_decision
        from unsloth import FastLanguageModel
        import torch
        self._score_decision = score_decision
        self.torch = torch
        self.maxlen = maxlen
        model, tok = FastLanguageModel.from_pretrained(
            model_name=base, max_seq_length=maxlen,
            load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
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
        self.model = model
        self.tk = getattr(tok, "tokenizer", tok)
        if self.tk.pad_token is None:
            self.tk.pad_token = self.tk.eos_token
        self.n = 0
        self.t = 0.0

    def score(self, prompt, cands, obs=None):
        # The prompt format is part of the model. build_sft trained on "[ACT]\n<state>"; if the
        # live serializer stops emitting that tag the model is off-distribution and nothing
        # fails, so check rather than silently prepending it.
        if not prompt.startswith("[ACT]"):
            raise ValueError("prompt does not start with [ACT] -- the trained format was "
                             "'[ACT]\\n<state>'. Refusing to score off-distribution: %r"
                             % prompt[:60])
        t0 = time.time()
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


def make_agent(spec, deck_name, deck_ids, profile):
    from lm.agent import make_lm_agent
    from tools import rl_config
    fmt = dict(rl_config.PROMPT_FMT)
    if spec == "engine":
        return make_lm_agent(deck_ids, profile, model=None), None
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
    args = ap.parse_args()

    from tools.arena import play
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    results = {}
    for deck in args.deck:
        ids = load_deck(deck)
        prof = tuning.get(deck, {})
        agentA, _ = make_agent(args.a, deck, ids, prof)
        agentB, scB = make_agent(args.b, deck, ids, prof)
        w = l = d = 0
        # per-seat tally: the aggregate hid that mega_lucario's LM went 8-7 when it always
        # moved first (tools/diag_pilot.py) but 4-36 with seats alternating. A pilot that
        # collapses in one seat is a different bug from a pilot that is simply weaker.
        seat = {0: [0, 0], 1: [0, 0]}
        t0 = time.time()
        verdict = "undecided"
        for g in range(args.max_games):
            # swap seats every game: first-player advantage is a systematic bias, and
            # alternating cancels it without needing to know how large it is
            b_seat = 1 if g % 2 == 0 else 0
            if b_seat == 1:
                r = play(agentA, agentB, ids, ids)
                b_won = (r == 1)
            else:
                r = play(agentB, agentA, ids, ids)
                b_won = (r == 0)
            if r is not None:
                seat[b_seat][0 if b_won else 1] += 1
            if r is None:
                d += 1
            elif b_won:
                w += 1
            else:
                l += 1
            if w + l >= 20:
                sup, ni, verdict = sprt(w, l, args.p0, args.p1, args.alpha, args.beta,
                                        args.margin)
                if verdict != "undecided":
                    break
            if (g + 1) % 20 == 0:
                sup, ni, _v = sprt(w, l, args.p0, args.p1, args.alpha, args.beta, args.margin)
                print("  %-22s %3d games  B %3d-%-3d (%.1f%%) draws %d  sup %+.2f ni %+.2f  %.0fs"
                      % (deck, g + 1, w, l, 100.0 * w / max(1, w + l), d, sup, ni,
                         time.time() - t0), flush=True)
        n = w + l
        p = w / max(1, n)
        se = math.sqrt(0.25 / max(1, n))
        sup, ni, verdict = sprt(w, l, args.p0, args.p1, args.alpha, args.beta, args.margin)
        print("%-24s B %d-%d = %.1f%% (95%% CI %.1f-%.1f)  draws %d  sup %+.2f ni %+.2f -> %s   %.0fs"
              % (deck, w, l, 100 * p, 100 * (p - 1.96 * se), 100 * (p + 1.96 * se), d,
                 sup, ni, verdict, time.time() - t0), flush=True)
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
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"a": args.a, "b": args.b, "decks": results}, f, indent=1)
        print("-> %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
