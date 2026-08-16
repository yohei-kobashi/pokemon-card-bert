#!/usr/bin/env python3
"""Serve the per-deck Qwen3-4B LoRAs over HTTP, so instance1 can play against them.

    # instance2 (the 48 GiB card, where the adapters live)
    python3 tools/score_server.py --port 8077 --from-registry --verify alakazam_nz

    # instance1
    ssh -N -L 8077:127.0.0.1:8077 -i /root/.ssh/id_i2 -p 19839 root@175.155.64.145 &
    python3 tools/mirror_match.py ... --opp-spec 'remote:http://127.0.0.1:8077|lora_alakazam_nz_r1'

WHY ONE SHARED BASE. Ten adapters is ten pilots but only one set of base weights: 4B in bf16 is
~8 GiB and eight copies do not fit in 48 GiB, let alone in instance1's 24 GiB next to a training
run. PEFT holds every LoRA on one base and switches with `set_adapter`, which is a pointer swap.

WHAT ELSE HAS TO SWAP, and would be silent if it did not. Each checkpoint carries its OWN
`domain_embeddings.pt` -- 3,067 x 2,560 rows for the card tokens, trained per deck. They are
close (max |diff| 0.0036 between two adapters) but not equal, and nothing errors if the wrong
ones are live: the model simply reads a slightly different board than it was trained on. So
`use()` swaps the embedding rows together with the adapter, and they are re-copied every switch
rather than assumed. 15.7 MiB of device-to-device copy, well under a millisecond.

WHAT IS NOT MERGED, and why the request is ~2x its floor. Folding a LoRA into the base weights
was worth 134 -> 71 ms per decision, and it is exactly the thing eight live adapters forbid.
Measured cost is reported by --bench; the alternative (merge on every switch) is worse under the
interleaved traffic eight concurrent opponent decks produce.

NO BATCHING, DELIBERATELY. mirror_match documents the measurement: batch 4 is 1.05x over batch 1
and batch 32 is WORSE (53.6 vs 45.2 ms/decision), because an ~800-token prefill of a 4B is
already most of this card's bf16 throughput. A batching queue here would add latency and
scheduling risk to buy nothing. GPU work is simply serialised under one lock; the threads exist
so that twenty game workers can queue without blocking each other's sockets.

TWO DECODING SCHEMES ARE LIVE. Seven of the ten adapters ship `cardfirst_vocab.json` and answer
with a CARD token; three answer with a menu INDEX. The scheme is read from each checkpoint, as
in QwenScorer -- a model and its decoder cannot be paired wrongly.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tools", "instance")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROBE = ("[ACT]\nDECK win[c743x4] eng[c13] T3.2 ME A[c5:100/100] pz3 dk20 bm5 H[c7,c9] "
         "|| SEL MAIN n1-1 :: 0=attach:c7@ACTIVE 1=end")


class SharedQwen:
    """One base, every adapter. `scorer(name)` returns an object with QwenScorer's `score`."""

    def __init__(self, base, adapters, maxlen=1024, dtype="bf16", device="cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.lock = threading.Lock()
        self.maxlen = maxlen
        self.device = device
        self.cur = None
        self.adapters = dict(adapters)
        if not self.adapters:
            raise SystemExit("no adapters given")

        names = list(self.adapters)
        first = self.adapters[names[0]]
        # The tokenizer is the CHECKPOINT's, not the base's: it carries the 3,067 card tokens.
        # All ten shipped adapters share one file (single md5), and a mismatch is checked below
        # rather than assumed -- two vocabularies on one embedding matrix is silent nonsense.
        tok = AutoTokenizer.from_pretrained(first)
        self.tk = getattr(tok, "tokenizer", tok)
        if self.tk.pad_token is None:
            self.tk.pad_token = self.tk.eos_token

        cfg = json.load(open(os.path.join(first, "adapter_config.json")))
        base = base or (cfg.get("base_model_name_or_path") or "").replace(
            "-unsloth-bnb-4bit", "")
        print("[srv] base %s" % base, flush=True)
        td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.dtype, self.base_id = td, base
        model = AutoModelForCausalLM.from_pretrained(base, dtype=td, device_map=device)

        # Resize ONCE to the checkpoints' vocabulary. mean_resizing=False for the same reason as
        # QwenScorer: every new row is overwritten by use() anyway, and the default's Cholesky
        # both allocates a large temporary and needs cuSOLVER.
        blob = torch.load(os.path.join(first, "domain_embeddings.pt"), map_location="cpu")
        self.n_base, n_new = int(blob["n_base"]), blob["rows"].shape[0]
        if len(self.tk) != self.n_base + n_new:
            print("[srv] warning: tokenizer has %d entries, embeddings want %d"
                  % (len(self.tk), self.n_base + n_new), flush=True)
        model.resize_token_embeddings(self.n_base + n_new, mean_resizing=False)

        self.rows, self.meta = {}, {}
        for i, name in enumerate(names):
            path = self.adapters[name]
            b = torch.load(os.path.join(path, "domain_embeddings.pt"), map_location="cpu")
            if int(b["n_base"]) != self.n_base or b["rows"].shape[0] != n_new:
                raise SystemExit("%s has a different vocabulary layout (%d+%d vs %d+%d) -- it "
                                 "cannot share this base" % (name, int(b["n_base"]),
                                                             b["rows"].shape[0], self.n_base,
                                                             n_new))
            t2 = AutoTokenizer.from_pretrained(path)
            t2 = getattr(t2, "tokenizer", t2)
            if len(t2) != len(self.tk):
                raise SystemExit("%s ships a different tokenizer (%d vs %d entries)"
                                 % (name, len(t2), len(self.tk)))
            self.rows[name] = b["rows"].to(device=device, dtype=td)
            cf = None
            cfp = os.path.join(path, "cardfirst_vocab.json")
            if os.path.exists(cfp):
                cf = json.load(open(cfp))
            self.meta[name] = {"cf": cf, "scheme_b": bool(cf and cf.get("scheme") == "b"),
                               "path": path}
            if i == 0:
                model = PeftModel.from_pretrained(model, path, adapter_name=name)
            else:
                model.load_adapter(path, adapter_name=name)
            print("[srv] loaded %-26s %s" % (name, "card-first" if cf else "index"), flush=True)
        model.eval()
        self.model = model
        self.emb = model.get_input_embeddings()

        self._klast, self.kv = self._probe()
        print("[srv] logits_to_keep=%s kv_reuse=%s" % (bool(self._klast), self.kv), flush=True)
        self.n, self.t, self.switches = 0, 0.0, 0     # before use(): it counts switches
        self._views = {n: self._view(n) for n in names}
        self.use(names[0])

    # -- capability probe (same shape as QwenScorer's: check, never assume) --------------
    def _probe(self):
        torch = self.torch
        klast, kv = {}, False
        try:
            ids = self.tk(PROBE, add_special_tokens=False)["input_ids"]
            t = torch.tensor([ids], device=self.model.device)
            with torch.no_grad():
                ref = torch.log_softmax(
                    self.model(input_ids=t, use_cache=False).logits[0, -1, :].float(), -1)
                try:
                    o = self.model(input_ids=t, use_cache=False, logits_to_keep=1)
                    lp = torch.log_softmax(o.logits[0, -1, :].float(), -1)
                    if o.logits.shape[1] == 1 and int(lp.argmax()) == int(ref.argmax()) \
                            and float((lp - ref).abs().max()) < 0.5:
                        klast = {"logits_to_keep": 1}
                except Exception as e:                                       # noqa: BLE001
                    print("[srv] logits_to_keep unsupported (%s)" % type(e).__name__, flush=True)
                o1 = self.model(input_ids=torch.tensor([ids[:-1]], device=self.model.device),
                                use_cache=True, **klast)
                pkv = getattr(o1, "past_key_values", None)
                if pkv is not None and hasattr(pkv, "crop"):
                    o2 = self.model(input_ids=torch.tensor([[ids[-1]]],
                                                           device=self.model.device),
                                    past_key_values=pkv, use_cache=True, **klast)
                    lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), -1)
                    kv = (int(lp2.argmax()) == int(ref.argmax())
                          and float((lp2 - ref).abs().max()) < 0.5)
        except Exception as e:                                               # noqa: BLE001
            print("[srv] probe failed (%s) -- plain forward" % e, flush=True)
        return klast, kv

    def _view(self, name):
        """A QwenScorer that owns no weights.

        Built with ``__new__`` instead of ``__init__`` on purpose: __init__ loads a base model,
        and the whole point here is that ten views share one. Everything ``score`` touches is
        assigned below, so the SCORING CODE is literally the one instance1 runs locally -- both
        the index trie and the card-first tie-break, kept in one place so they cannot drift.
        """
        from mirror_match import QwenScorer
        v = QwenScorer.__new__(QwenScorer)
        m = self.meta[name]
        v.model, v.tk, v.torch, v.maxlen = self.model, self.tk, self.torch, self.maxlen
        v._klast, v.kv = self._klast, self.kv
        v.cf, v.scheme_b = m["cf"], m["scheme_b"]
        from eval_teacher import score_decision
        v._score_decision = score_decision
        v.n, v.t = 0, 0.0
        return v

    def use(self, name):
        if name == self.cur:
            return
        self.model.set_adapter(name)
        with self.torch.no_grad():
            self.emb.weight[self.n_base:] = self.rows[name]
        self.cur = name
        self.switches += 1

    def score(self, name, prompt, cands):
        with self.lock:
            self.use(name)
            t0 = time.time()
            out = self._views[name].score(prompt, cands)
            self.t += time.time() - t0
            self.n += 1
            return out

    def stats(self):
        return {"decisions": self.n, "seconds": round(self.t, 1),
                "ms_per_decision": round(1000.0 * self.t / max(1, self.n), 1),
                "adapter_switches": self.switches, "current": self.cur,
                "adapters": sorted(self.adapters)}


def verify(shared, names, tol=0.02):
    """Is a shared-base view the same pilot as a standalone build of that checkpoint?

    The whole handover rests on this. If swapping rows-plus-adapter is not equivalent to loading
    the checkpoint alone, then instance1 gates its reranker against opponents that exist nowhere
    else -- and nothing would say so, because a wrong-but-plausible opponent only shows up as a
    win rate. Two things could break it and both are silent: PEFT could leave another adapter
    active, and the embedding rows could be some other deck's.

    Compares the FULL next-token log-prob vector, not just the candidate scores. Both decoding
    schemes read that vector (the index trie takes digit tokens out of it, card-first takes card
    tokens), so agreeing on all 154k entries covers both without needing real menus. The
    reference is loaded standalone, one adapter on a fresh base, which is the path a local run
    takes.
    """
    torch = shared.torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    prompts = [PROBE,
               PROBE.replace("pz3", "pz1").replace("T3.2", "T7.1"),
               PROBE.replace("0=attach:c7@ACTIVE 1=end",
                             "0=attach:c7@ACTIVE 1=play:c9@BENCH0 2=retreat 3=end")]
    ids = [torch.tensor([shared.tk(p, add_special_tokens=False)["input_ids"]],
                        device=shared.device) for p in prompts]

    # every shared view first, so an adapter left active by the previous call would show up
    mine = {}
    for name in names:
        with shared.lock:
            shared.use(name)
            with torch.no_grad():
                mine[name] = [torch.log_softmax(
                    shared.model(input_ids=t).logits[0, -1, :].float(), -1) for t in ids]

    worst, bad = 0.0, []
    for name in names:
        print("[verify] standalone build of %s" % name, flush=True)
        m = AutoModelForCausalLM.from_pretrained(shared.base_id, dtype=shared.dtype,
                                                 device_map=shared.device)
        b = torch.load(os.path.join(shared.adapters[name], "domain_embeddings.pt"),
                       map_location="cpu")
        m.resize_token_embeddings(shared.n_base + b["rows"].shape[0], mean_resizing=False)
        with torch.no_grad():
            m.get_input_embeddings().weight[shared.n_base:] = b["rows"].to(
                device=shared.device, dtype=shared.dtype)
        m = PeftModel.from_pretrained(m, shared.adapters[name])
        m.eval()
        for i, t in enumerate(ids):
            with torch.no_grad():
                ref = torch.log_softmax(m(input_ids=t).logits[0, -1, :].float(), -1)
            d = float((ref - mine[name][i]).abs().max())
            same = int(ref.argmax()) == int(mine[name][i].argmax())
            worst = max(worst, d)
            if not same or d > tol:
                bad.append((name, i, d, same))
            print("[verify]   %-24s prompt %d  argmax %s  max|d|=%.4g"
                  % (name, i, "same" if same else "DIFF", d), flush=True)
        del m
        if shared.device == "cuda":
            torch.cuda.empty_cache()
    if bad:
        raise SystemExit("[verify] FAILED (%s), worst max|d| %.4g > %.4g. The shared base is "
                         "NOT the same pilot -- do not gate against it." % (bad, worst, tol))
    print("[verify] OK -- %d adapters reproduce their standalone checkpoints (worst %.4g)"
          % (len(names), worst), flush=True)


def bench(shared, name, n=30, ntok=800, ncand=6):
    """ms per decision, under the traffic shape a gate actually produces."""
    filler = " ".join("c%d" % (i % 900 + 1) for i in range(ntok))
    base = "[ACT]\nDECK win[%s] || SEL MAIN n1-1 :: " % filler
    cands = ["opt%d" % i for i in range(ncand)]
    names = sorted(shared.adapters)
    for tag, seq in (("one adapter", [name] * n),
                     ("round-robin", [names[i % len(names)] for i in range(n)])):
        shared.score(seq[0], base + "0=end", cands)                 # warm
        t0 = time.time()
        for i, a in enumerate(seq):
            shared.score(a, base + "%d=end" % i, cands)
        dt = (time.time() - t0) / n
        print("[bench] %-12s %6.1f ms/decision  -> %.1f decisions/s"
              % (tag, 1000 * dt, 1.0 / dt), flush=True)


def make_handler(shared, token):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                 # one line per decision would drown the log
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/stats", ""):
                return self._send(200, shared.stats())
            return self._send(404, {"error": "no such path"})

        def do_POST(self):
            if token and self.headers.get("X-Score-Token") != token:
                return self._send(403, {"error": "bad token"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:                                       # noqa: BLE001
                return self._send(400, {"error": "bad body: %s" % e})
            path = self.path.rstrip("/")
            if path in ("/health", "/stats"):
                return self._send(200, shared.stats())
            if path != "/score":
                return self._send(404, {"error": "no such path"})
            name = req.get("adapter") or shared.cur
            if name not in shared.adapters:
                # Loud, not defaulted. Serving the wrong adapter is the one failure that would
                # not show up as an error anywhere downstream -- only as a win rate.
                return self._send(400, {"error": "unknown adapter %r; have %s"
                                                 % (name, sorted(shared.adapters))})
            t0 = time.time()
            try:
                sc = shared.score(name, req["prompt"], req.get("cands") or [])
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                return self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})
            return self._send(200, {"scores": sc, "wire": time.time() - t0,
                                    "adapter": name})
    return H


def from_registry(reg_path=None, only=None):
    """Every deck the registry points at a qwen adapter -> {adapter dir name: path}."""
    from lm import registry as R
    reg = R.load(reg_path)
    out = {}
    for deck in sorted(reg.get("decks") or {}):
        try:
            r = R.resolve(deck, reg, require_exists=True)
        except R.RegistryError as e:
            print("[srv] skipping %s: %s" % (deck, e), flush=True)
            continue
        kind, _, path = r["spec"].partition(":")
        if kind != "qwen":
            continue
        if only and deck not in only:
            continue
        out[os.path.basename(path.rstrip("/"))] = path
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", action="append", default=[],
                    help="name=/path, repeatable")
    ap.add_argument("--from-registry", action="store_true",
                    help="serve every qwen: entry in models/adapters.json")
    ap.add_argument("--decks", default="", help="--from-registry: restrict to these decks")
    ap.add_argument("--base", default="", help="default: read from the first adapter's config")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 by default: reach it over an ssh tunnel, not the open "
                         "internet. There is no transport security here beyond --token.")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--token", default=os.environ.get("PTCG_SCORE_TOKEN", ""))
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--device", default="cuda", help="cpu runs the correctness checks without "
                                                     "taking VRAM off a training run")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--verify", default="",
                    help="comma-separated adapters (or 'all') to check against standalone "
                         "builds before serving")
    ap.add_argument("--bench", type=int, default=0, help="time N decisions before serving")
    ap.add_argument("--no-serve", action="store_true", help="verify/bench, then exit")
    a = ap.parse_args()

    ads = {}
    if a.from_registry:
        ads.update(from_registry(only=[d for d in a.decks.split(",") if d] or None))
    for s in a.adapter:
        name, _, path = s.partition("=")
        ads[name if path else os.path.basename(name.rstrip("/"))] = path or name
    if not ads:
        raise SystemExit("nothing to serve: pass --adapter or --from-registry")
    print("[srv] serving %d adapters: %s" % (len(ads), ", ".join(sorted(ads))), flush=True)

    shared = SharedQwen(a.base or None, ads, maxlen=a.maxlen, dtype=a.dtype, device=a.device)
    if a.verify:
        verify(shared, sorted(ads) if a.verify == "all"
               else [n for n in a.verify.split(",") if n])
    if a.bench:
        bench(shared, sorted(ads)[0], n=a.bench)
    if a.no_serve:
        return

    srv = ThreadingHTTPServer((a.host, a.port), make_handler(shared, a.token))
    srv.daemon_threads = True
    print("[srv] listening on %s:%d%s" % (a.host, a.port,
                                          " (token required)" if a.token else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[srv] %s" % json.dumps(shared.stats()), flush=True)


if __name__ == "__main__":
    main()
