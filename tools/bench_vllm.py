#!/usr/bin/env python3
"""Does vLLM beat the tuned transformers scorer, and is it exact enough to use?

The earlier "batching is dead" measurement was HF's PADDED batching: every sequence in a batch is
left-padded to the batch maximum, so with prompts at mean 368 / p90 460 tokens a batch of 32 does
real work on pad. That is why batch 32 came out WORSE than batch 1. vLLM batches RAGGED, adds
CUDA graphs and prefix caching, and the tuned scorer still only reaches ~52 TFLOP/s against this
card's ~91 TFLOPS bf16 peak -- so the question was not settled and this settles it.

Reference to beat: 65.8 ms/decision = 15.2 decisions/s, single stream, hf backend with the LoRA
merged, logits_to_keep and KV reuse on.

Two ways to get the numbers the card-first scheme needs, because they trade exactness against
speed differently:

  topk     one request per decision, `logprobs=K`. One prefill per decision, fully batched. But
           it only returns the K most likely tokens, so a candidate the model rates poorly can
           fall outside and its score is unknown. COVERAGE IS REPORTED, not assumed -- a missing
           candidate is a silently wrong move, not a rounding error.
  extend   one request per unique first token, prompt + that token, reading `prompt_logprobs` at
           the last position. Exact for every candidate however unlikely. Costs more requests but
           they share a prefix, so it is a test of whether prefix caching actually pays (vLLM has
           historically disabled prefix reuse for requests that ask for prompt_logprobs -- if it
           still does, this mode will be slow and that IS the finding).

Run with the GPU otherwise idle.
"""
import argparse
import gzip
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

_OPT = re.compile(r"(?:^| )(\d+)=(\S+)")


def load(path, n):
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            c = [t for _, t in _OPT.findall(d["prompt"].rsplit(":: ", 1)[-1])]
            if len(c) >= 2:
                out.append((d["prompt"], c))
                if len(out) >= n:
                    break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="exported standalone HF dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--util", type=float, default=0.80)
    ap.add_argument("--modes", default="topk,extend")
    ap.add_argument("--no-prefix-cache", action="store_true")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from lm.action_token import first_token, second_token, equivalent, to_scheme_b

    dec = load(a.data, a.n)
    tk = AutoTokenizer.from_pretrained(a.model)
    scheme_b = os.path.exists(os.path.join(a.model, "cardfirst_vocab.json")) and \
        json.load(open(os.path.join(a.model, "cardfirst_vocab.json"))).get("scheme") == "b"
    print("[data] %d decisions | scheme_b %s" % (len(dec), scheme_b), flush=True)

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.maxlen,
              gpu_memory_utilization=a.util, max_logprobs=max(a.topk, 32),
              enable_prefix_caching=not a.no_prefix_cache, enforce_eager=False)

    prompts, heads_per = [], []
    for prompt, cands in dec:
        prompts.append(to_scheme_b(prompt) if scheme_b else prompt)
        heads_per.append([first_token(c) for c in cands])
    hid_per = [{h: tk.convert_tokens_to_ids(h) for h in set(hs)} for hs in heads_per]

    modes = [m for m in a.modes.split(",") if m]

    if "topk" in modes:
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=a.topk)
        for _ in range(1):                                   # warm the graphs
            llm.generate(prompts[:8], sp, use_tqdm=False)
        t0 = time.time()
        out = llm.generate(prompts, sp, use_tqdm=False)
        el = time.time() - t0
        covered = full = 0
        for o, hid in zip(out, hid_per):
            lp = o.outputs[0].logprobs[0]
            have = sum(1 for v in hid.values() if v in lp)
            covered += have
            full += (have == len(hid))
        tot = sum(len(h) for h in hid_per)
        print("\n[topk k=%d] %.1f decisions/s   %.2f ms/decision   %.2fx vs 15.2/s"
              % (a.topk, len(dec) / el, 1000 * el / len(dec), (len(dec) / el) / 15.2), flush=True)
        print("  candidate first tokens inside the top %d: %d/%d = %.2f%%"
              % (a.topk, covered, tot, 100.0 * covered / tot), flush=True)
        print("  decisions with EVERY candidate covered: %d/%d = %.2f%%"
              % (full, len(dec), 100.0 * full / len(dec)), flush=True)
        if full < len(dec):
            print("  -> topk alone cannot score every decision; the uncovered ones would need a "
                  "fallback, and a fallback that guesses is a wrong move", flush=True)

    if "extend" in modes:
        # One request per unique first token, each the decision's prompt plus that token, built
        # from TOKEN IDS: appending the token as text and re-tokenising can merge it with the
        # preceding character and score a different token than the one being asked about.
        from vllm import TokensPrompt
        base_ids = [tk(p, add_special_tokens=False)["input_ids"] for p in prompts]
        owner, tp = [], []
        for i, hid in enumerate(hid_per):
            for h, tid in hid.items():
                if tid is None:
                    continue
                owner.append((i, h, tid))
                tp.append(TokensPrompt(prompt_token_ids=base_ids[i] + [tid]))
        sp2 = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        llm.generate(tp[:8], sp2, use_tqdm=False)
        t0 = time.time()
        out2 = llm.generate(tp, sp2, use_tqdm=False)
        el2 = time.time() - t0
        print("\n[extend] %d requests for %d decisions   %.1f decisions/s   %.2f ms/decision"
              " %.2fx vs 15.2/s"
              % (len(tp), len(dec), len(dec) / el2, 1000 * el2 / len(dec),
                 (len(dec) / el2) / 15.2), flush=True)
        got = sum(1 for o in out2 if o.prompt_logprobs and o.prompt_logprobs[-1])
        print("  requests returning a last-position logprob: %d/%d" % (got, len(out2)),
              flush=True)


if __name__ == "__main__":
    main()
