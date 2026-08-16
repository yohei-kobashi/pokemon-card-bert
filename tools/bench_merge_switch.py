#!/usr/bin/env python3
"""Can the server have merged weights AND eight adapters?

    PYTHONPATH=cg-lib:tools:. python3 tools/bench_merge_switch.py --adapters /root/out/lora_a,/root/out/lora_b

Folding a LoRA into the base is the largest single speedup this workload has (mirror_match
measured 134 -> 71 ms per decision) and score_server.py currently forgoes it, because eight live
adapters cannot all be folded into one set of weights. But PEFT can merge and UNMERGE in place,
so the real question is not "merged or not" -- it is whether a switch costs less than the
decisions it buys.

    switch cost = unmerge(current) + merge(next)
    decisions saved per switch = (run length) x (unmerged ms - merged ms)

If a switch is ~1 s and a decision saves ~60 ms, a run of 17 decisions on one adapter breaks
even. instance1's gate plays eight opponents, so whether runs are that long depends on how the
server orders its queue -- which is something we control.

Measures both halves, plus a correctness check: after unmerge->merge->unmerge cycles, does the
model still score identically? Repeated in-place W +/- BA in bf16 accumulates rounding, and a
drift that only shows after a hundred switches is exactly the kind of thing that never gets
attributed to the right cause.
"""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters", required=True, help="comma-separated adapter dirs")
    ap.add_argument("--base", default="unsloth/Qwen3-4B-Base")
    ap.add_argument("--tokens", type=int, default=368)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--pristine-on-gpu", action="store_true",
                    help="hold the snapshot in VRAM (~6.8 GiB) -- a device-to-device copy is "
                         "far cheaper than the host round trip, and 48 GiB has room")
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    paths = [p for p in a.adapters.split(",") if p]
    names = [os.path.basename(p.rstrip("/")) for p in paths]
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    for i, (n, p) in enumerate(zip(names, paths)):
        if i == 0:
            model = PeftModel.from_pretrained(model, p, adapter_name=n)
        else:
            model.load_adapter(p, adapter_name=n)   # returns _IncompatibleKeys, NOT the model
    model.eval()
    ids = torch.randint(1000, 50000, (1, a.tokens), device="cuda")

    def fwd_ms(reps=8):
        with torch.no_grad():
            for _ in range(2):
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t0 = time.time()
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
                torch.cuda.synchronize()
                ts.append(1000 * (time.time() - t0))
        return statistics.median(ts)

    def logits():
        with torch.no_grad():
            return torch.log_softmax(
                model(input_ids=ids, use_cache=False).logits[0, -1].float(), -1)

    model.set_adapter(names[0])
    ref = logits()
    unmerged = fwd_ms()

    t0 = time.time()
    model.merge_adapter()
    torch.cuda.synchronize()
    merge_s = time.time() - t0
    merged = fwd_ms()
    drift_merged = float((logits() - ref).abs().max())

    t0 = time.time()
    model.unmerge_adapter()
    torch.cuda.synchronize()
    unmerge_s = time.time() - t0
    drift_back = float((logits() - ref).abs().max())

    print("forward @ %d tokens" % a.tokens)
    print("  unmerged (server today) %7.1f ms" % unmerged)
    print("  merged                  %7.1f ms   %.2fx faster" % (merged, unmerged / merged))
    print("\nswitch cost")
    print("  merge_adapter           %7.0f ms" % (1000 * merge_s))
    print("  unmerge_adapter         %7.0f ms" % (1000 * unmerge_s))
    sw = 1000 * (merge_s + unmerge_s)
    save = unmerged - merged
    print("  full switch             %7.0f ms" % sw)
    if save > 0:
        print("  BREAK-EVEN RUN LENGTH   %7.1f decisions on one adapter" % (sw / save))
    print("\nnumeric drift (bf16 W +/- BA is not exactly reversible)")
    print("  after merge             %.4g" % drift_merged)
    print("  after unmerge           %.4g   (should return to ~0)" % drift_back)

    worst = drift_back
    for c in range(a.cycles):
        model.merge_adapter()
        model.unmerge_adapter()
        worst = max(worst, float((logits() - ref).abs().max()))
    print("  after %d more cycles     %.4g %s"
          % (a.cycles, worst,
             "" if worst < 0.05 else "  <-- DRIFTS: unmerge cannot restore the base"))

    # --- the design that survives that: never unmerge, restore from a pristine copy ----------
    #
    # W += BA rounds once into bf16 and throws away what W was, so W -= BA lands somewhere else.
    # Keeping the untouched weights in HOST memory and copying them back is exact by
    # construction, and 629 GiB of RAM makes it free in the resource that is scarce here.
    print("\npristine-restore design (snapshot the base, never unmerge)")
    model.unmerge_adapter()
    mods = [m for m in model.modules()
            if hasattr(m, "base_layer") and hasattr(getattr(m, "base_layer"), "weight")
            and hasattr(m, "merged_adapters")]
    dev = "cuda" if a.pristine_on_gpu else "cpu"
    t0 = time.time()
    pristine = [m.base_layer.weight.detach().to(dev, copy=True) for m in mods]
    nbytes = sum(p.numel() * p.element_size() for p in pristine)
    print("  %d LoRA-target weights, %.2f GiB held on %s (%.1f s to snapshot)"
          % (len(mods), nbytes / 2 ** 30, dev, time.time() - t0))

    def restore():
        """Put the weights back AND tell PEFT they are unmerged.

        The bookkeeping half is not optional and cost this benchmark a wrong answer: PEFT keeps
        a `merged_adapters` list per layer, and `set_adapter` on a model it believes is merged
        silently calls unmerge FIRST -- subtracting BA from weights that had already been
        restored. The result drifted by 19.94 and looked like the design failing, when it was
        the harness."""
        with torch.no_grad():
            for m, p in zip(mods, pristine):
                m.base_layer.weight.copy_(p, non_blocking=True)
                m.merged_adapters = []
        torch.cuda.synchronize()

    restore()
    t0 = time.time()
    restore()
    rest_s = time.time() - t0
    print("  restore                 %7.0f ms" % (1000 * rest_s))
    print("  switch = restore+merge  %7.0f ms   break-even %.1f decisions"
          % (1000 * (rest_s + merge_s), 1000 * (rest_s + merge_s) / max(1e-9, save)))

    # Exactness: the same adapter, merged from pristine weights, must give the SAME logits
    # every time no matter what was served in between.
    restore()
    model.set_adapter(names[0])
    model.merge_adapter()
    first = logits()
    worst2 = 0.0
    for c in range(a.cycles):
        restore()
        model.set_adapter(names[(c + 1) % len(names)])
        model.merge_adapter()
        logits()                                     # serve something from the other adapter
        restore()
        model.set_adapter(names[0])
        model.merge_adapter()
        worst2 = max(worst2, float((logits() - first).abs().max()))
    print("  drift over %d switches   %.4g %s"
          % (a.cycles, worst2,
             "-- EXACT, merged serving is usable" if worst2 < 1e-6
             else "  <-- still drifts, do not merge"))


if __name__ == "__main__":
    main()
