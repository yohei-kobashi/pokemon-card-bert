#!/usr/bin/env python3
"""Why is one 4B decision 134 ms, and does batching really not help?

    PYTHONPATH=cg-lib:tools:. python3 tools/bench_prefill.py --adapter /root/out/lora_marnie_grimmsnarl_r1

mirror_match records "batch 4 is 1.05x over batch 1 and batch 32 is WORSE (53.6 vs 45.2
ms/decision)" and explains it as the prefill already using "most of this card's bf16
throughput", against a peak of ~91 TFLOPS. That reasoning does not survive checking:

  * 91 TFLOPS is the FP32 figure for an RTX 6000 Ada. This is an RTX 5880 Ada, and the workload
    runs on BF16 tensor cores -- roughly twice the FP32 rate. So the quoted peak is both the
    wrong card and the wrong unit, which makes ~45 ms look like a ceiling when it may be half.
  * A compute-bound GEMM getting SLOWER per item as the batch grows is not what saturation looks
    like. Saturation flattens a curve; it does not bend it downward. Something else dominates at
    batch 32, and the candidates are all configuration: the attention implementation, padding to
    the longest sequence, KV-cache allocation, and where the logits are computed.

So this measures instead of arguing. It reports achieved TFLOP/s against a peak MEASURED on this
card with a plain GEMM, so the comparison is to something real rather than to a spec sheet.

It also CHECKS THE ANSWERS. A batched prefill that pads is only useful if it produces the same
log-probs as the unbatched one; left- vs right-padding and an attention mask that a kernel
ignores are exactly the kind of thing that speeds a benchmark up by quietly computing nonsense.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def require_idle(force=False, max_util=12, max_mib=3000, samples=5):
    """Refuse to produce numbers on a card someone else is using.

    This exists because it already went wrong. The first sweep here ran while another process
    held the GPU at 100%, and it produced a clean-looking table that could not answer its own
    question: "is there idle capacity a bigger batch could fill" is unanswerable when the card
    is already saturated by someone else, so batch 32 coming out at 0.88x said nothing about
    batching and everything about queueing. The measured "peak" was 53.8 TFLOP/s, roughly half
    what this card should do, and every absolute number inherited that.

    A benchmark that quietly reports contaminated numbers is worse than one that refuses."""
    import subprocess
    q = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    us, ms = [], []
    for _ in range(samples):
        out = subprocess.run(q, capture_output=True, text=True).stdout.strip().splitlines()[0]
        u, m = [int(x) for x in out.split(",")]
        us.append(u)
        ms.append(m)
        time.sleep(0.4)
    busy = max(us) > max_util or max(ms) > max_mib
    print("GPU before starting: util %d-%d%%, %d MiB used" % (min(us), max(us), max(ms)))
    if busy and not force:
        raise SystemExit(
            "REFUSING TO MEASURE: the card is in use (util up to %d%%, %d MiB). Numbers taken "
            "now cannot answer whether batching finds idle capacity -- there is none to find. "
            "Wait for the card, or pass --force and label the output as contended."
            % (max(us), max(ms)))
    if busy:
        print("*** --force: these numbers are CONTENDED and must not be compared to idle ones ***")


def peak_gemm(torch, dtype, secs=1.5):
    """The card's real bf16 ceiling, measured. n=8192 square GEMM is big enough to be pure
    tensor-core work and small enough to fit anywhere."""
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0, it = time.time(), 0
    while time.time() - t0 < secs:
        a @ b
        it += 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    return 2.0 * n ** 3 * it / dt / 1e12


def model_flops(cfg, ntok, batch):
    """Forward FLOPs for a prefill, counting the matmuls that matter."""
    L, d, ffn = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
    hd = getattr(cfg, "head_dim", d // cfg.num_attention_heads)
    qd, kd = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    per_layer = 2 * (d * qd + 2 * d * kd + qd * d + 3 * d * ffn)       # projections + MLP
    attn = 2 * 2 * ntok * qd                                          # QK^T and AV, per token
    return batch * ntok * L * (per_layer + attn) / 2e12 * 2           # -> TFLOP, causal ~ /2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="", help="load this LoRA (unmerged, as the server does)")
    ap.add_argument("--base", default="unsloth/Qwen3-4B-Base")
    ap.add_argument("--tokens", default="368",
                    help="comma list. Real decisions are ~368 tokens; a bigger prompt makes the "
                         "GEMM taller and is the case LEAST likely to show a batching gain, so "
                         "measuring only at 800 stacks the deck against batching.")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--attn", default="sdpa,eager", help="implementations to compare")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--merge", action="store_true", help="also time the merged LoRA")
    ap.add_argument("--force", action="store_true", help="measure even on a busy card")
    a = ap.parse_args()

    require_idle(a.force)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(a.base)
    peak = peak_gemm(torch, torch.bfloat16)
    print("card        %s" % torch.cuda.get_device_name(0))
    print("peak bf16   %.1f TFLOP/s  (measured, 8192^3 GEMM)" % peak)
    print("free VRAM   %.1f GiB\n" % (torch.cuda.mem_get_info()[0] / 2 ** 30))

    batches = [int(x) for x in a.batches.split(",") if x]
    for impl in [x for x in a.attn.split(",") if x]:
        print("=== attn_implementation=%s ===" % impl)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                a.base, dtype=torch.bfloat16, device_map="cuda", attn_implementation=impl)
        except Exception as e:                                           # noqa: BLE001
            print("  unavailable: %s\n" % e)
            continue
        tag = "base"
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
            tag = "lora"
            if a.merge:
                model = model.merge_and_unload()
                tag = "merged"
        model.eval()
        for ntok in [int(x) for x in a.tokens.split(",") if x]:
            print("  -- %d tokens --" % ntok)
            print("  %-6s %-7s %8s %10s %9s %8s %s"
                  % ("batch", "weights", "ms/dec", "dec/s", "TFLOP/s", "MFU", "vs batch1"))
            base_ms = None
            for bs in batches:
                ids = torch.randint(1000, 50000, (bs, ntok), device="cuda")
                try:
                    with torch.no_grad():
                        for _ in range(2):
                            model(input_ids=ids, use_cache=False, logits_to_keep=1)
                        torch.cuda.synchronize()
                        t0 = time.time()
                        for _ in range(a.reps):
                            model(input_ids=ids, use_cache=False, logits_to_keep=1)
                        torch.cuda.synchronize()
                    dt = (time.time() - t0) / a.reps
                except torch.cuda.OutOfMemoryError:
                    print("  %-6d %-7s OOM" % (bs, tag))
                    torch.cuda.empty_cache()
                    continue
                ms = 1000 * dt / bs
                tf = model_flops(cfg, ntok, bs) / dt
                base_ms = base_ms or ms
                print("  %-6d %-7s %8.1f %10.1f %9.1f %7.0f%% %8.2fx"
                      % (bs, tag, ms, bs / dt, tf, 100 * tf / peak, base_ms / ms))
                del ids
                torch.cuda.empty_cache()
        # Does a padded batch give the SAME answer as one-at-a-time?
        check_padding(torch, model, int(a.tokens.split(",")[0]))
        del model
        torch.cuda.empty_cache()
        print()


def check_padding(torch, model, ntok):
    """Real decisions have different prompt lengths, so a batch must pad -- and a padded batch
    is only useful if it gives the SAME answer as one-at-a-time.

    The first run of this failed (3/4 argmax, max|d| 3.76) and the cause is not the mask: it is
    RoPE. With left padding, the default position_ids are 0..L-1 over the padded row, so every
    real token sits at a position shifted by the pad width and the rotary embedding encodes a
    different place in the sequence. The mask correctly stops attention to the pads and the
    answer is still wrong.

    The fix is to derive positions from the mask -- `cumsum(-1) - 1`, clamped -- so the first
    REAL token is position 0 in every row. Both are measured here, because "batching is unsafe"
    and "batching was called wrongly" lead to opposite decisions."""
    lens = sorted({max(16, ntok - d) for d in (0, 137, 311, 512)}, reverse=True)
    # Realistic ids, not randint. On random input the distribution is flat and its deep tail is
    # numerically unstable, so max|d| over 154k entries reports ~1.9 while every argmax agrees --
    # a number that looks like a correctness failure and is really just the tail. What decides a
    # MOVE is the ranking of the handful of candidate tokens, so that is what is measured.
    txt = ("[ACT]\nDECK win[%s] || SEL MAIN n1-1 :: 0=attach:c7@ACTIVE 1=play:c9@BENCH0 2=end"
           % " ".join("c%d" % (i % 900 + 1) for i in range(max(lens))))
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
    full = tk(txt, add_special_tokens=False)["input_ids"]
    seqs = [torch.tensor([(full * 4)[:n]], device="cuda") for n in lens]
    K = 32                                   # a real menu is ~6 candidates; 32 is generous
    with torch.no_grad():
        solo = [torch.log_softmax(model(input_ids=s, use_cache=False,
                                        logits_to_keep=1).logits[0, -1].float(), -1)
                for s in seqs]
        top = [torch.topk(x, K).indices for x in solo]
        m = max(lens)
        pad = torch.zeros(len(lens), m, dtype=torch.long, device="cuda")
        mask = torch.zeros(len(lens), m, dtype=torch.long, device="cuda")
        for i, s in enumerate(seqs):                       # LEFT pad: last position is real
            pad[i, m - s.shape[1]:] = s[0]
            mask[i, m - s.shape[1]:] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)

        for tag, kw in (("mask only          ", {}),
                        ("mask + position_ids", {"position_ids": pos})):
            out = model(input_ids=pad, attention_mask=mask, use_cache=False,
                        logits_to_keep=1, **kw)
            bat = [torch.log_softmax(out.logits[i, -1].float(), -1) for i in range(len(lens))]
            same = sum(int(x.argmax()) == int(y.argmax()) for x, y in zip(solo, bat))
            # gap and order over the tokens a candidate set would actually be drawn from
            dtop = max(float((x[t] - y[t]).abs().max()) for x, y, t in zip(solo, bat, top))
            order = sum(bool(torch.equal(torch.topk(y[t], 8).indices,
                                         torch.arange(8, device=y.device)))
                        for y, t in zip(bat, top))
            ok = same == len(lens) and order == len(lens) and dtop < 0.5
            print("  padding %s: %d/%d argmax, %d/%d top-8 order kept, max|d| over top-%d "
                  "%.4g %s" % (tag, same, len(lens), order, len(lens), K, dtop,
                               "OK" if ok else "<-- CHANGES MOVES"))


if __name__ == "__main__":
    main()
