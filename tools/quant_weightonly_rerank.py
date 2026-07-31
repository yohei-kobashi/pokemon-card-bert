"""WEIGHT-ONLY quantization of the reranker ONNX (int4/int8 block-wise), the alternative to
dynamic INT8 -- and the one the measurements point at.

WHY: dynamic INT8 quantizes BOTH weights and activations. On this model the weights are
well behaved (max |w| = 4.44, no outliers) yet dynamic INT8 collapses argmax agreement to
40-67%, and the decision signal it has to preserve is tiny (top1-vs-top2 logit gap: median
0.80, 58% of decisions below 1.0). So the damage is on the ACTIVATION side. Weight-only
quantization keeps activations in fp32 and dequantizes int4/int8 weights per block on the
fly (ORT's MatMulNBits op, same idea as GGUF Q4_K) -- it removes the failing half.

Bonus: int4/block-128 is ~59 MiB for the 111M-param encoder, less than half of INT8's
~111 MiB, so it also buys submission headroom.

The word embedding is a Gather, not a MatMul, so it is untouched here (fp32). Combine with
vocab pruning (only ~3k of 53,339 tokens ever occur) to make it ~14 MiB.

Usage:
  python tools/quant_weightonly_rerank.py --fp32 /root/onnx/rerank_diag/model.onnx \
     --model /root/out/rerank_gte_mp --data <rerank.jsonl.gz> --n 40 --work /root/onnx/wonly
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

# (label, bits, block_size, symmetric, accuracy_level)
# accuracy_level is the MatMulNBits COMPUTE precision: 1=fp32, 2=fp16, 3=bf16, 4=int8.
# Leaving it unset lets ORT pick, and an int8 choice would re-introduce exactly the
# activation quantization that broke dynamic INT8 -- so pin fp32 for the accuracy variants
# and test int8 compute separately as the speed option.
VARIANTS = [
    ("wonly_int8_blk128_fp32c", 8, 128, True, 1),
    ("wonly_int4_blk32_fp32c", 4, 32, True, 1),
    ("wonly_int4_blk128_fp32c", 4, 128, True, 1),
    ("wonly_int4_blk32_asym_fp32c", 4, 32, False, 1),
    ("wonly_int4_blk32_int8c", 4, 32, True, 4),
]


def _mb(n):
    return f"{n / 1024 / 1024:.1f} MiB"


def _gz(path):
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=True) as tf:
        subprocess.run(["tar", "czf", tf.name, "-C", os.path.dirname(path),
                        os.path.basename(path)], check=True)
        return os.path.getsize(tf.name)


def _records(data, n, stride=977):
    recs = []
    with gzip.open(data, "rt") as fh:
        for i, line in enumerate(fh):
            if i % stride == 0:
                recs.append(json.loads(line))
                if len(recs) >= n:
                    break
    return recs


def _quantize(fp32_path, out_path, bits, block, symmetric, accuracy_level):
    """MatMulNBitsQuantizer needs the `onnx_ir` package (its absence surfaces as an odd
    ModuleNotFoundError from the import chain, not as a bad-signature error)."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    model = onnx.load(fp32_path)
    q = MatMulNBitsQuantizer(model, bits=bits, block_size=block, is_symmetric=symmetric,
                             accuracy_level=accuracy_level)
    q.process()
    q.model.save_model_to_file(out_path, use_external_data_format=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.makedirs(args.work, exist_ok=True)
    recs = _records(args.data, args.n)
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager")
    if getattr(ref_model.config, "reference_compile", None):
        ref_model.config.reference_compile = False
    ref_model = ref_model.eval().to(dev)

    batches, refs = [], []
    for r in recs:
        enc = tok([[r["state"], c] for c in r["candidates"]], padding=True,
                  truncation="only_first", max_length=args.max_len, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        att = enc["attention_mask"].astype(np.int64)
        with torch.no_grad():
            out = ref_model(input_ids=torch.tensor(ids).to(dev),
                            attention_mask=torch.tensor(att).to(dev)
                            ).logits.squeeze(-1).float().cpu().numpy()
        batches.append((ids, att))
        refs.append(out)
    del ref_model
    print(f"reference ready: {len(recs)} decisions\n", flush=True)

    keep = set(x for x in args.only.split(",") if x)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    rows = []
    for label, bits, block, sym, acc in VARIANTS:
        if keep and label not in keep:
            continue
        out_path = os.path.join(args.work, f"{label}.onnx")
        if not os.path.exists(out_path):
            try:
                _quantize(args.fp32, out_path, bits, block, sym, acc)
            except Exception as e:
                print(f"{label:28s} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
        sess = ort.InferenceSession(out_path, so, providers=["CPUExecutionProvider"])
        top = 0
        dsum = dmax = 0.0
        for (ids, att), ref in zip(batches, refs):
            got = sess.run(["logits"], {"input_ids": ids, "attention_mask": att})[0].reshape(-1)
            d = float(np.abs(ref - got).max())
            dsum += d
            dmax = max(dmax, d)
            top += int(int(np.argmax(ref)) == int(np.argmax(got)))
        raw, gz = os.path.getsize(out_path), _gz(out_path)
        row = dict(label=label, bits=bits, block=block, symmetric=sym, accuracy_level=acc,
                   argmax=top / len(refs), mean_diff=dsum / len(refs), max_diff=dmax,
                   raw_bytes=raw, gz_bytes=gz)
        rows.append(row)
        print(f"{label:28s} argmax {row['argmax']:6.1%}  |diff| mean {row['mean_diff']:8.4f} "
              f"max {row['max_diff']:8.4f}  raw {_mb(raw):>10s}  tar.gz {_mb(gz):>10s}",
              flush=True)
        del sess

    json.dump(rows, open(os.path.join(args.work, "weightonly.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
