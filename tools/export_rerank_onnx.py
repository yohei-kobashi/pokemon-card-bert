"""Export the trained cross-encoder reranker to ONNX and dynamically quantize it to INT8.

WHY: the reranker must ship inside a Kaggle submission tarball capped at 197.65625 MiB
(207,257,600 bytes) and run on 2 vCPU. llama-cpp-python 0.3.34 (our bundled runtime) has no
ModernBERT support, so the deploy runtime is ONNX Runtime instead.

SIZE ARITHMETIC (why Gather must be quantized too): with the +2971 domain tokens the word
embedding is 53339 x 768 = 41.0M params. Left at fp32 that ONE tensor is 164 MB and the
budget is gone before the encoder (~111M params) is counted. INT8 everywhere ~= 152 MB.

Steps: fp32 ONNX export -> quant_pre_process -> quantize_dynamic(MatMul + Gather) -> verify
against PyTorch on REAL decisions (per-candidate logits, argmax agreement) -> report sizes.

Usage:
  python tools/export_rerank_onnx.py --model /root/out/rerank_gte_mp --out /root/onnx/rerank \
      --data /root/data/rerank/curengine_0724_mp.rerank.jsonl.gz --verify-n 200
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

TAR_CAP = 207257600  # 197.65625 MiB, the measured Kaggle submission cap


def _mb(n):
    return f"{n / 1024 / 1024:.1f} MiB"


def _dir_bytes(d):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(d) for f in fs)


def _gz_bytes(paths, exclude=()):
    """Bytes these paths add to a .tar.gz -- the cap applies to the compressed tarball, and
    the INT8 graph still compresses ~20%, so measuring uncompressed badly over-counts."""
    import subprocess
    import tempfile
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return 0
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=True) as tf:
        cmd = ["tar", "czf", tf.name]
        for e in exclude:
            cmd += [f"--exclude={e}"]
        for p in paths:
            cmd += ["-C", os.path.dirname(p) or ".", os.path.basename(p)]
        subprocess.run(cmd, check=True)
        return os.path.getsize(tf.name)


def export_fp32(model_dir, out_path, max_len, dynamic_seq):
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager")
    # ModernBERT ships reference_compile=True, which wraps submodules in torch.compile and
    # makes the trace non-exportable.
    if getattr(model.config, "reference_compile", None):
        model.config.reference_compile = False
    model.eval()

    seq = max_len if not dynamic_seq else max_len // 2
    ids = torch.randint(0, 1000, (2, seq), dtype=torch.long)
    att = torch.ones((2, seq), dtype=torch.long)
    dyn = {"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
           "logits": {0: "batch"}}
    if dynamic_seq:
        dyn["input_ids"][1] = "seq"
        dyn["attention_mask"][1] = "seq"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model, (ids, att), out_path,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes=dyn, opset_version=17, do_constant_folding=True)
    return out_path


def quantize(fp32_path, int8_path, gather=True, per_channel=True):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32_path.replace(".onnx", ".pre.onnx")
    try:
        quant_pre_process(fp32_path, pre, skip_symbolic_shape=True)
        src = pre
    except Exception as e:                                    # non-fatal; quantize raw graph
        print(f"  quant_pre_process skipped: {type(e).__name__}: {e}", flush=True)
        src = fp32_path
    ops = ["MatMul", "Gather"] if gather else ["MatMul"]
    quantize_dynamic(src, int8_path, weight_type=QuantType.QInt8,
                     op_types_to_quantize=ops, per_channel=per_channel,
                     extra_options={"MatMulConstBOnly": True})
    if src == pre and os.path.exists(pre):
        os.remove(pre)
    return int8_path


def _load_records(data, n):
    """Real decisions: {state, candidates, chosen}. Sample spread across the file."""
    recs = []
    with gzip.open(data, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 977 == 0:                                  # stride-sample, cheap
                recs.append(json.loads(line))
                if len(recs) >= n:
                    break
    return recs


def verify(model_dir, onnx_paths, recs, max_len, pad_to, traced_seq=None):
    """STAGED fidelity check on real decisions, to separate the two failure modes:

      PyTorch fp32  vs  ONNX fp32   -> is the TRACE faithful? (ModernBERT's sliding-window
                                       mask is built from seq_len, so a traced constant would
                                       break every length except the one exported at)
      PyTorch fp32  vs  ONNX int8   -> what does dynamic quantization cost on top?

    Also reports agreement split by whether the batch's padded length equals the traced
    length -- if the trace baked the mask, mismatches concentrate at other lengths.
    """
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager")
    if getattr(model.config, "reference_compile", None):
        model.config.reference_compile = False
    model = model.eval().to(dev)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = {k: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
            for k, p in onnx_paths.items()}

    acc = {k: dict(top=0, n=0, diffs=[], top_at_traced=0, n_at_traced=0) for k in sess}
    for r in recs:
        pairs = [[r["state"], c] for c in r["candidates"]]
        kw = dict(padding="max_length", max_length=pad_to) if pad_to else dict(padding=True)
        enc = tok(pairs, truncation="only_first", max_length=max_len,
                  return_tensors="np", **kw)
        ids = enc["input_ids"].astype(np.int64)
        att = enc["attention_mask"].astype(np.int64)
        with torch.no_grad():
            ref = model(input_ids=torch.tensor(ids).to(dev),
                        attention_mask=torch.tensor(att).to(dev)
                        ).logits.squeeze(-1).float().cpu().numpy()
        at_traced = traced_seq is not None and ids.shape[1] == traced_seq
        for k, s in sess.items():
            got = s.run(["logits"], {"input_ids": ids, "attention_mask": att})[0].reshape(-1)
            a = acc[k]
            a["n"] += 1
            a["diffs"].append(float(np.abs(ref - got).max()))
            hit = int(int(np.argmax(ref)) == int(np.argmax(got)))
            a["top"] += hit
            if at_traced:
                a["n_at_traced"] += 1
                a["top_at_traced"] += hit
    out = {}
    for k, a in acc.items():
        out[k] = dict(n=a["n"], argmax_agreement=a["top"] / max(1, a["n"]),
                      max_abs_logit_diff=max(a["diffs"]),
                      mean_abs_logit_diff=sum(a["diffs"]) / len(a["diffs"]),
                      n_at_traced_len=a["n_at_traced"],
                      argmax_agreement_at_traced_len=(
                          a["top_at_traced"] / a["n_at_traced"] if a["n_at_traced"] else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="trained reranker dir")
    ap.add_argument("--out", required=True, help="output dir for the onnx files")
    ap.add_argument("--data", default="", help="rerank jsonl.gz for numeric verification")
    ap.add_argument("--verify-n", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--fixed-seq", action="store_true",
                    help="export with a FIXED seq length = --max-len (pad everything). Use if "
                         "dynamic-seq verification fails (traced sliding-window mask).")
    ap.add_argument("--per-tensor", action="store_true",
                    help="per-tensor instead of per-channel weight scales (per-channel Gather "
                         "support is shaky; use to isolate embedding-quantization damage)")
    ap.add_argument("--no-gather-quant", action="store_true",
                    help="quantize MatMul only, leaving the 41M-param embedding fp32 (will "
                         "almost certainly blow the size cap; for measurement only)")
    ap.add_argument("--keep-fp32", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fp32 = os.path.join(args.out, "model.onnx")
    int8 = os.path.join(args.out, "model_int8.onnx")

    print(f"[1/4] exporting fp32 ONNX (seq={'fixed ' + str(args.max_len) if args.fixed_seq else 'dynamic'})",
          flush=True)
    export_fp32(args.model, fp32, args.max_len, dynamic_seq=not args.fixed_seq)
    print(f"      fp32 onnx: {_mb(os.path.getsize(fp32))}", flush=True)

    print("[2/4] dynamic INT8 quantization", flush=True)
    quantize(fp32, int8, gather=not args.no_gather_quant,
             per_channel=not args.per_tensor)
    print(f"      int8 onnx: {_mb(os.path.getsize(int8))}", flush=True)

    print("[3/4] staged fidelity check on real decisions", flush=True)
    v = None
    if args.data:
        recs = _load_records(args.data, args.verify_n)
        traced = args.max_len if args.fixed_seq else args.max_len // 2
        v = verify(args.model, {"onnx_fp32": fp32, "onnx_int8": int8}, recs, args.max_len,
                   pad_to=args.max_len if args.fixed_seq else 0, traced_seq=traced)
        for k, r in v.items():
            extra = ""
            if r["n_at_traced_len"]:
                extra = (f"  |  at traced len {traced}: "
                         f"{r['argmax_agreement_at_traced_len']:.1%} (n={r['n_at_traced_len']})")
            print(f"      {k:10s} n={r['n']} argmax {r['argmax_agreement']:.1%}  "
                  f"|diff| mean {r['mean_abs_logit_diff']:.4f} max {r['max_abs_logit_diff']:.4f}"
                  f"{extra}", flush=True)

    print("[4/4] submission budget (measured on the COMPRESSED tarball, which is what the "
          "197.65625 MiB cap applies to)", flush=True)
    tok_bytes = sum(os.path.getsize(os.path.join(args.model, f))
                    for f in ("tokenizer.json", "tokenizer_config.json")
                    if os.path.exists(os.path.join(args.model, f)))
    ort_dir = None
    try:
        import onnxruntime
        ort_dir = os.path.dirname(onnxruntime.__file__)
    except Exception:
        pass
    ort_bytes = _dir_bytes(ort_dir) if ort_dir else 0
    # onnxruntime ships tests/tools/quantization/transformers helpers we never load at inference
    ORT_STRIP = ["*/test*", "*/tools/*", "*/transformers/*", "*/quantization/*",
                 "*/datasets/*", "__pycache__"]
    gz = {}
    gz["model_int8.onnx"] = _gz_bytes([int8])
    gz["tokenizer"] = _gz_bytes([os.path.join(args.model, f)
                                 for f in ("tokenizer.json", "tokenizer_config.json")
                                 if os.path.exists(os.path.join(args.model, f))])
    if ort_dir:
        gz["onnxruntime (stripped)"] = _gz_bytes([ort_dir], exclude=ORT_STRIP)
    gz["repo code+decks (allowance)"] = 3 * 1024 * 1024
    for k, b in gz.items():
        print(f"      {k:38s} {_mb(b)}", flush=True)
    total = sum(gz.values())
    print(f"      {'TOTAL (tar.gz)':38s} {_mb(total)} vs cap {_mb(TAR_CAP)} -> "
          f"{'FITS' if total < TAR_CAP else 'OVER'} "
          f"(headroom {_mb(TAR_CAP - total)})", flush=True)

    if not args.keep_fp32:
        os.remove(fp32)
    json.dump(dict(int8_bytes=os.path.getsize(int8), tokenizer_bytes=tok_bytes,
                   onnxruntime_bytes=ort_bytes, gz=gz, total_gz=total, cap=TAR_CAP, verify=v,
                   fixed_seq=args.fixed_seq, gather_quant=not args.no_gather_quant,
                   per_channel=not args.per_tensor),
              open(os.path.join(args.out, "export_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
